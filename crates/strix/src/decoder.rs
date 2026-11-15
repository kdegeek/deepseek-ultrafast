//! Batch-one, target-only native decoder for the pinned original checkpoint.
//! Host vector scheduling and bounded attention; every learned matrix projection
//! (including routing, compression, Engram, and output head) runs on AMD Vulkan.
use crate::{
    config::{Config, rope_table},
    engram::EngramConsts,
    math::{self, Mix},
    reference::Format,
    runtime::Gpu,
    weights::{Store, Weights},
};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    time::Instant,
};
use tokenizers::Tokenizer;

pub const REVISION: &str = "314113f8a7dadb388215396ba5fbee2c610a04d6";
pub const MAX_CONTEXT: usize = 128;

#[derive(Default)]
struct LayerState {
    window: Vec<Vec<f32>>,
    compressed: Vec<Vec<f32>>,
    keys: Vec<Vec<f32>>,
    pending_kv: Vec<Vec<f32>>,
    pending_scores: Vec<Vec<f32>>,
}

pub struct Decoder {
    cfg: Config,
    pub weights: Weights,
    tokenizer: Tokenizer,
    hashes: EngramConsts,
    tokens: Vec<u32>,
    states: Vec<LayerState>,
    rope: [(Vec<f32>, Vec<f32>); 2],
    topk: Vec<usize>,
    trace_dir: Option<PathBuf>,
    pub context: usize,
    pub initialization_seconds: f64,
    pub validated_text_tensors: usize,
}

impl Decoder {
    pub fn open(
        dir: &Path,
        cache_bytes: usize,
        context: usize,
        trace_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let started = Instant::now();
        ensure!(
            (1..=MAX_CONTEXT).contains(&context),
            "native context must be 1..={MAX_CONTEXT}; long-context selection is not enabled"
        );
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(dir.join("source-manifest.json"))?)?;
        ensure!(
            manifest["repo"] == "deepseek-ai/DeepSeek-V4.1-Flash"
                && manifest["revision"] == REVISION,
            "requires the pinned original checkpoint"
        );
        let raw = std::fs::read(dir.join("inference/config.json"))?;
        let config: Value = serde_json::from_slice(&raw)?;
        ensure!(
            config == serde_json::from_str::<Value>(include_str!("original-config.json"))?,
            "unsupported architecture/configuration; native decoder is pinned to original Flash"
        );
        let cfg: Config = serde_json::from_slice(&raw)?;
        let store = Store::open(dir)?;
        let validated_text_tensors = validate_text_tensors(&store, &cfg)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let hashes = EngramConsts::compute(dir, &cfg)?;
        let rope = [false, true].map(|compressed| {
            rope_table(
                cfg.rope_head_dim,
                context,
                if compressed { cfg.original_seq_len } else { 0 },
                if compressed {
                    cfg.compress_rope_theta
                } else {
                    cfg.rope_theta
                },
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
            )
        });
        let states = (0..cfg.n_layers).map(|_| LayerState::default()).collect();
        if let Some(path) = &trace_dir {
            std::fs::create_dir_all(path)?;
        }
        let weights = Weights::new(store, Gpu::new()?, cache_bytes)?;
        Ok(Self {
            cfg,
            weights,
            tokenizer,
            hashes,
            tokens: Vec::new(),
            states,
            rope,
            topk: Vec::new(),
            trace_dir,
            context,
            initialization_seconds: started.elapsed().as_secs_f64(),
            validated_text_tensors,
        })
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        let ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        ensure!(!ids.is_empty(), "empty prompt");
        ensure!(
            ids.iter()
                .all(|&t| (t as usize) < self.cfg.vocab_size && t < 129264),
            "image/control tokens for vision are unsupported"
        );
        Ok(ids)
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }

    fn trace(&self, label: &str, values: &[f32]) -> Result<()> {
        self.trace_value(label, &values)
    }

    fn trace_value<T: Serialize>(&self, label: &str, values: &T) -> Result<()> {
        if let Some(dir) = &self.trace_dir {
            std::fs::write(
                dir.join(format!("p{:03}-{label}.json", self.tokens.len() - 1)),
                serde_json::to_vec(values)?,
            )?;
        }
        Ok(())
    }

    fn norm(&mut self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let gamma = self.weights.store.small(name)?;
        ensure!(gamma.len() == x.len(), "norm shape {name}");
        Ok(math::rms(x, &gamma, self.cfg.norm_eps))
    }

    fn mixes(&mut self, x: &[f32], prefix: &str) -> Result<Mix> {
        let mut z = self.weights.linear(&format!("{prefix}_fn"), x, false)?;
        let scale = math::inv_rms(x, self.cfg.norm_eps);
        for v in &mut z {
            *v *= scale;
        }
        let scale = self.weights.store.small(&format!("{prefix}_scale"))?;
        let base = self.weights.store.small(&format!("{prefix}_base"))?;
        let mix = math::sinkhorn(
            &z,
            &scale,
            &base,
            self.cfg.hc_mult,
            self.cfg.hc_sinkhorn_iters,
            self.cfg.hc_eps,
        );
        self.trace(&format!("{prefix}-pre"), &mix.pre)?;
        self.trace(&format!("{prefix}-post"), &mix.post)?;
        self.trace(&format!("{prefix}-comb"), &mix.comb)?;
        Ok(mix)
    }

    fn rotary(&self, x: &mut [f32], layer: usize, position: usize, head_dim: usize, inverse: bool) {
        let rd = self.cfg.rope_head_dim;
        let (cos, sin) = &self.rope[usize::from(self.cfg.compress_ratios[layer] > 0)];
        let offset = position * rd / 2;
        math::rope(
            x,
            head_dim,
            rd,
            &cos[offset..offset + rd / 2],
            &sin[offset..offset + rd / 2],
            inverse,
        );
    }

    fn engram(&mut self, x: &mut [f32], layer: usize) -> Result<()> {
        let Some(index) = self.cfg.engram_index(layer) else {
            return Ok(());
        };
        let (n, heads, dim) = (
            self.cfg.engram_max_ngram_size,
            self.cfg.engram_n_heads,
            self.cfg.dim,
        );
        let mut rolling = 0i64;
        let mut embedding = Vec::new();
        let mut hash_ids = Vec::new();
        for shift in 0..n {
            let token = if shift < self.tokens.len() {
                self.hashes.token_map[self.tokens[self.tokens.len() - 1 - shift] as usize]
            } else {
                self.hashes.pad_id
            };
            rolling ^= i64::from(token) * self.hashes.multipliers[index * n + shift];
            if shift > 0 {
                for head in 0..heads {
                    let col = index * (n - 1) * heads + (shift - 1) * heads + head;
                    let row =
                        (rolling % self.hashes.primes[col] + self.hashes.offsets[col]) as usize;
                    hash_ids.push(row);
                    embedding.extend(
                        self.weights
                            .store
                            .row(&format!("layers.{layer}.engram.embed.weight"), row)?,
                    );
                }
            }
        }
        self.trace_value(&format!("l{layer:02}-engram-hashes"), &hash_ids)?;
        let kv = self.weights.linear(
            &format!("layers.{layer}.engram.wkv.weight"),
            &embedding,
            true,
        )?;
        let q = self
            .weights
            .store
            .small(&format!("layers.{layer}.engram.q_weight"))?;
        let k = self
            .weights
            .store
            .small(&format!("layers.{layer}.engram.k_weight"))?;
        math::engram(x, &kv, &q, &k, dim, self.cfg.norm_eps);
        self.trace(&format!("l{layer:02}-engram"), x)
    }

    fn compressed(
        &mut self,
        x: &[f32],
        qr: &[f32],
        layer: usize,
        position: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let ratio = self.cfg.compress_ratios[layer];
        let source = self.cfg.kv_source_of(layer).context("missing KV source")?;
        let p = format!("layers.{layer}.attn");
        let mut latent = None;
        if self.cfg.is_kv_source(layer) {
            let value =
                self.weights
                    .linear(&format!("{p}.compressor.wkv.weight"), x, ratio == 1)?;
            if ratio == 1 {
                latent = Some(self.norm(&value, &format!("{p}.compressor.norm.weight"))?);
            } else {
                let score =
                    self.weights
                        .linear(&format!("{p}.compressor.wgate.weight"), x, false)?;
                let state = &mut self.states[layer];
                state.pending_kv.push(value);
                state.pending_scores.push(score);
                if (position + 1).is_multiple_of(ratio) {
                    let mut pooled = vec![0.; self.cfg.head_dim];
                    for (i, v) in pooled.iter_mut().enumerate() {
                        let mut score: Vec<_> = state.pending_scores.iter().map(|s| s[i]).collect();
                        math::softmax(&mut score);
                        *v = math::bf16(
                            score
                                .iter()
                                .zip(&state.pending_kv)
                                .map(|(s, k)| s * k[i])
                                .sum(),
                        );
                    }
                    state.pending_kv.clear();
                    state.pending_scores.clear();
                    latent = Some(self.norm(&pooled, &format!("{p}.compressor.norm.weight"))?);
                }
            }
        }
        let len = (position + 1) / ratio;
        if self.cfg.is_index_source(layer) {
            if len == 0 {
                self.topk.clear();
            } else {
                if let Some(latent) = &latent {
                    let key =
                        self.weights
                            .linear(&format!("{p}.indexer.wk.weight"), latent, true)?;
                    let mut key = self.norm(&key, &format!("{p}.indexer.k_norm.weight"))?;
                    self.rotary(
                        &mut key,
                        layer,
                        position + 1 - ratio,
                        self.cfg.index_head_dim,
                        false,
                    );
                    math::quant_fp4(&mut key, 32, false)?;
                    self.states[source].keys.push(key);
                }
                let mut query =
                    self.weights
                        .linear(&format!("{p}.indexer.wq_b.weight"), qr, true)?;
                self.rotary(&mut query, layer, position, self.cfg.index_head_dim, false);
                math::quant_fp4(&mut query, 32, false)?;
                let mut weights =
                    self.weights
                        .linear(&format!("{p}.indexer.weights_proj.weight"), x, true)?;
                let scale = (self.cfg.index_head_dim as f32).sqrt().recip()
                    * (self.cfg.index_n_heads as f32).sqrt().recip();
                for v in &mut weights {
                    *v = math::bf16(*v * scale);
                }
                ensure!(
                    self.states[source].keys.len() == len,
                    "index KV length mismatch"
                );
                let mut scores = Vec::with_capacity(len);
                for key in &self.states[source].keys {
                    // Official index einsum and elementwise weighted scores are BF16.
                    let sum = query
                        .chunks_exact(self.cfg.index_head_dim)
                        .zip(&weights)
                        .map(|(q, w)| {
                            math::bf16(
                                math::bf16(q.iter().zip(key).map(|(a, b)| a * b).sum()).max(0.) * w,
                            )
                        })
                        .sum();
                    scores.push(math::bf16(sum));
                }
                self.trace(&format!("l{layer:02}-index-scores"), &scores)?;
                ensure!(
                    len <= self.cfg.index_topk
                        && len.div_ceil(self.cfg.candidate_block_size)
                            <= self.cfg.candidate_topk_blocks,
                    "context requires unimplemented candidate pruning"
                );
                // Every reachable position is selected under the enforced context cap.
                // This is exactly topk(len), sorted by position, including candidate masks.
                self.topk = (0..len).collect();
            }
        }
        if let Some(mut latent) = latent {
            self.rotary(
                &mut latent,
                layer,
                position + 1 - ratio,
                self.cfg.head_dim,
                false,
            );
            math::quant_fp4(&mut latent, 16, true)?;
            self.states[source].compressed.push(latent);
        }
        ensure!(
            self.states[source].compressed.len() == len,
            "compressed KV length mismatch"
        );
        Ok(self
            .topk
            .iter()
            .map(|&i| self.states[source].compressed[i].clone())
            .collect())
    }

    fn attention(&mut self, x: &[f32], layer: usize, position: usize) -> Result<Vec<f32>> {
        let p = format!("layers.{layer}.attn");
        let qr = self.weights.linear(&format!("{p}.wq_a.weight"), x, true)?;
        let qr = self.norm(&qr, &format!("{p}.q_norm.weight"))?;
        let mut q = self
            .weights
            .linear(&format!("{p}.wq_b.weight"), &qr, true)?;
        self.rotary(&mut q, layer, position, self.cfg.head_dim, false);
        let kv = self.weights.linear(&format!("{p}.wkv.weight"), x, true)?;
        let mut kv = self.norm(&kv, &format!("{p}.kv_norm.weight"))?;
        self.rotary(&mut kv, layer, position, self.cfg.head_dim, false);
        math::quant_fp8(&mut kv)?;
        math::round_bf16(&mut kv);
        self.states[layer].window.push(kv);
        let mut values = if position == 0 {
            self.states[layer].window.clone()
        } else {
            (position + 1..self.cfg.window_size)
                .map(|_| Vec::new())
                .chain(self.states[layer].window.iter().cloned())
                .collect()
        };
        if self.cfg.compress_ratios[layer] > 0 {
            values.extend(self.compressed(x, &qr, layer, position)?);
        }
        let sinks = self.weights.store.small(&format!("{p}.attn_sink"))?;
        let mut output = math::attention(&q, &values, &sinks, self.cfg.head_dim);
        self.rotary(&mut output, layer, position, self.cfg.head_dim, true);
        math::quant_fp8(&mut output)?;
        math::round_bf16(&mut output);
        let mut grouped = Vec::new();
        for (group, chunk) in output
            .chunks_exact(output.len() / self.cfg.o_groups)
            .enumerate()
        {
            grouped.extend(self.weights.range(
                &format!("{p}.wo_a.weight"),
                chunk,
                group * self.cfg.o_lora_rank,
                self.cfg.o_lora_rank,
                true,
            )?);
        }
        self.weights
            .linear(&format!("{p}.wo_b.weight"), &grouped, true)
    }

    fn expert(&mut self, x: &[f32], prefix: &str, weight: f32) -> Result<Vec<f32>> {
        let gate = self
            .weights
            .linear(&format!("{prefix}.w1.weight"), x, true)?;
        let up = self
            .weights
            .linear(&format!("{prefix}.w3.weight"), x, true)?;
        let limit = self.cfg.swiglu_limit;
        let hidden: Vec<_> = gate
            .iter()
            .zip(up)
            .map(|(g, u)| {
                let (g, u) = if limit > 0.0 {
                    (g.min(limit), u.clamp(-limit, limit))
                } else {
                    (*g, u)
                };
                math::bf16((g * math::sigmoid(g)) * u * weight)
            })
            .collect();
        self.weights
            .linear(&format!("{prefix}.w2.weight"), &hidden, true)
    }

    fn moe(&mut self, x: &[f32], layer: usize) -> Result<Vec<f32>> {
        let p = format!("layers.{layer}.ffn");
        let logits = self.weights.linear(&format!("{p}.gate.weight"), x, false)?;
        let bias = self.weights.store.small(&format!("{p}.gate.bias"))?;
        let mut routed = math::route(
            &logits,
            &bias,
            self.cfg.n_activated_experts,
            self.cfg.route_scale,
        );
        self.trace(
            &format!("l{layer:02}-experts"),
            &routed.iter().map(|&(i, _)| i as f32).collect::<Vec<_>>(),
        )?;
        let mut y = vec![0.; x.len()];
        routed.sort_by_key(|&(i, _)| i);
        for (i, weight) in routed {
            let out = self.expert(x, &format!("{p}.experts.{i}"), weight)?;
            for (a, b) in y.iter_mut().zip(out) {
                *a += b;
            }
        }
        let shared = self.expert(x, &format!("{p}.shared_experts"), 1.)?;
        for (a, b) in y.iter_mut().zip(shared) {
            *a = math::bf16(*a + b);
        }
        Ok(y)
    }

    pub fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let position = self.tokens.len();
        ensure!(
            position < self.context && token < 129264 && (token as usize) < self.cfg.vocab_size,
            "token/context outside native text bounds"
        );
        self.tokens.push(token);
        let embedded = self.weights.store.row("embed.weight", token as usize)?;
        let mut x = embedded.repeat(self.cfg.hc_mult);
        let mut pre = vec![0.; self.cfg.hc_mult];
        pre[0] = 1.;
        for layer in 0..self.cfg.n_layers {
            self.engram(&mut x, layer)?;
            let mix = self.mixes(&x, &format!("layers.{layer}.hc_attn"))?;
            let collapsed = math::hc_pre(&x, &pre, self.cfg.dim);
            let input = self.norm(&collapsed, &format!("layers.{layer}.attn_norm.weight"))?;
            self.trace(&format!("l{layer:02}-attn-input"), &input)?;
            let out = self.attention(&input, layer, position)?;
            self.trace(&format!("l{layer:02}-attn"), &out)?;
            x = math::hc_post(&out, &x, &mix);
            self.trace(&format!("l{layer:02}-post-attn"), &x)?;
            let ffn_mix = self.mixes(&x, &format!("layers.{layer}.hc_ffn"))?;
            let collapsed = math::hc_pre(&x, &mix.pre, self.cfg.dim);
            let input = self.norm(&collapsed, &format!("layers.{layer}.ffn_norm.weight"))?;
            self.trace(&format!("l{layer:02}-ffn-input"), &input)?;
            let out = self.moe(&input, layer)?;
            self.trace(&format!("l{layer:02}-ffn"), &out)?;
            x = math::hc_post(&out, &x, &ffn_mix);
            pre = ffn_mix.pre;
            self.trace(&format!("l{layer:02}-output"), &x)?;
            if self.trace_dir.is_some() {
                eprintln!("position {position} layer {layer} complete");
            }
        }
        let collapsed = math::hc_pre(&x, &pre, self.cfg.dim);
        let x = self.norm(&collapsed, "norm.weight")?;
        self.trace("head-input", &x)?;
        let logits = self.weights.linear("head.weight", &x, false)?;
        self.trace("logits", &logits)?;
        Ok(logits)
    }
}

fn validate_text_tensors(store: &Store, c: &Config) -> Result<usize> {
    let mut used = BTreeSet::new();
    let mut expect = |name: String, shape: &[usize], dtype: Dtype| -> Result<()> {
        store.expect(&name, shape, dtype)?;
        used.insert(name);
        Ok(())
    };
    for name in ["embed.weight", "head.weight"] {
        expect(name.into(), &[c.vocab_size, c.dim], Dtype::BF16)?;
    }
    expect("norm.weight".into(), &[c.dim], Dtype::BF16)?;
    for l in 0..c.n_layers {
        let p = format!("layers.{l}");
        let mut mats = vec![
            (
                format!("{p}.attn.wq_a.weight"),
                c.q_lora_rank,
                c.dim,
                Format::Fp8,
            ),
            (
                format!("{p}.attn.wq_b.weight"),
                c.n_heads * c.head_dim,
                c.q_lora_rank,
                Format::Fp8,
            ),
            (
                format!("{p}.attn.wkv.weight"),
                c.head_dim,
                c.dim,
                Format::Fp8,
            ),
            (
                format!("{p}.attn.wo_a.weight"),
                c.o_groups * c.o_lora_rank,
                c.n_heads * c.head_dim / c.o_groups,
                Format::Fp8,
            ),
            (
                format!("{p}.attn.wo_b.weight"),
                c.dim,
                c.o_groups * c.o_lora_rank,
                Format::Fp8,
            ),
            (
                format!("{p}.ffn.gate.weight"),
                c.n_routed_experts,
                c.dim,
                Format::Bf16,
            ),
        ];
        for (name, dim) in [
            ("attn_norm", c.dim),
            ("ffn_norm", c.dim),
            ("attn.q_norm", c.q_lora_rank),
            ("attn.kv_norm", c.head_dim),
        ] {
            expect(format!("{p}.{name}.weight"), &[dim], Dtype::BF16)?;
        }
        expect(format!("{p}.attn.attn_sink"), &[c.n_heads], Dtype::F32)?;
        expect(
            format!("{p}.ffn.gate.bias"),
            &[c.n_routed_experts],
            Dtype::F32,
        )?;
        for sub in ["attn", "ffn"] {
            mats.push((
                format!("{p}.hc_{sub}_fn"),
                (2 + c.hc_mult) * c.hc_mult,
                c.hc_mult * c.dim,
                Format::F32,
            ));
            expect(
                format!("{p}.hc_{sub}_base"),
                &[(2 + c.hc_mult) * c.hc_mult],
                Dtype::F32,
            )?;
            expect(format!("{p}.hc_{sub}_scale"), &[3], Dtype::F32)?;
        }
        for expert in (0..c.n_routed_experts)
            .map(|e| format!("experts.{e}"))
            .chain(["shared_experts".to_owned()])
        {
            let format = if expert == "shared_experts" {
                Format::Fp8
            } else {
                Format::Fp4
            };
            for w in ["w1", "w2", "w3"] {
                let (rows, cols) = if w == "w2" {
                    (c.dim, c.moe_inter_dim)
                } else {
                    (c.moe_inter_dim, c.dim)
                };
                mats.push((format!("{p}.ffn.{expert}.{w}.weight"), rows, cols, format));
            }
        }
        if c.is_kv_source(l) {
            mats.push((
                format!("{p}.attn.compressor.wkv.weight"),
                c.head_dim,
                c.dim,
                Format::Bf16,
            ));
            if c.compress_ratios[l] > 1 {
                mats.push((
                    format!("{p}.attn.compressor.wgate.weight"),
                    c.head_dim,
                    c.dim,
                    Format::Bf16,
                ));
            }
            expect(
                format!("{p}.attn.compressor.norm.weight"),
                &[c.head_dim],
                Dtype::BF16,
            )?;
            mats.push((
                format!("{p}.attn.indexer.wk.weight"),
                c.index_head_dim,
                c.head_dim,
                Format::Bf16,
            ));
            expect(
                format!("{p}.attn.indexer.k_norm.weight"),
                &[c.index_head_dim],
                Dtype::BF16,
            )?;
        }
        if c.is_index_source(l) {
            mats.push((
                format!("{p}.attn.indexer.wq_b.weight"),
                c.index_n_heads * c.index_head_dim,
                c.q_lora_rank,
                Format::Fp8,
            ));
            mats.push((
                format!("{p}.attn.indexer.weights_proj.weight"),
                c.index_n_heads,
                c.dim,
                Format::Bf16,
            ));
        }
        if let Some(e) = c.engram_index(l) {
            expect(
                format!("{p}.engram.embed.weight"),
                &[c.engram_num_embeddings[e], c.engram_head_dim],
                Dtype::F8_E4M3,
            )?;
            expect(
                format!("{p}.engram.embed.scale"),
                &[c.engram_num_embeddings[e], c.engram_head_dim / 32],
                Dtype::F8_E8M0,
            )?;
            mats.push((
                format!("{p}.engram.wkv.weight"),
                c.dim * (c.hc_mult + 1),
                (c.engram_max_ngram_size - 1) * c.engram_n_heads * c.engram_head_dim,
                Format::Fp8,
            ));
            for n in ["q_weight", "k_weight"] {
                expect(format!("{p}.engram.{n}"), &[c.hc_mult, c.dim], Dtype::BF16)?;
            }
        }
        for (name, rows, cols, format) in mats {
            let dtype = match format {
                Format::F32 => Dtype::F32,
                Format::Bf16 => Dtype::BF16,
                Format::Fp8 => Dtype::F8_E4M3,
                Format::Fp4 => Dtype::I8,
            };
            expect(
                name.clone(),
                &[rows, cols / if format == Format::Fp4 { 2 } else { 1 }],
                dtype,
            )?;
            if matches!(format, Format::Fp8 | Format::Fp4) {
                expect(
                    name.strip_suffix("weight").unwrap().to_owned() + "scale",
                    &[
                        if format == Format::Fp4 {
                            rows
                        } else {
                            rows.div_ceil(32)
                        },
                        cols / 32,
                    ],
                    Dtype::F8_E8M0,
                )?;
            }
        }
    }
    for name in store.tensors.keys().filter(|n| n.starts_with("layers.")) {
        ensure!(
            used.contains(name) || name.ends_with(".ffn.gate.bias_vl"),
            "unhandled text tensor {name}"
        );
    }
    Ok(used.len())
}

#[derive(Serialize)]
pub struct Step {
    pub position: usize,
    pub input_token: u32,
    pub next_token: u32,
    pub seconds: f64,
    pub top_logits: Vec<(usize, f32)>,
}

pub fn generate(
    dir: &Path,
    prompt: &str,
    max_tokens: usize,
    cache_bytes: usize,
    context: usize,
    trace_dir: Option<PathBuf>,
) -> Result<Value> {
    ensure!(
        max_tokens > 0 && max_tokens <= MAX_CONTEXT,
        "invalid generation limit"
    );
    let mut decoder = Decoder::open(dir, cache_bytes, context, trace_dir)?;
    let prompt_tokens = decoder.encode(prompt)?;
    ensure!(
        prompt_tokens.len() + max_tokens - 1 <= context,
        "prompt and generation exceed native context"
    );
    let started = Instant::now();
    let mut generated = Vec::new();
    let mut steps = Vec::new();
    let mut next = 0;
    let mut stop = "length";
    for position in 0..prompt_tokens.len() + max_tokens - 1 {
        let input = prompt_tokens.get(position).copied().unwrap_or(next);
        let timer = Instant::now();
        let logits = decoder.forward(input)?;
        let mut order: Vec<_> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        next = order[0] as u32;
        ensure!(
            next < 129264,
            "model selected unsupported vision token {next}"
        );
        let step = Step {
            position,
            input_token: input,
            next_token: next,
            seconds: timer.elapsed().as_secs_f64(),
            top_logits: order[..10].iter().map(|&i| (i, logits[i])).collect(),
        };
        eprintln!(
            "position {position}, input {input}, next {next}, {:.3}s",
            step.seconds
        );
        steps.push(step);
        if position + 1 >= prompt_tokens.len() {
            generated.push(next);
            if next == 1 {
                stop = "eos";
                break;
            }
        }
    }
    let visible = generated.strip_suffix(&[1]).unwrap_or(&generated);
    let text = decoder.decode(visible)?;
    let state_capacity_bytes: usize = decoder
        .states
        .iter()
        .map(|s| {
            [
                &s.window,
                &s.compressed,
                &s.keys,
                &s.pending_kv,
                &s.pending_scores,
            ]
            .iter()
            .map(|v| {
                v.capacity() * std::mem::size_of::<Vec<f32>>()
                    + v.iter().map(|row| row.capacity() * 4).sum::<usize>()
            })
            .sum::<usize>()
        })
        .sum();
    Ok(
        json!({"mode":"experimental native Rust/Vulkan target-only greedy text","oracle_status":"experimental; independent acceptance harness reports prompt-specific token parity; broad logit equivalence is not established","revision":REVISION,"checkpoint":dir,"device":decoder.weights.gpu.info,"prompt":prompt,"prompt_tokens":prompt_tokens,"generated_tokens":generated,"text":text,"stop_reason":stop,"initialization_seconds":decoder.initialization_seconds,"elapsed_seconds_including_prefill":started.elapsed().as_secs_f64(),"steps":steps,"weight_cache":decoder.weights.stats,"checkpoint_read_bytes":decoder.weights.store.read_bytes,"peak_checkpoint_read_bytes":decoder.weights.store.peak_read_bytes,"persistent_kv_host_capacity_bytes":state_capacity_bytes,"validated_text_tensors":decoder.validated_text_tensors,"context_limit":context,"host_operations":"tokenizer, Engram hashes, activation quantization, norms, RoPE, hyperconnections, routing selection, bounded attention; all learned matrix projections execute on AMD Vulkan","memory_note":"Weight cache counts actual packed Vulkan buffers including scale padding. Host weight staging bound and persistent KV capacity reported separately. Driver allocation/cache, process working set, transient activations, tokenizer/catalog and other processes are not included in cache budget."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires AMD Vulkan; independent five-layer CPU checkpoint oracle"]
    fn amd_reduced_decoder_oracle() -> Result<()> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reduced");
        let cfg = Config::load(&dir.join("config.json"))?;
        let expected: Value = serde_json::from_slice(&std::fs::read(dir.join("expected.json"))?)?;
        let hashes: EngramConsts = serde_json::from_value(expected["hashes"].clone())?;
        let rope = [false, true].map(|compressed| {
            rope_table(
                cfg.rope_head_dim,
                8,
                if compressed { cfg.original_seq_len } else { 0 },
                if compressed {
                    cfg.compress_rope_theta
                } else {
                    cfg.rope_theta
                },
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
            )
        });
        let states = (0..cfg.n_layers).map(|_| LayerState::default()).collect();
        let mut decoder = Decoder {
            weights: Weights::new(Store::open(&dir)?, Gpu::new()?, 64 * 1024 * 1024)?,
            cfg,
            hashes,
            rope,
            states,
            tokenizer: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            tokens: Vec::new(),
            topk: Vec::new(),
            trace_dir: None,
            context: 8,
            initialization_seconds: 0.,
            validated_text_tensors: 0,
        };
        for (position, token) in expected["tokens"].as_array().unwrap().iter().enumerate() {
            let actual = decoder.forward(token.as_u64().unwrap() as u32)?;
            let reference = expected["logits"][position].as_array().unwrap();
            let atol = expected["atol"].as_f64().unwrap() as f32;
            let rtol = expected["rtol"].as_f64().unwrap() as f32;
            for (i, (&a, e)) in actual.iter().zip(reference).enumerate() {
                let e = e.as_f64().unwrap() as f32;
                ensure!(
                    a.is_finite() && (a - e).abs() <= atol + rtol * e.abs(),
                    "reduced position {position}, logit {i}: native {a}, CPU {e}"
                );
            }
            let greedy = actual
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            ensure!(
                greedy == expected["greedy"][position].as_u64().unwrap() as usize,
                "reduced greedy mismatch"
            );
        }
        ensure!(
            decoder.weights.stats.peak_resident_bytes <= decoder.weights.stats.budget_bytes,
            "weight budget exceeded"
        );
        Ok(())
    }
}
