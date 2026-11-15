//! Validation-only identical-input probes. No Python dependency in native code.
use anyhow::{Result, bail, ensure};
use deepseek_ultrafast_strix::{
    math,
    runtime::Gpu,
    weights::{Store, Weights},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;

#[derive(Deserialize)]
struct Probe {
    label: String,
    #[serde(flatten)]
    op: Operation,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Operation {
    Router {
        name: String,
        input: Vec<f32>,
        bias: Vec<f32>,
        topk: usize,
        scale: f32,
    },
    Linear {
        name: String,
        input: Vec<f32>,
        round: bool,
    },
    Range {
        name: String,
        input: Vec<f32>,
        first: usize,
        rows: usize,
        round: bool,
    },
    Mix {
        prefix: String,
        input: Vec<f32>,
    },
    Rms {
        input: Vec<f32>,
        gamma: Vec<f32>,
        eps: f32,
    },
    Pre {
        input: Vec<f32>,
        pre: Vec<f32>,
    },
    Post {
        input: Vec<f32>,
        residual: Vec<f32>,
        post: Vec<f32>,
        comb: Vec<f32>,
    },
    Quant {
        input: Vec<f32>,
        bits: u8,
        group: usize,
        e4m3: bool,
    },
    Attention {
        query: Vec<f32>,
        kv: Vec<Vec<f32>>,
        sinks: Vec<f32>,
        dim: usize,
    },
    Engram {
        input: Vec<f32>,
        kv: Vec<f32>,
        q: Vec<f32>,
        k: Vec<f32>,
        dim: usize,
        eps: f32,
        #[serde(default)]
        gates_only: bool,
    },
    Rope {
        input: Vec<f32>,
        cos: Vec<f32>,
        sin: Vec<f32>,
        inverse: bool,
    },
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "numerical_probe <checkpoint> <requests.json> <new-output.json>"
    );
    ensure!(
        !Path::new(&args[2]).exists(),
        "refusing to overwrite probe evidence"
    );
    let requests: Vec<Probe> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let mut weights = Weights::new(
        Store::open(Path::new(&args[0]))?,
        Gpu::new()?,
        1024 * 1024 * 1024,
    )?;
    let mut results = serde_json::Map::new();
    for request in requests {
        ensure!(!results.contains_key(&request.label), "duplicate label");
        let result: Value = match request.op {
            Operation::Router {
                name,
                input,
                bias,
                topk,
                scale,
            } => {
                let logits = weights.linear(&name, &input, false)?;
                ensure!(
                    bias.len() == logits.len() && topk > 0 && topk <= bias.len(),
                    "router shape"
                );
                let routed = math::route(&logits, &bias, topk, scale);
                json!({"logits":logits, "indices":routed.iter().map(|&(i,_)|i).collect::<Vec<_>>(),
                    "weights":routed.iter().map(|&(_,w)|w).collect::<Vec<_>>()})
            }
            Operation::Linear { name, input, round } => {
                json!(weights.linear(&name, &input, round)?)
            }
            Operation::Range {
                name,
                input,
                first,
                rows,
                round,
            } => json!(weights.range(&name, &input, first, rows, round)?),
            Operation::Mix { prefix, input } => {
                let raw = weights.linear(&format!("{prefix}_fn"), &input, false)?;
                let inv = math::inv_rms(&input, 1e-20);
                let z: Vec<_> = raw.iter().map(|x| x * inv).collect();
                let scale = weights.store.small(&format!("{prefix}_scale"))?;
                let base = weights.store.small(&format!("{prefix}_base"))?;
                let mix = math::sinkhorn(&z, &scale, &base, 4, 20, 1e-6);
                json!({"raw":raw, "inv":inv, "z":z, "pre":mix.pre, "post":mix.post, "comb":mix.comb})
            }
            Operation::Rms { input, gamma, eps } => {
                ensure!(!input.is_empty() && input.len() == gamma.len(), "RMS shape");
                json!({"inv":math::inv_rms(&input, eps), "output":math::rms(&input, &gamma, eps)})
            }
            Operation::Pre { input, pre } => {
                ensure!(
                    !pre.is_empty() && input.len().is_multiple_of(pre.len()),
                    "pre shape"
                );
                json!(math::hc_pre(&input, &pre, input.len() / pre.len()))
            }
            Operation::Post {
                input,
                residual,
                post,
                comb,
            } => {
                ensure!(
                    !input.is_empty()
                        && residual.len() == post.len() * input.len()
                        && comb.len() == post.len() * post.len(),
                    "post shape"
                );
                json!(math::hc_post(
                    &input,
                    &residual,
                    &math::Mix {
                        pre: vec![],
                        post,
                        comb
                    }
                ))
            }
            Operation::Quant {
                mut input,
                bits,
                group,
                e4m3,
            } => {
                match bits {
                    8 => math::quant_fp8(&mut input)?,
                    4 => math::quant_fp4(&mut input, group, e4m3)?,
                    _ => bail!("quantization bits must be 4 or 8"),
                }
                json!(input)
            }
            Operation::Attention {
                query,
                kv,
                sinks,
                dim,
            } => {
                ensure!(
                    dim > 0
                        && query.len() == sinks.len() * dim
                        && kv.iter().all(|x| x.is_empty() || x.len() == dim),
                    "attention shape"
                );
                json!(math::attention(&query, &kv, &sinks, dim))
            }
            Operation::Engram {
                mut input,
                kv,
                q,
                k,
                dim,
                eps,
                gates_only,
            } => {
                ensure!(
                    dim > 0
                        && input.len().is_multiple_of(dim)
                        && kv.len() == input.len() + dim
                        && q.len() == input.len()
                        && k.len() == input.len(),
                    "Engram shape"
                );
                if gates_only {
                    json!(math::engram_gates(&input, &kv, &q, &k, dim, eps))
                } else {
                    math::engram(&mut input, &kv, &q, &k, dim, eps);
                    json!(input)
                }
            }
            Operation::Rope {
                mut input,
                cos,
                sin,
                inverse,
            } => {
                let dim = cos.len() * 2;
                ensure!(
                    dim > 0 && sin.len() == cos.len() && input.len().is_multiple_of(dim),
                    "RoPE shape"
                );
                math::rope(&mut input, dim, dim, &cos, &sin, inverse);
                json!(input)
            }
        };
        results.insert(request.label, result);
    }
    std::fs::write(&args[2], serde_json::to_vec(&results)?)?;
    Ok(())
}
