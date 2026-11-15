//! Scalar scheduling/vector equations from the pinned official model. All model
//! weight projections are dispatched by weights.rs; there is no host GEMM path.
use crate::reference::{decode_e2m1, decode_e4m3};
use anyhow::{Result, ensure};

pub fn bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}
pub fn round_bf16(x: &mut [f32]) {
    for v in x {
        *v = bf16(*v);
    }
}
pub fn e8m0(code: u8) -> f32 {
    crate::reference::decode_e8m0(code)
}

fn encode(x: f32, max: u8, decode: fn(u8) -> f32) -> f32 {
    let a = x.abs();
    let mut lo = 0;
    let mut hi = max;
    while lo < hi {
        let middle = (lo + hi) / 2;
        let midpoint = (decode(middle) + decode(middle + 1)) * 0.5;
        if a > midpoint || (a == midpoint && middle & 1 != 0) {
            lo = middle + 1;
        } else {
            hi = middle;
        }
    }
    decode(lo).copysign(x)
}

fn pow2_scale(max: f32, reciprocal: f32) -> f32 {
    let bits = (max * reciprocal).to_bits();
    f32::from_bits(((bits >> 23) + u32::from(bits & 0x7f_ffff != 0)) << 23)
}

pub fn quant_fp8(x: &mut [f32]) -> Result<()> {
    ensure!(
        !x.is_empty() && x.len().is_multiple_of(32) && x.iter().all(|v| v.is_finite()),
        "invalid FP8 activation"
    );
    for block in x.as_chunks_mut::<32>().0 {
        let max = block.iter().map(|v| v.abs()).fold(1e-4, f32::max);
        let scale = pow2_scale(max, 1.0 / 448.0);
        for v in block {
            *v = encode(*v / scale, 126, decode_e4m3) * scale;
        }
    }
    Ok(())
}

pub fn quant_fp4(x: &mut [f32], group: usize, e4m3_scale: bool) -> Result<()> {
    ensure!(
        matches!(group, 16 | 32)
            && !x.is_empty()
            && x.len().is_multiple_of(group)
            && x.iter().all(|v| v.is_finite()),
        "invalid FP4 activation"
    );
    for block in x.chunks_exact_mut(group) {
        let max = block.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let scale = if e4m3_scale {
            encode(max.max(6.0 / 512.0) / 6.0, 126, decode_e4m3)
        } else {
            pow2_scale(max.max(6.0 * 2f32.powi(-126)), 1.0 / 6.0)
        };
        for v in block {
            *v = bf16(encode(*v / scale, 7, decode_e2m1) * scale);
        }
    }
    Ok(())
}

pub fn inv_rms(x: &[f32], eps: f32) -> f32 {
    fn square_sum(x: &[f32]) -> f32 {
        if x.len() <= 32 {
            x.iter().map(|v| v * v).sum()
        } else {
            let (a, b) = x.split_at(x.len() / 2);
            square_sum(a) + square_sum(b)
        }
    }
    // FP32 pairwise reduction avoids a 20,480-element serial accumulation;
    // the reference's device reduction is also parallel rather than serial.
    (square_sum(x) / x.len() as f32 + eps).sqrt().recip()
}
pub fn rms(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let scale = inv_rms(x, eps);
    x.iter()
        .zip(gamma)
        .map(|(v, g)| bf16(v * scale * g))
        .collect()
}
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Original Flash's temperature-one sqrt-softplus router, in top-k score order.
pub fn route(logits: &[f32], bias: &[f32], topk: usize, route_scale: f32) -> Vec<(usize, f32)> {
    let scores: Vec<_> = logits
        .iter()
        .map(|&v| {
            if v > 20. {
                v.sqrt()
            } else {
                v.exp().ln_1p().sqrt()
            }
        })
        .collect();
    let mut indices: Vec<_> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| {
        (scores[b] + bias[b])
            .total_cmp(&(scores[a] + bias[a]))
            .then(a.cmp(&b))
    });
    indices.truncate(topk);
    let sum = indices.iter().map(|&i| scores[i]).sum::<f32>() + 1e-20;
    indices
        .into_iter()
        .map(|i| (i, scores[i] / sum * route_scale))
        .collect()
}

/// Pinned Engram per-stream normalization and signed-square-root gate.
pub fn engram_gates(x: &[f32], kv: &[f32], q: &[f32], k: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let hc = x.len() / dim;
    let softmax_scale = (dim as f64).sqrt().recip() as f32;
    (0..hc)
        .map(|h| {
            let stream = &x[h * dim..(h + 1) * dim];
            let key = &kv[h * dim..(h + 1) * dim];
            let rstd = inv_rms(stream, eps) * inv_rms(key, eps);
            let products: Vec<_> = stream
                .iter()
                .zip(key)
                .enumerate()
                .map(|(i, (a, b))| a * (q[h * dim + i] * k[h * dim + i]) * b)
                .collect();
            // Avoid a long serial FP32 reduction. The original layer-14 gate
            // is calibrated against HIP and CPU f64 in engram_variance.py.
            let dot = pairwise_sum(&products) * rstd * softmax_scale;
            sigmoid(dot.abs().max(1e-6).sqrt().copysign(dot))
        })
        .collect()
}

fn pairwise_sum(x: &[f32]) -> f32 {
    if x.len() <= 32 {
        x.iter().sum()
    } else {
        let (left, right) = x.split_at(x.len() / 2);
        pairwise_sum(left) + pairwise_sum(right)
    }
}

pub fn engram(x: &mut [f32], kv: &[f32], q: &[f32], k: &[f32], dim: usize, eps: f32) {
    let hc = x.len() / dim;
    let gates = engram_gates(x, kv, q, k, dim, eps);
    for (h, gate) in gates.into_iter().enumerate() {
        let stream = &mut x[h * dim..(h + 1) * dim];
        for (i, v) in stream.iter_mut().enumerate() {
            *v = bf16(*v + gate * kv[hc * dim + i]);
        }
    }
}
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in x.iter_mut() {
        *v = (*v - max).exp();
    }
    let sum: f32 = x.iter().sum();
    for v in x {
        *v /= sum;
    }
}

pub fn rope(
    x: &mut [f32],
    head_dim: usize,
    rope_dim: usize,
    cos: &[f32],
    sin: &[f32],
    inverse: bool,
) {
    for head in x.chunks_exact_mut(head_dim) {
        for (i, pair) in head[head_dim - rope_dim..]
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .enumerate()
        {
            let (a, b) = (pair[0], pair[1]);
            let s = if inverse { -sin[i] } else { sin[i] };
            pair[0] = bf16(a * cos[i] - b * s);
            pair[1] = bf16(a * s + b * cos[i]);
        }
    }
}

pub struct Mix {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
}
pub fn sinkhorn(
    z: &[f32],
    scale: &[f32],
    base: &[f32],
    n: usize,
    iterations: usize,
    eps: f32,
) -> Mix {
    let pre = (0..n)
        .map(|i| sigmoid(z[i] * scale[0] + base[i]) + eps)
        .collect();
    let post = (n..2 * n)
        .map(|i| 2.0 * sigmoid(z[i] * scale[1] + base[i]))
        .collect();
    let mut comb: Vec<_> = (2 * n..z.len())
        .map(|i| z[i] * scale[2] + base[i])
        .collect();
    for row in comb.chunks_exact_mut(n) {
        softmax(row);
        for v in row {
            *v += eps;
        }
    }
    for iteration in 0..iterations {
        if iteration > 0 {
            for row in comb.chunks_exact_mut(n) {
                let sum = row.iter().sum::<f32>() + eps;
                for v in row {
                    *v /= sum;
                }
            }
        }
        for col in 0..n {
            let sum = (0..n).map(|row| comb[row * n + col]).sum::<f32>() + eps;
            for row in 0..n {
                comb[row * n + col] /= sum;
            }
        }
    }
    Mix { pre, post, comb }
}
pub fn hc_pre(x: &[f32], pre: &[f32], dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            bf16(
                pre.iter()
                    .enumerate()
                    .map(|(h, p)| x[h * dim + i] * p)
                    .sum(),
            )
        })
        .collect()
}
pub fn hc_post(x: &[f32], residual: &[f32], mix: &Mix) -> Vec<f32> {
    let (dim, n) = (x.len(), mix.post.len());
    (0..n * dim)
        .map(|i| {
            let (out, d) = (i / dim, i % dim);
            // comb's first axis is SOURCE, second axis DESTINATION (official sum dim=2).
            bf16(
                mix.post[out] * x[d]
                    + (0..n)
                        .map(|src| mix.comb[src * n + out] * residual[src * dim + d])
                        .sum::<f32>(),
            )
        })
        .collect()
}

/// Reference-compatible online softmax, including BF16 probability staging per
/// 64-key block and a learned sink with zero value. KV ordering is significant.
pub fn attention(q: &[f32], kv: &[Vec<f32>], sinks: &[f32], dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; q.len()];
    // The pinned model computes head_dim**-0.5 as a Python scalar, then the
    // kernel multiplies FP32 scores by it. Division by an FP32 sqrt has a
    // different rounding boundary (and can change the BF16 probability).
    let softmax_scale = (dim as f64).sqrt().recip() as f32;
    for (h, query) in q.chunks_exact(dim).enumerate() {
        let mut maximum = -1e30f32;
        let mut denominator = 0.0;
        let numerator = &mut output[h * dim..(h + 1) * dim];
        for keys in kv.chunks(64) {
            let mut scores: Vec<_> = keys
                .iter()
                .map(|key| {
                    if key.is_empty() {
                        f32::NEG_INFINITY
                    } else {
                        query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * softmax_scale
                    }
                })
                .collect();
            let new_max = scores.iter().copied().fold(maximum, f32::max);
            let correction = (maximum - new_max).exp();
            for v in &mut scores {
                *v = (*v - new_max).exp();
            }
            denominator = denominator * correction + scores.iter().sum::<f32>();
            for (d, v) in numerator.iter_mut().enumerate() {
                *v = *v * correction
                    + scores
                        .iter()
                        .zip(keys)
                        .map(|(p, k)| bf16(*p) * k.get(d).copied().unwrap_or(0.0))
                        .sum::<f32>();
            }
            maximum = new_max;
        }
        let final_max = maximum.max(sinks[h]);
        let correction = (maximum - final_max).exp();
        denominator = denominator * correction + (sinks[h] - final_max).exp();
        for v in numerator {
            *v = bf16(*v * correction / denominator.max(f32::MIN_POSITIVE));
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attention_scalar_multiply_bf16_boundary() {
        // Pinned kernel.py: score = (query @ values.T) * softmax_scale.
        // Independent PyTorch 2.11 CPU/HIP output: 0.5. The former division
        // by sqrt(512f32) yielded 0.50390625 with these identical operands.
        // One nonzero product removes accumulation-order ambiguity.
        let mut q = vec![0.; 512];
        let mut kv = vec![0.; 512];
        q[0] = 12.;
        kv[0] = 1.;
        let out = attention(&q, &[kv], &[0.522_517_5], 512);
        assert_eq!(out[0], 0.5);
        assert!(out[1..].iter().all(|&v| v == 0.));
    }
    #[test]
    fn rounding_ties_and_saturation() {
        for code in 0..126 {
            let mid = (decode_e4m3(code) + decode_e4m3(code + 1)) * 0.5;
            assert_eq!(
                encode(mid, 126, decode_e4m3),
                decode_e4m3(code + (code & 1))
            );
        }
        assert_eq!(encode(1e9, 126, decode_e4m3), 448.0);
        assert_eq!(encode(-0.0, 7, decode_e2m1).to_bits(), (-0.0f32).to_bits());
        assert_eq!(e8m0(0), 2f32.powi(-127));
        assert!(e8m0(255).is_nan());
    }
    #[test]
    fn hyperconnection_axes() {
        let mix = Mix {
            pre: vec![],
            post: vec![2., 3.],
            comb: vec![1., 2., 3., 4.],
        };
        assert_eq!(hc_post(&[5.], &[7., 11.], &mix), vec![50., 73.]);
        assert_eq!(hc_pre(&[1., 2., 3., 4.], &[1., 0.], 2), vec![1., 2.]);
    }
    #[test]
    fn sink_is_zero_value() {
        assert_eq!(
            attention(&[0., 0.], &[vec![2., 4.]], &[0.], 2),
            vec![1., 2.]
        );
        assert_eq!(attention(&[0., 0.], &[], &[1000.], 2), vec![0., 0.]);
    }
    #[test]
    fn quantized_scales_and_zero() -> Result<()> {
        let mut x = vec![0.; 32];
        quant_fp8(&mut x)?;
        assert!(x.iter().all(|v| *v == 0.));
        x.fill(3.);
        quant_fp4(&mut x, 16, true)?;
        assert_eq!(x, vec![3.; 32]);
        assert!(quant_fp8(&mut [f32::NAN; 32]).is_err());
        Ok(())
    }
}
