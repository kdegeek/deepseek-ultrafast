//! Independent scalar references for the native Vulkan baseline.
//!
//! Accumulations use f64 so hardware tests detect indexing/decoding mistakes
//! without requiring the GPU reduction tree to match a serial f32 reduction.
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Format {
    F32 = 0,
    Bf16 = 1,
    Fp8 = 2,
    Fp4 = 3,
}

pub fn pack_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut word = [0; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(word)
        })
        .collect()
}

pub fn decode_e4m3(code: u8) -> f32 {
    let sign = if code & 128 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    if exponent == 15 && mantissa == 7 {
        return f32::NAN;
    }
    if exponent == 0 {
        sign * f32::from(mantissa) / 512.0
    } else {
        sign * (1.0 + f32::from(mantissa) / 8.0) * 2f32.powi(i32::from(exponent) - 7)
    }
}

pub fn decode_e2m1(code: u8) -> f32 {
    let magnitude = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(code & 7)];
    if code & 8 == 0 { magnitude } else { -magnitude }
}

/// OCP E8M0: code zero is 2^-127 (a subnormal F32), code 255 is NaN.
pub fn decode_e8m0(code: u8) -> f32 {
    match code {
        0 => f32::from_bits(0x0040_0000),
        255 => f32::NAN,
        _ => f32::from_bits(u32::from(code) << 23),
    }
}

fn byte_at(words: &[u32], index: usize) -> u8 {
    (words[index / 4] >> ((index % 4) * 8)) as u8
}

pub fn validate_matrix(
    weights: &[u32],
    scales: &[u32],
    rows: usize,
    cols: usize,
    format: Format,
) -> Result<()> {
    ensure!(rows > 0 && cols > 0, "matrix dimensions must be nonzero");
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow::anyhow!("matrix dimensions overflow"))?;
    ensure!(
        elements <= u32::MAX as usize,
        "matrix exceeds shader index range"
    );
    let per_word = match format {
        Format::F32 => 1,
        Format::Bf16 => 2,
        Format::Fp8 => 4,
        Format::Fp4 => 8,
    };
    ensure!(
        weights.len() == elements.div_ceil(per_word),
        "incorrect packed weight length"
    );
    let scale_count = match format {
        Format::F32 | Format::Bf16 => 0,
        Format::Fp8 => {
            ensure!(
                cols.is_multiple_of(32),
                "FP8 columns must be divisible by 32"
            );
            rows.div_ceil(32) * (cols / 32)
        }
        Format::Fp4 => {
            ensure!(
                cols.is_multiple_of(32),
                "FP4 columns must be divisible by 32"
            );
            rows * (cols / 32)
        }
    };
    ensure!(
        scales.len() == scale_count.div_ceil(4),
        "incorrect packed scale length"
    );
    for index in 0..scale_count {
        ensure!(
            byte_at(scales, index) != 255,
            "non-finite E8M0 scale at {index}"
        );
    }
    for index in 0..elements {
        let finite = match format {
            Format::F32 => f32::from_bits(weights[index]).is_finite(),
            Format::Bf16 => ((weights[index / 2] >> ((index % 2) * 16)) & 0x7f80) != 0x7f80,
            Format::Fp8 => (byte_at(weights, index) & 0x7f) != 0x7f,
            Format::Fp4 => true,
        };
        ensure!(finite, "non-finite weight at {index}");
    }
    Ok(())
}

pub fn matvec(
    weights: &[u32],
    scales: &[u32],
    rows: usize,
    cols: usize,
    format: Format,
    input: &[f32],
) -> Result<Vec<f32>> {
    validate_matrix(weights, scales, rows, cols, format)?;
    ensure!(input.len() == cols, "incorrect input vector length");
    ensure!(
        input.iter().all(|x| x.is_finite()),
        "input vector must be finite"
    );
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut sum = 0.0f64;
        for (col, &x) in input.iter().enumerate() {
            let index = row * cols + col;
            let w = match format {
                Format::F32 => f32::from_bits(weights[index]),
                Format::Bf16 => {
                    f32::from_bits(((weights[index / 2] >> ((index % 2) * 16)) & 65535) << 16)
                }
                Format::Fp8 => {
                    decode_e4m3(byte_at(weights, index))
                        * decode_e8m0(byte_at(scales, (row / 32) * (cols / 32) + col / 32))
                }
                Format::Fp4 => {
                    decode_e2m1(((weights[index / 8] >> ((index % 8) * 4)) & 15) as u8)
                        * decode_e8m0(byte_at(scales, row * (cols / 32) + col / 32))
                }
            };
            sum += f64::from(w) * f64::from(x);
        }
        output.push(sum as f32);
    }
    Ok(output)
}

pub fn rms_norm(
    input: &[f32],
    gamma: &[f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Result<Vec<f32>> {
    ensure!(rows > 0 && cols > 0, "matrix dimensions must be nonzero");
    ensure!(
        rows.checked_mul(cols) == Some(input.len()),
        "incorrect RMS input length"
    );
    ensure!(gamma.len() == cols, "incorrect RMS weight length");
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "epsilon must be positive and finite"
    );
    ensure!(
        input.iter().chain(gamma).all(|x| x.is_finite()),
        "RMS inputs must be finite"
    );
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(cols) {
        let square_sum: f64 = row.iter().map(|&x| f64::from(x).powi(2)).sum();
        let inv_rms = (square_sum / cols as f64 + f64::from(epsilon))
            .sqrt()
            .recip();
        output.extend(
            row.iter()
                .zip(gamma)
                .map(|(&x, &g)| (f64::from(x) * inv_rms * f64::from(g)) as f32),
        );
    }
    Ok(output)
}

pub fn silu_mul(input: &[f32], other: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        input.len() == other.len(),
        "SiLU inputs must have equal lengths"
    );
    ensure!(
        input.iter().chain(other).all(|x| x.is_finite()),
        "SiLU inputs must be finite"
    );
    Ok(input
        .iter()
        .zip(other)
        .map(|(&x, &y)| {
            let x = f64::from(x);
            let silu = if x >= 0.0 {
                x / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                x * e / (1.0 + e)
            };
            (silu * f64::from(y)) as f32
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_fp4_codes_and_signed_zero() {
        for (code, expected) in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
            .into_iter()
            .enumerate()
        {
            assert_eq!(decode_e2m1(code as u8), expected);
            assert_eq!(decode_e2m1(code as u8 + 8), -expected);
        }
        assert!(decode_e2m1(8).is_sign_negative());
    }

    #[test]
    fn all_fp8_codes_have_expected_spacing_and_sign() {
        let mut previous = 0.0;
        for code in 0..127u8 {
            let value = decode_e4m3(code);
            assert!(value.is_finite() && value >= previous);
            if code > 0 {
                let exponent = code >> 3;
                let expected_spacing = if exponent <= 1 {
                    1.0 / 512.0
                } else {
                    2f32.powi(i32::from((code - 1) >> 3) - 10)
                };
                assert_eq!(value - previous, expected_spacing);
            }
            assert_eq!(decode_e4m3(code | 128), -value);
            previous = value;
        }
        assert_eq!(decode_e4m3(126), 448.0);
        assert!(decode_e4m3(127).is_nan() && decode_e4m3(255).is_nan());
        assert!(decode_e4m3(128).is_sign_negative());
    }

    #[test]
    fn scale_convention_matches_upstream() {
        assert_eq!(decode_e8m0(0), 2f32.powi(-127));
        for code in 1..255u8 {
            assert_eq!(decode_e8m0(code), 2f32.powi(i32::from(code) - 127));
        }
        assert!(decode_e8m0(255).is_nan());
    }

    #[test]
    fn little_endian_packing_and_partial_word() {
        assert_eq!(pack_bytes(&[1, 2, 3, 4, 5]), vec![0x04030201, 5]);
        assert!(pack_bytes(&[]).is_empty());
    }

    #[test]
    fn fp4_scales_are_per_row_and_32_columns() {
        let weights = pack_bytes(&[0x22; 2 * 64 / 2]);
        let scales = pack_bytes(&[127, 128, 129, 130]);
        assert_eq!(
            matvec(&weights, &scales, 2, 64, Format::Fp4, &vec![1.0; 64]).unwrap(),
            vec![96.0, 384.0]
        );
    }

    #[test]
    fn fp4_low_nibble_precedes_high_nibble_across_words() {
        let bytes: Vec<u8> = (0..16)
            .map(|i| ((2 * i + 1) % 16) << 4 | ((2 * i) % 16))
            .collect();
        let input: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let values = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let expected: f32 = input
            .iter()
            .enumerate()
            .map(|(i, x)| x * values[i % 16])
            .sum();
        assert_eq!(
            matvec(&pack_bytes(&bytes), &[127], 1, 32, Format::Fp4, &input).unwrap(),
            vec![expected]
        );
    }

    #[test]
    fn fp8_scales_are_per_32_by_32_block_with_row_tail() {
        let weights = pack_bytes(&vec![0x38; 33 * 64]);
        let scales = pack_bytes(&[127, 128, 129, 130]);
        let output = matvec(&weights, &scales, 33, 64, Format::Fp8, &vec![1.0; 64]).unwrap();
        assert_eq!(&output[..32], &[96.0; 32]);
        assert_eq!(output[32], 384.0);
    }

    #[test]
    fn bf16_and_f32_allow_odd_widths() {
        let values: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let bf16 = pack_bytes(
            &values
                .iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let f32s: Vec<_> = values.iter().map(|x| x.to_bits()).collect();
        for (weights, format) in [(&bf16, Format::Bf16), (&f32s, Format::F32)] {
            assert_eq!(
                matvec(weights, &[], 2, 3, format, &[1.0, 2.0, 3.0]).unwrap(),
                vec![6.0, -12.0]
            );
        }
    }

    #[test]
    fn malformed_buffers_and_nonfinite_values_are_rejected() {
        assert!(validate_matrix(&[], &[], 0, 3, Format::F32).is_err());
        assert!(validate_matrix(&[0], &[], usize::MAX, 2, Format::F32).is_err());
        assert!(validate_matrix(&[0], &[], 1, 3, Format::Fp4).is_err());
        assert!(validate_matrix(&[0; 4], &[], 1, 32, Format::Fp4).is_err());
        assert!(validate_matrix(&[f32::NAN.to_bits()], &[], 1, 1, Format::F32).is_err());
        assert!(validate_matrix(&[0x7f80], &[], 1, 1, Format::Bf16).is_err());
        assert!(validate_matrix(&[0x7f; 8], &[127], 1, 32, Format::Fp8).is_err());
        assert!(validate_matrix(&[0; 4], &[255], 1, 32, Format::Fp4).is_err());
    }

    #[test]
    fn rms_norm_rows_and_zero_input() {
        assert_eq!(
            rms_norm(&[0.0; 6], &[1.0; 3], 2, 3, 1e-5).unwrap(),
            vec![0.0; 6]
        );
        let out = rms_norm(&[1.0, 1.0, -2.0, -2.0], &[2.0, 3.0], 2, 2, 1e-5).unwrap();
        for (actual, expected) in out.iter().zip([2.0, 3.0, -2.0, -3.0]) {
            assert!((actual - expected).abs() < 2e-5);
        }
        assert!(rms_norm(&[1.0], &[1.0], 1, 1, 0.0).is_err());
        assert!(rms_norm(&[1.0], &[1.0], 1, 1, f32::NAN).is_err());
    }

    #[test]
    fn silu_extreme_finite_values_and_length_errors() {
        let out = silu_mul(&[-1000.0, 0.0, 1000.0], &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(out, vec![0.0, 0.0, 3000.0]);
        assert!(silu_mul(&[1.0], &[]).is_err());
        assert!(silu_mul(&[f32::INFINITY], &[1.0]).is_err());
    }
}
