//! Deterministic hardware acceptance checks. Inputs are synthetic, not model output.
use crate::{
    reference::{self, Format},
    runtime::Gpu,
};
use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Serialize)]
pub struct Check {
    pub operation: String,
    pub elements: usize,
    pub max_abs_error: f32,
}

pub fn compare(name: &str, actual: &[f32], expected: &[f32]) -> Result<Check> {
    ensure!(
        actual.len() == expected.len() && !actual.is_empty(),
        "{name}: output shape mismatch"
    );
    let mut max_abs_error = 0.0f32;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        ensure!(
            a.is_finite() && e.is_finite(),
            "{name}: non-finite result at {i}: {a} / {e}"
        );
        let error = (a - e).abs();
        ensure!(
            error <= 2e-4 + 2e-4 * e.abs(),
            "{name}: mismatch at {i}: GPU {a}, CPU {e}, error {error}"
        );
        max_abs_error = max_abs_error.max(error);
    }
    Ok(Check {
        operation: name.into(),
        elements: actual.len(),
        max_abs_error,
    })
}

pub fn run(gpu: &Gpu) -> Result<Vec<Check>> {
    let mut checks = Vec::new();
    // Odd row count crosses both 32-row scale tiles and the 64-lane workgroup.
    for (rows, cols) in [(1usize, 32usize), (33, 96), (65, 256)] {
        let n = rows * cols;
        let input: Vec<f32> = (0..cols)
            .map(|i| ((i * 17 % 37) as f32 - 18.0) / 19.0)
            .collect();
        let f32_weights: Vec<f32> = (0..n)
            .map(|i| ((i * 13 % 43) as f32 - 21.0) / 23.0)
            .collect();
        let bf16: Vec<u8> = f32_weights
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        // All finite E4M3 codes, sign and block-dependent scales.
        let fp8: Vec<u8> = (0..n)
            .map(|i| ((i % 127) as u8) | (((i / 127) % 2) as u8 * 128))
            .collect();
        let fp4: Vec<u8> = (0..n / 2)
            .map(|i| ((i % 16) | (((i + 5) % 16) << 4)) as u8)
            .collect();
        for (format, weights, scales) in [
            (
                Format::F32,
                f32_weights.iter().map(|x| x.to_bits()).collect(),
                vec![],
            ),
            (Format::Bf16, reference::pack_bytes(&bf16), vec![]),
            (
                Format::Fp8,
                reference::pack_bytes(&fp8),
                reference::pack_bytes(
                    &(0..rows.div_ceil(32) * (cols / 32))
                        .map(|i| 122 + (i % 5) as u8)
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                Format::Fp4,
                reference::pack_bytes(&fp4),
                reference::pack_bytes(
                    &(0..rows * (cols / 32))
                        .map(|i| 125 + (i % 5) as u8)
                        .collect::<Vec<_>>(),
                ),
            ),
        ] {
            let expected = reference::matvec(&weights, &scales, rows, cols, format, &input)?;
            let matrix = gpu.upload_matrix(&weights, &scales, rows, cols, format)?;
            let actual = gpu.matvec(&matrix, &input)?;
            checks.push(compare(
                &format!("matvec/{format:?}/{rows}x{cols}"),
                &actual,
                &expected,
            )?);
            // Reusing resident weights with different inputs detects stale state/readback.
            let zeros = vec![0.0; cols];
            checks.push(compare(
                &format!("matvec-zero/{format:?}/{rows}x{cols}"),
                &gpu.matvec(&matrix, &zeros)?,
                &vec![0.0; rows],
            )?);
        }
    }
    for (rows, cols) in [(1, 1), (3, 67), (2, 256)] {
        let input: Vec<f32> = (0..rows * cols)
            .map(|i| (i % 31) as f32 / 7.0 - 2.0)
            .collect();
        let gamma: Vec<f32> = (0..cols).map(|i| (i % 5) as f32 / 3.0).collect();
        checks.push(compare(
            &format!("rms_norm/{rows}x{cols}"),
            &gpu.rms_norm(&input, &gamma, rows, cols, 1e-5)?,
            &reference::rms_norm(&input, &gamma, rows, cols, 1e-5)?,
        )?);
        checks.push(compare(
            &format!("silu_mul/{}", input.len()),
            &gpu.silu_mul(&input, &input)?,
            &reference::silu_mul(&input, &input)?,
        )?);
    }
    Ok(checks)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn comparison_rejects_nan_and_wrong_shapes() {
        assert!(compare("nan", &[f32::NAN], &[0.0]).is_err());
        assert!(compare("shape", &[0.0], &[]).is_err());
        assert!(compare("error", &[1.0], &[0.0]).is_err());
    }
    #[test]
    #[ignore = "requires an AMD hardware Vulkan adapter; run explicitly"]
    fn amd_hardware_acceptance() -> Result<()> {
        let gpu = Gpu::new()?;
        println!("{}", serde_json::to_string_pretty(&gpu.info)?);
        let checks = run(&gpu)?;
        println!("{}", serde_json::to_string_pretty(&checks)?);
        Ok(())
    }
}
