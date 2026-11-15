use anyhow::{Context, Result, bail, ensure};
use deepseek_ultrafast_strix::{
    checkpoint, decoder,
    reference::{self, Format},
    runtime::Gpu,
    verification,
};
use serde_json::json;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("generate-experimental") if (3..=7).contains(&args.len()) => {
            eprintln!(
                "Experimental native decoder: consult docs/STRIX_HALO.md for scoped oracle acceptance and limitations."
            );
            let max_tokens = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(8);
            let gib: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(8);
            let context = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(128);
            let bytes = gib
                .checked_mul(1024 * 1024 * 1024)
                .context("cache size overflow")?;
            let result = decoder::generate(
                Path::new(&args[1]),
                &args[2],
                max_tokens,
                bytes,
                context,
                args.get(6).map(Into::into),
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Some("device") if args.len() == 1 => {
            println!("{}", serde_json::to_string_pretty(&Gpu::new()?.info)?);
        }
        Some("verify") if args.len() == 1 => {
            let gpu = Gpu::new()?;
            let checks = verification::run(&gpu)?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"device":gpu.info, "fixture":"synthetic", "checks":checks, "passed":true})
                )?
            );
        }
        Some("inspect") if (2..=3).contains(&args.len()) => {
            let report = checkpoint::inspect(Path::new(&args[1]))?;
            let mut value = serde_json::to_value(&report)?;
            if let Some(budget) = args.get(2) {
                let gib: f64 = budget.parse().context("budget must be a number of GiB")?;
                ensure!(
                    gib.is_finite() && gib > 0.0,
                    "budget must be finite and positive"
                );
                value["weight_budget_gib"] = json!(gib);
                value["stored_weights_within_budget"] =
                    json!(report.stored_bytes as f64 <= gib * 1073741824.0);
                value["budget_note"] = json!(
                    "Weights only. Runtime allocations, KV cache, OS reserve and GPU allocation limits are additional; this does not establish inference feasibility."
                );
            }
            println!("{}", serde_json::to_string_pretty(&value)?);
        }
        Some("check-matrix") if args.len() == 3 => {
            let matrix = checkpoint::load_matrix(Path::new(&args[1]), &args[2])?;
            let format = match matrix.format {
                0 => Format::F32,
                1 => Format::Bf16,
                2 => Format::Fp8,
                3 => Format::Fp4,
                x => bail!("unknown format {x}"),
            };
            let weights = reference::pack_bytes(&matrix.weights);
            let scales = reference::pack_bytes(&matrix.scales);
            let input: Vec<f32> = (0..matrix.cols)
                .map(|i| ((i * 17 % 37) as f32 - 18.0) / 19.0)
                .collect();
            let expected =
                reference::matvec(&weights, &scales, matrix.rows, matrix.cols, format, &input)?;
            let gpu = Gpu::new()?;
            let resident =
                gpu.upload_matrix(&weights, &scales, matrix.rows, matrix.cols, format)?;
            let check =
                verification::compare(&args[2], &gpu.matvec(&resident, &input)?, &expected)?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"device":gpu.info, "checkpoint":args[1], "input":"synthetic", "check":check, "passed":true})
                )?
            );
        }
        Some("--help" | "-h") | None => println!(
            "Strix Halo native Vulkan tools\n\nstrix device\nstrix verify\nstrix inspect <checkpoint-directory> [weight-budget-GiB]\nstrix check-matrix <checkpoint-directory> <tensor-name>\nstrix generate-experimental <checkpoint-directory> <raw-prompt> [max-tokens=8] [cache-GiB=8] [context=128] [trace-directory]\n\nExperimental decoding has scoped original-model acceptance and known intermediate-logit differences. The Python reference remains the validated API path."
        ),
        _ => bail!("unknown command or incorrect argument count; use --help"),
    }
    Ok(())
}
