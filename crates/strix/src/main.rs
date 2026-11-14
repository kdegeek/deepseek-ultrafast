use anyhow::{Context, Result, bail, ensure};
use deepseek_ultrafast_strix::{
    checkpoint,
    reference::{self, Format},
    runtime::Gpu,
    verification,
};
use serde_json::json;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
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
            "Strix Halo Vulkan port tools (full text generation is not implemented yet)\n\nstrix device\nstrix verify\nstrix inspect <checkpoint-directory> [weight-budget-GiB]\nstrix check-matrix <checkpoint-directory> <tensor-name>"
        ),
        _ => bail!("unknown command or incorrect argument count; use --help"),
    }
    Ok(())
}
