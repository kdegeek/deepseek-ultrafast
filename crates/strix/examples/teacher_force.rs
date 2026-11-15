//! Original-checkpoint validation, with all logits, fixed inputs and no Python.
use anyhow::{Result, ensure};
use deepseek_ultrafast_strix::decoder::Decoder;
use serde::Deserialize;
use serde_json::json;
use std::path::Path;

#[derive(Deserialize)]
struct Case {
    label: String,
    prompt: String,
    tokens: Vec<u32>,
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "teacher_force <checkpoint> <cases.json> <new-output-dir>"
    );
    let output = Path::new(&args[2]);
    std::fs::create_dir(output)?;
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    for case in cases {
        ensure!(
            !case.label.is_empty()
                && case
                    .label
                    .bytes()
                    .all(|x| x.is_ascii_alphanumeric() || x == b'-'),
            "invalid label"
        );
        let mut decoder = Decoder::open(Path::new(&args[0]), 8 * 1024 * 1024 * 1024, 128, None)?;
        let encoded = decoder.encode(&case.prompt)?;
        ensure!(
            case.tokens.starts_with(&encoded) && case.tokens.len() <= 128,
            "tokenization/context mismatch"
        );
        let dir = output.join(&case.label);
        std::fs::create_dir(&dir)?;
        let mut next = vec![];
        for (position, token) in case.tokens.iter().enumerate() {
            let logits = decoder.forward(*token)?;
            let greedy = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .unwrap()
                .0;
            next.push(greedy);
            std::fs::write(
                dir.join(format!("p{position:03}-logits.json")),
                serde_json::to_vec(&logits)?,
            )?;
            eprintln!("{} position {position}: next {greedy}", case.label);
        }
        std::fs::write(
            dir.join("summary.json"),
            serde_json::to_vec_pretty(&json!({
                "prompt":case.prompt, "prompt_tokens":encoded, "input_tokens":case.tokens,
                "next_tokens":next, "weight_cache":decoder.weights.stats,
                "peak_read_bytes":decoder.weights.store.peak_read_bytes,
            }))?,
        )?;
    }
    Ok(())
}
