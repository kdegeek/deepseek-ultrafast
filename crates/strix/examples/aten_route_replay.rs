//! Validation only. The serving command cannot enable conditional route replay.
#[cfg(all(feature = "native-aten", windows))]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use deepseek_ultrafast_strix::{
        aten::Api,
        decoder::{Backend, Decoder},
    };
    use serde_json::{Value, json};
    use std::{path::Path, time::Instant};
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "aten_route_replay <checkpoint> <reference-directory> <new-output-directory>"
    );
    let reference = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    std::fs::create_dir(output)?;
    let record: Value = serde_json::from_slice(&std::fs::read(reference.join("reference.json"))?)?;
    let case = &record["case"];
    let api = Api::from_env()?;
    let mut decoder = Decoder::open_backend(
        Path::new(&args[0]),
        8 * 1024 * 1024 * 1024,
        128,
        None,
        Backend::AtenOps(api.clone()),
    )?;
    let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone())?;
    let prompt_tokens: Vec<u32> = serde_json::from_value(case["prompt_tokens"].clone())?;
    ensure!(
        decoder.encode(case["prompt"].as_str().context("prompt")?)? == prompt_tokens
            && tokens.starts_with(&prompt_tokens)
            && !tokens.is_empty()
            && tokens.len() <= 128,
        "input/tokenization mismatch"
    );
    decoder.enable_routing_replay(reference.to_path_buf())?;
    let started = Instant::now();
    let mut next = vec![];
    for (position, token) in tokens.iter().enumerate() {
        let logits = decoder.forward(*token)?;
        let greedy = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0;
        next.push(greedy);
        std::fs::write(
            output.join(format!("p{position:03}-logits.json")),
            serde_json::to_vec(&logits)?,
        )?;
        eprintln!(
            "conditional replay position {position}: {greedy} ({:.1}s)",
            started.elapsed().as_secs_f64()
        );
    }
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&json!({"case":case,
        "next_tokens":next,"seconds":started.elapsed().as_secs_f64(),"runtime":api.identity,
        "modules":api.modules()?,"memory":api.memory()?,"routes":decoder.routing_replay_report(),
        "weight_cache":decoder.weights.stats,"peak_read_bytes":decoder.weights.store.peak_read_bytes}))?,
    )?;
    Ok(())
}
#[cfg(not(all(feature = "native-aten", windows)))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("requires Windows and --features native-aten")
}
