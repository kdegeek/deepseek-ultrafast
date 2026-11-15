//! Explicit experimental native ATen/HIP validation. Never selected by default.
#[cfg(all(feature = "native-aten", windows))]
fn main() -> anyhow::Result<()> {
    use anyhow::ensure;
    use deepseek_ultrafast_strix::{
        aten::Api,
        decoder::{Backend, Decoder},
    };
    use serde::Deserialize;
    use serde_json::json;
    use std::{path::Path, time::Instant};
    #[derive(Deserialize)]
    struct Case {
        label: String,
        prompt: String,
        tokens: Vec<u32>,
        prompt_tokens: Vec<u32>,
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 4 && matches!(args[3].as_str(), "hc-vulkan" | "aten-ops"),
        "aten_teacher_force <checkpoint> <cases.json> <new-output-dir> <hc-vulkan|aten-ops>"
    );
    let output = Path::new(&args[2]);
    std::fs::create_dir(output)?;
    let api = Api::from_env()?;
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    for case in cases {
        let label = &case.label;
        ensure!(
            !label.is_empty()
                && label
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-'),
            "invalid label"
        );
        let dir = output.join(label);
        std::fs::create_dir(&dir)?;
        let trace = std::env::var_os("STRIX_ATEN_TRACE").map(|_| dir.join("trace"));
        let backend = if args[3] == "aten-ops" {
            Backend::AtenOps(api.clone())
        } else {
            Backend::AtenHc(api.clone())
        };
        let mut decoder = Decoder::open_backend(
            Path::new(&args[0]),
            8 * 1024 * 1024 * 1024,
            128,
            trace,
            backend,
        )?;
        let prompt = &case.prompt;
        let encoded = decoder.encode(prompt)?;
        let tokens = case.tokens;
        ensure!(
            encoded == case.prompt_tokens
                && tokens.starts_with(&encoded)
                && !tokens.is_empty()
                && tokens.len() <= 128,
            "tokenization/context mismatch"
        );
        let mut next = vec![];
        let started = Instant::now();
        for (position, &token) in tokens.iter().enumerate() {
            let logits = decoder.forward(token)?;
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
            eprintln!(
                "{label} position {position}: next {greedy} ({:.1}s)",
                started.elapsed().as_secs_f64()
            );
        }
        std::fs::write(
            dir.join("summary.json"),
            serde_json::to_vec_pretty(
                &json!({"backend":args[3],"experimental":true,"prompt":prompt,"prompt_tokens":encoded,"input_tokens":tokens,"next_tokens":next,"weight_cache":decoder.weights.stats,"aten_hc_resident_bytes":decoder.compatibility.as_ref().unwrap().resident_bytes,"total_weight_budget_bytes":8u64*1024*1024*1024,"aten_memory":api.memory()?,"peak_read_bytes":decoder.weights.store.peak_read_bytes,"seconds":started.elapsed().as_secs_f64()}),
            )?,
        )?;
    }
    std::fs::write(
        output.join("runtime.json"),
        serde_json::to_vec_pretty(&api.identity)?,
    )?;
    std::fs::write(
        output.join("loaded-modules-final.json"),
        serde_json::to_vec_pretty(&api.modules()?)?,
    )?;
    Ok(())
}
#[cfg(not(all(feature = "native-aten", windows)))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("requires Windows and --features native-aten")
}
