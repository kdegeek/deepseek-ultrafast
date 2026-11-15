use anyhow::{Context, Result, bail};
use deepseek_ultrafast_qwen::serving::{self, Config};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut config = Config {
        model_path: PathBuf::new(),
        library: "target/qwen-resident/build/bin/Release/qwen_native.dll".into(),
        port: 18083,
        context: 32768,
        batch: 2048,
        ubatch: 512,
        threads: 8,
        gpu_layers: 999,
        flash_attention: true,
        kv_type: "q8_0".into(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if matches!(flag.as_str(), "--help" | "-h") {
            println!(
                "qwen-resident --model <verified.gguf> [--library <qwen_native.dll>] [--port 18083] [--context 32768] [--batch 2048] [--ubatch 512] [--threads 8] [--gpu-layers 999] [--kv-type q8_0|f16] [--no-flash-attention]\nLoopback only. Readiness: /readyz. Chat and raw text: /v1/chat/completions, /v1/completions. See docs/QWEN_STRIX.md."
            );
            return Ok(());
        }
        if flag == "--no-flash-attention" {
            config.flash_attention = false;
            continue;
        }
        let value = args
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model" => config.model_path = value.into(),
            "--library" => config.library = value.into(),
            "--port" => config.port = value.parse().context("port")?,
            "--context" => config.context = value.parse().context("context")?,
            "--batch" => config.batch = value.parse().context("batch")?,
            "--ubatch" => config.ubatch = value.parse().context("ubatch")?,
            "--threads" => config.threads = value.parse().context("threads")?,
            "--gpu-layers" => config.gpu_layers = value.parse().context("gpu-layers")?,
            "--kv-type" => config.kv_type = value,
            _ => bail!("unknown argument: {flag}"),
        }
    }
    anyhow::ensure!(
        !config.model_path.as_os_str().is_empty(),
        "--model is required"
    );
    config.model_path = config.model_path.canonicalize().context("model path")?;
    serving::run(config)
}
