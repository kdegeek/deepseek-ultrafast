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
        mtp_draft: 0,
        dflash_draft: 0,
        draft_sampling: "greedy".into(),
        draft_model_path: None,
        request_timeout_seconds: 3600,
        dll_directories: Vec::new(),
        preload_libraries: Vec::new(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if matches!(flag.as_str(), "--help" | "-h") {
            println!(
                "qwen-resident --model <verified.gguf> [--library <qwen_native.dll>] [--port 18083] [--context 32768] [--batch 2048] [--ubatch 512] [--threads 8] [--gpu-layers 999] [--kv-type q8_0|f16] [--mtp-draft 0..3] [--draft-model <verified-draft.gguf> --dflash-draft 1..7] [--draft-sampling greedy|probabilistic] [--request-timeout-seconds 3600] [--no-flash-attention]\nPrivate HIP dependencies: repeat --dll-directory <directory> and --preload-library <dll>; prefer the verified launcher.\nLoopback only. Readiness: /readyz. Chat and raw text: /v1/chat/completions, /v1/completions. See docs/QWEN_STRIX.md."
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
            "--dll-directory" => config.dll_directories.push(value.into()),
            "--preload-library" => config.preload_libraries.push(value.into()),
            "--draft-model" => config.draft_model_path = Some(value.into()),
            "--dflash-draft" => config.dflash_draft = value.parse().context("dflash-draft")?,
            "--draft-sampling" => config.draft_sampling = value,
            "--port" => config.port = value.parse().context("port")?,
            "--context" => config.context = value.parse().context("context")?,
            "--batch" => config.batch = value.parse().context("batch")?,
            "--ubatch" => config.ubatch = value.parse().context("ubatch")?,
            "--threads" => config.threads = value.parse().context("threads")?,
            "--gpu-layers" => config.gpu_layers = value.parse().context("gpu-layers")?,
            "--kv-type" => config.kv_type = value,
            "--mtp-draft" => config.mtp_draft = value.parse().context("mtp-draft")?,
            "--request-timeout-seconds" => {
                config.request_timeout_seconds = value.parse().context("request-timeout-seconds")?
            }
            _ => bail!("unknown argument: {flag}"),
        }
    }
    anyhow::ensure!(
        !config.model_path.as_os_str().is_empty(),
        "--model is required"
    );
    config.model_path = config.model_path.canonicalize().context("model path")?;
    if let Some(draft) = config.draft_model_path.as_mut() {
        *draft = draft.canonicalize().context("draft model path")?;
    }
    serving::run(config)
}
