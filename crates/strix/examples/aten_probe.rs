//! Replay unchanged original-input component requests on native HIP.
#[cfg(all(feature = "native-aten", windows))]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use deepseek_ultrafast_strix::{
        aten::{Api, Compatibility},
        weights::Store,
    };
    use serde_json::{Value, json};
    use std::path::Path;
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "aten_probe <checkpoint> <requests.json> <new-output-dir>"
    );
    std::fs::create_dir(&args[2])?;
    let api = Api::from_env()?;
    let mut compat = Compatibility::new(api.clone());
    let mut store = Store::open(Path::new(&args[0]))?;
    let mut projector = deepseek_ultrafast_strix::aten::Projector::new(api.clone());
    let mut stats = deepseek_ultrafast_strix::weights::Stats {
        budget_bytes: 1024 * 1024 * 1024,
        ..Default::default()
    };
    let request_bytes = std::fs::read(&args[1])?;
    let requests: Vec<Value> = serde_json::from_slice(&request_bytes)?;
    std::fs::write(Path::new(&args[2]).join("requests.json"), request_bytes)?;
    let mut actual = serde_json::Map::new();
    for req in requests {
        if !matches!(
            req["op"].as_str(),
            Some(
                "mix"
                    | "rms"
                    | "pre"
                    | "post"
                    | "linear"
                    | "range"
                    | "router"
                    | "engram"
                    | "attention"
                    | "swiglu"
                    | "router_stages"
                    | "rope"
                    | "rope_config"
                    | "rope_aten"
                    | "rope_aten_config"
            )
        ) {
            continue;
        }
        let vector = |name: &str| -> anyhow::Result<Vec<f32>> {
            Ok(serde_json::from_value(req[name].clone())?)
        };
        let number = |name: &str| {
            req[name]
                .as_f64()
                .with_context(|| format!("missing number {name}"))
        };
        let integer = |name: &str| -> anyhow::Result<usize> {
            let value = req[name]
                .as_u64()
                .with_context(|| format!("missing integer {name}"))?;
            ensure!(value <= 1 << 30, "integer {name} exceeds probe bound");
            Ok(value as usize)
        };
        let input = if req["op"] == "attention" {
            vector("query")?
        } else {
            vector("input")?
        };
        let result = match req["op"].as_str().unwrap_or("") {
            "rope_aten_config" => {
                let cfg = deepseek_ultrafast_strix::config::Config::load(
                    &Path::new(&args[0]).join("inference/config.json"),
                )?;
                compat.init_rotary(&cfg, 128)?;
                let mut value = input.clone();
                compat.rotary(
                    &mut value,
                    true,
                    integer("position")?,
                    cfg.rope_head_dim,
                    cfg.rope_head_dim,
                    req["inverse"].as_bool().unwrap_or(false),
                )?;
                json!(value)
            }
            "rope_aten" => {
                let cos = vector("cos")?;
                let sin = vector("sin")?;
                json!(api.vectors(
                    17,
                    &[&input, &cos, &sin],
                    &[
                        (cos.len() * 2) as f64,
                        (cos.len() * 2) as f64,
                        0.,
                        f64::from(req["inverse"].as_bool().unwrap_or(false))
                    ],
                    input.len()
                )?)
            }
            "rope" | "rope_config" => {
                let mut value = input.clone();
                let (cos, sin) = if req["op"] == "rope" {
                    (vector("cos")?, vector("sin")?)
                } else {
                    let cfg = deepseek_ultrafast_strix::config::Config::load(
                        &Path::new(&args[0]).join("inference/config.json"),
                    )?;
                    let p = integer("position")?;
                    ensure!(p < 128, "rotary position exceeds context bound");
                    let (c, s) = deepseek_ultrafast_strix::config::rope_table(
                        cfg.rope_head_dim,
                        p + 1,
                        cfg.original_seq_len,
                        cfg.compress_rope_theta,
                        cfg.rope_factor,
                        cfg.beta_fast,
                        cfg.beta_slow,
                    );
                    let d = cfg.rope_head_dim / 2;
                    (
                        c[p * d..(p + 1) * d].to_vec(),
                        s[p * d..(p + 1) * d].to_vec(),
                    )
                };
                ensure!(
                    !cos.is_empty()
                        && cos.len() == sin.len()
                        && input.len().is_multiple_of(cos.len() * 2),
                    "rotary input shape"
                );
                deepseek_ultrafast_strix::math::rope(
                    &mut value,
                    cos.len() * 2,
                    cos.len() * 2,
                    &cos,
                    &sin,
                    req["inverse"].as_bool().unwrap_or(false),
                );
                json!(value)
            }
            "router_stages" => {
                let n = integer("topk")?;
                let d = input.len();
                let v = api.vectors(
                    14,
                    &[&input, &vector("bias")?],
                    &[n as f64, number("scale")?],
                    2 * d + 2 * n,
                )?;
                json!({"scores":v[..d],"corrected":v[d..2*d],"indices":v[2*d..2*d+n],"weights":v[2*d+n..]})
            }
            "swiglu" => json!(api.vectors(
                7,
                &[&input, &vector("up")?],
                &[number("limit")?, number("weight")?],
                input.len()
            )?),
            "linear" | "range" | "router" => {
                let name = req["name"].as_str().context("missing weight name")?;
                let first = req["first"].as_u64().unwrap_or(0) as usize;
                let rows = req["rows"]
                    .as_u64()
                    .map(|n| n as usize)
                    .unwrap_or(*store.info(name)?.shape.first().context("matrix rank")?);
                let v = projector.range(
                    &mut store,
                    &mut stats,
                    name,
                    &input,
                    first,
                    rows,
                    req["round"].as_bool().unwrap_or(false),
                )?;
                if req["op"] == "router" {
                    let n = integer("topk")?;
                    let r = api.vectors(
                        8,
                        &[&v, &vector("bias")?],
                        &[n as f64, number("scale")?],
                        n * 2,
                    )?;
                    json!({"logits":v,"indices":r[..n],"weights":r[n..]})
                } else {
                    json!(v)
                }
            }
            "engram" => json!(api.vectors(
                6,
                &[&input, &vector("kv")?, &vector("q")?, &vector("k")?],
                &[number("dim")?, number("eps")?],
                input.len()
            )?),
            "attention" => {
                let dim = integer("dim")?;
                let values: Vec<Vec<f32>> = serde_json::from_value(req["kv"].clone())?;
                let mut flat = vec![];
                let mut valid = vec![];
                for row in values {
                    valid.push(if row.is_empty() { 0. } else { 1. });
                    if row.is_empty() {
                        flat.extend(vec![0.; dim]);
                    } else {
                        flat.extend(row);
                    }
                }
                json!(api.vectors(
                    10,
                    &[&input, &flat, &vector("sinks")?, &valid],
                    &[dim as f64],
                    input.len()
                )?)
            }
            "mix" => {
                let v = compat.mix(
                    &mut store,
                    req["prefix"].as_str().context("missing HC prefix")?,
                    &input,
                    1e-6,
                    1e-20,
                    20,
                )?;
                json!({"raw":v[..24],"inv":v[24],"z":v[25..49],"pre":v[49..53],"post":v[53..57],"comb":v[57..]})
            }
            "rms" => {
                let v = api.vectors(
                    2,
                    &[&input, &vector("gamma")?],
                    &[number("eps")?],
                    input.len() + 1,
                )?;
                json!({"inv":v[0],"output":v[1..]})
            }
            "pre" => json!(api.pre(&input, &vector("pre")?)?),
            "post" => json!(api.vectors(
                4,
                &[
                    &input,
                    &vector("residual")?,
                    &vector("post")?,
                    &vector("comb")?
                ],
                &[],
                input.len() * 4
            )?),
            _ => continue,
        };
        let label = req["label"].as_str().context("missing probe label")?;
        ensure!(
            actual.insert(label.into(), result).is_none(),
            "duplicate probe label {label}"
        );
    }
    std::fs::write(
        Path::new(&args[2]).join("actual.json"),
        serde_json::to_vec(&actual)?,
    )?;
    std::fs::write(
        Path::new(&args[2]).join("identity.json"),
        serde_json::to_vec_pretty(
            &json!({"runtime":api.identity,"resident_hc_bytes":compat.resident_bytes,"peak_read_bytes":store.peak_read_bytes,"loaded_modules_final":api.modules()?,"aten_memory":api.memory()?}),
        )?,
    )?;
    Ok(())
}
#[cfg(not(all(feature = "native-aten", windows)))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("requires Windows and --features native-aten")
}
