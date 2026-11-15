use deepseek_ultrafast_strix::{math, verification};
use serde_json::Value;

fn vector(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

#[test]
fn independent_pytorch_cpu_equations() -> anyhow::Result<()> {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/native-math.json"))?;
    for case in fixture["quantization"].as_array().unwrap() {
        let mut actual = vector(&case["input"]);
        match case["kind"].as_str().unwrap() {
            "fp8" => math::quant_fp8(&mut actual)?,
            "fp4_e8m0" => math::quant_fp4(&mut actual, 32, false)?,
            "fp4_e4m3" => math::quant_fp4(&mut actual, 16, true)?,
            _ => unreachable!(),
        }
        assert_eq!(actual, vector(&case["output"]));
    }
    let s = &fixture["sinkhorn"];
    let mix = math::sinkhorn(
        &vector(&s["z"]),
        &vector(&s["scale"]),
        &vector(&s["base"]),
        4,
        20,
        1e-6,
    );
    for (name, actual) in [("pre", &mix.pre), ("post", &mix.post), ("comb", &mix.comb)] {
        verification::compare(name, actual, &vector(&s[name]))?;
    }
    let h = &fixture["hc"];
    assert_eq!(
        math::hc_pre(&vector(&h["residual"]), &mix.pre, 32),
        vector(&h["collapsed"])
    );
    assert_eq!(
        math::hc_post(&vector(&h["input"]), &vector(&h["residual"]), &mix),
        vector(&h["output"])
    );
    let a = &fixture["attention"];
    let raw = vector(&a["kv"]);
    let kv = a["indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let i = v.as_i64().unwrap();
            if i < 0 {
                Vec::new()
            } else {
                raw[i as usize * 32..(i as usize + 1) * 32].to_vec()
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(
        math::attention(&vector(&a["query"]), &kv, &vector(&a["sinks"]), 32),
        vector(&a["output"])
    );
    Ok(())
}
