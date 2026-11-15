//! Strict text-only OpenAI wire adapter; GGUF owns tokenization and chat encoding.
use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const MODEL: &str = "Qwen/Qwen3.8-27B";
pub const MAX_TOKENS: u64 = 8192;

#[derive(Clone, Debug)]
pub struct Request {
    pub native: Value,
    pub stream: bool,
    pub include_usage: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: String,
    content: String,
}

fn integer(body: &Value, name: &str, default: u64, min: u64, max: u64) -> Result<u64> {
    let number = body.get(name).map_or(Some(default), Value::as_u64);
    ensure!(
        number.is_some_and(|n| (min..=max).contains(&n)),
        "{name} must be an integer in [{min}, {max}]"
    );
    Ok(number.unwrap())
}
fn number(body: &Value, name: &str, default: f64, min: f64, max: f64) -> Result<f64> {
    let number = body.get(name).map_or(Some(default), Value::as_f64);
    ensure!(
        number.is_some_and(|n| n.is_finite() && (min..=max).contains(&n)),
        "{name} must be a finite number in [{min}, {max}]"
    );
    Ok(number.unwrap())
}
fn boolean(body: &Value, name: &str, default: bool) -> Result<bool> {
    body.get(name)
        .map_or(Some(default), Value::as_bool)
        .ok_or_else(|| anyhow::anyhow!("{name} must be boolean"))
}

pub fn validate(body: Value, chat: bool) -> Result<Request> {
    let object = body
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("request must be a JSON object"))?;
    for key in object.keys() {
        ensure!(
            matches!(
                key.as_str(),
                "model"
                    | "max_tokens"
                    | "stream"
                    | "stream_options"
                    | "temperature"
                    | "top_p"
                    | "top_k"
                    | "min_p"
                    | "seed"
                    | "repeat_penalty"
                    | "presence_penalty"
                    | "frequency_penalty"
                    | "n"
                    | "stop"
                    | "enable_thinking"
            ) || key == if chat { "messages" } else { "prompt" },
            "unsupported field: {key}"
        );
    }
    ensure!(body["model"] == MODEL, "model must be {MODEL}");
    integer(&body, "n", 1, 1, 1)?;
    let stream = boolean(&body, "stream", false)?;
    let include_usage = if let Some(options) = body.get("stream_options") {
        ensure!(stream, "stream_options requires stream=true");
        let object = options
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("stream_options must be an object"))?;
        ensure!(
            object.keys().all(|k| k == "include_usage"),
            "only stream_options.include_usage is supported"
        );
        boolean(options, "include_usage", false)?
    } else {
        false
    };
    let stop = match body.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| anyhow::anyhow!("stop entries must be strings"))
            })
            .collect::<Result<Vec<_>>>()?,
        _ => bail!("stop must be a string or array of strings"),
    };
    ensure!(
        stop.len() <= 4
            && stop
                .iter()
                .all(|s| (1..=256).contains(&s.len()) && !s.contains('\0')),
        "stop supports at most four nonempty strings of at most 256 UTF-8 bytes without NUL"
    );
    let thinking = boolean(&body, "enable_thinking", false)?;
    let mut native = json!({
        "max_tokens":integer(&body,"max_tokens",512,1,MAX_TOKENS)?,
        "temperature":number(&body,"temperature",if thinking {1.0} else {0.7},0.0,2.0)?,
        "top_p":number(&body,"top_p",if thinking {0.95} else {0.8},0.000001,1.0)?,
        "top_k":integer(&body,"top_k",20,0,1000)?,
        "min_p":number(&body,"min_p",0.0,0.0,1.0)?,
        "repeat_penalty":number(&body,"repeat_penalty",1.0,0.01,2.0)?,
        "presence_penalty":number(&body,"presence_penalty",if chat && !thinking {1.5} else {0.0},-2.0,2.0)?,
        "frequency_penalty":number(&body,"frequency_penalty",0.0,-2.0,2.0)?,
        "enable_thinking":thinking,
        "stop":stop,
    });
    // Absence leaves randomness to the native sampler, not a clock-based seed.
    if body.get("seed").is_some() {
        // UINT32_MAX is llama.cpp's random-seed sentinel, not a fixed seed.
        native["seed"] = json!(integer(&body, "seed", 0, 0, u32::MAX as u64 - 1)?);
    }
    if chat {
        let messages: Vec<Message> = serde_json::from_value(body["messages"].clone())?;
        ensure!(
            (1..=256).contains(&messages.len()),
            "messages must contain 1..=256 text messages"
        );
        ensure!(
            messages.last().is_some_and(|m| m.role == "user"),
            "the final message must have role=user"
        );
        for (i, message) in messages.iter().enumerate() {
            ensure!(
                matches!(message.role.as_str(), "system" | "user" | "assistant"),
                "only system, user, and assistant roles are supported"
            );
            ensure!(
                message.role != "system" || i == 0,
                "system message is supported only at the beginning"
            );
            ensure!(
                !message.content.contains('\0'),
                "message content must not contain NUL"
            );
        }
        native["messages"] = serde_json::to_value(messages)?;
    } else {
        ensure!(
            !object.contains_key("enable_thinking"),
            "enable_thinking is supported only by chat completions"
        );
        let prompt = body["prompt"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("prompt must be one nonempty string"))?;
        ensure!(
            !prompt.is_empty() && !prompt.contains('\0'),
            "prompt must be nonempty without NUL"
        );
        native["prompt"] = json!(prompt);
    }
    Ok(Request {
        native,
        stream,
        include_usage,
    })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}
#[derive(Clone, Debug)]
pub struct Generated {
    pub text: String,
    pub finish_reason: String,
    pub usage: Usage,
    pub metrics: Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn base() -> Value {
        json!({"model":MODEL,"prompt":"Write a useful answer"})
    }
    #[test]
    fn sampling_defaults_and_explicit_options() {
        let request = validate(base(), false).unwrap();
        assert_eq!(request.native["max_tokens"], 512);
        assert_eq!(request.native["temperature"], 0.7);
        assert!(request.native.get("seed").is_none());
        assert_eq!(request.native["presence_penalty"], 0.0);
        let mut body = base();
        for (name, value) in [
            ("temperature", json!(0)),
            ("seed", json!(123)),
            ("top_k", json!(0)),
            ("stop", json!("END")),
        ] {
            body[name] = value;
        }
        let native = validate(body, false).unwrap().native;
        assert_eq!(native["seed"], 123);
        assert_eq!(native["stop"], json!(["END"]));
    }
    #[test]
    fn thinking_and_non_thinking_sampling_defaults() {
        let mut body =
            json!({"model":MODEL,"messages":[{"role":"user","content":"Explain your answer."}]});
        let native = validate(body.clone(), true).unwrap().native;
        assert_eq!(native["temperature"], 0.7);
        assert_eq!(native["top_p"], 0.8);
        assert_eq!(native["presence_penalty"], 1.5);
        body["enable_thinking"] = json!(true);
        let native = validate(body.clone(), true).unwrap().native;
        assert_eq!(native["temperature"], 1.0);
        assert_eq!(native["top_p"], 0.95);
        assert_eq!(native["presence_penalty"], 0.0);
        body["temperature"] = json!(0);
        body["top_p"] = json!(0.75);
        body["presence_penalty"] = json!(-0.5);
        body["frequency_penalty"] = json!(1.5);
        let native = validate(body, true).unwrap().native;
        assert_eq!(native["temperature"], 0.0);
        assert_eq!(native["top_p"], 0.75);
        assert_eq!(native["presence_penalty"], -0.5);
        assert_eq!(native["frequency_penalty"], 1.5);
    }
    #[test]
    fn unsupported_and_malformed_fields_fail() {
        for (key, value) in [
            ("tools", json!([])),
            ("logprobs", json!(true)),
            ("n", json!(2)),
            ("seed", json!(-1)),
            ("seed", json!(4294967295u64)),
            ("max_tokens", json!(2.2)),
            ("max_tokens", json!(0)),
            ("temperature", json!("0")),
            ("top_p", json!(0)),
            ("top_k", json!(-1)),
            ("stream", json!(1)),
            ("repeat_penalty", json!(0)),
            ("presence_penalty", json!(2.1)),
            ("frequency_penalty", json!(-2.1)),
            ("stop", json!([""])),
            ("enable_thinking", json!(false)),
        ] {
            let mut body = base();
            body[key] = value;
            assert!(validate(body, false).is_err(), "accepted {key}");
        }
    }
    #[test]
    fn gguf_receives_unmodified_multiturn_unicode() {
        let messages = json!([{"role":"system","content":"You are helpful."},{"role":"user","content":"こんにちは 🐈"},{"role":"assistant","content":"Hello!"},{"role":"user","content":"Say café."}]);
        let request = validate(
            json!({"model":MODEL,"messages":messages,"enable_thinking":true}),
            true,
        )
        .unwrap();
        assert_eq!(request.native["messages"], messages);
        assert_eq!(request.native["enable_thinking"], true);
    }
    #[test]
    fn streaming_options_and_message_shapes() {
        let mut body = base();
        body["stream_options"] = json!({"include_usage":true});
        assert!(validate(body.clone(), false).is_err());
        body["stream"] = json!(true);
        assert!(validate(body, false).unwrap().include_usage);
        for messages in [
            json!([]),
            json!([{"role":"assistant","content":"hi"}]),
            json!([{"role":"user","content":[{"type":"text","text":"hi"}]}]),
            json!([{"role":"user","content":"hi","name":"extra"}]),
        ] {
            assert!(validate(json!({"model":MODEL,"messages":messages}), true).is_err());
        }
    }
}
