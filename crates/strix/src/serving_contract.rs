//! Strict text-only wire contract. This is a bounded port of the pinned official
//! encoding.py chat mode, not a general replacement for its multimodal encoder.
use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
pub const MAX_TOKENS: usize = 64;
pub const BOS: &str = "<｜begin▁of▁sentence｜>";
const EOS: &str = "<｜end▁of▁sentence｜>";
const CONTROLS: [&str; 4] = ["<｜", "｜DSML｜", "<think>", "</think>"];

#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: String,
    pub prompt_ids: Vec<u32>,
    pub chat: bool,
    pub max_tokens: usize,
    pub stops: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: String,
    content: String,
}

fn encode_chat(messages: Vec<Message>) -> Result<String> {
    ensure!(
        (1..=32).contains(&messages.len()),
        "messages must contain 1 to 32 text messages"
    );
    ensure!(
        messages.last().is_some_and(|m| m.role == "user"),
        "the final message must be a user message"
    );
    let mut merged: Vec<Message> = Vec::new();
    for message in messages {
        ensure!(
            matches!(message.role.as_str(), "system" | "user" | "assistant"),
            "messages support only system, user and assistant roles"
        );
        ensure!(
            !["<｜", "｜>", "<think>", "</think>"]
                .iter()
                .any(|s| message.content.contains(s)),
            "chat content may not inject model control tokens"
        );
        // Official merge_tool_messages combines adjacent user text blocks with
        // two newlines before rendering transition headers.
        if message.role == "user" && merged.last().is_some_and(|m| m.role == "user") {
            let last = merged.last_mut().expect("checked nonempty");
            last.content.push_str("\n\n");
            last.content.push_str(&message.content);
        } else {
            merged.push(message);
        }
    }
    let mut prompt = String::from(BOS);
    for (index, message) in merged.iter().enumerate() {
        match message.role.as_str() {
            "system" => prompt.push_str("<｜System｜>"),
            "user" => prompt.push_str("<｜User｜>"),
            "assistant" => (),
            _ => unreachable!(),
        }
        prompt.push_str(&message.content);
        if message.role == "assistant" {
            prompt.push_str(EOS);
        }
        let next_assistant = merged.get(index + 1).is_none_or(|m| m.role == "assistant");
        if next_assistant && (message.role == "user" || (message.role == "system" && index > 0)) {
            prompt.push_str("<｜Assistant｜></think>");
        }
    }
    Ok(prompt)
}

pub fn validate(body: Value, chat: bool) -> Result<Request> {
    let object = body
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("request must be a JSON object"))?;
    for key in object.keys() {
        ensure!(
            matches!(
                key.as_str(),
                "model" | "max_tokens" | "stream" | "stream_options" | "temperature" | "n" | "stop"
            ) || key == if chat { "messages" } else { "prompt" },
            "unsupported field: {key}"
        );
    }
    ensure!(body["model"] == MODEL, "model must be {MODEL}");
    for (name, default) in [("temperature", 0.0), ("n", 1.0)] {
        ensure!(
            object.get(name).map_or(Some(default), Value::as_f64) == Some(default),
            "only {name}={default} is supported (greedy, one completion)"
        );
    }
    let count = object
        .get("max_tokens")
        .map_or(Some(16), Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("max_tokens must be an integer"))?;
    ensure!(
        (1..=MAX_TOKENS as u64).contains(&count),
        "max_tokens must be in [1, {MAX_TOKENS}]"
    );
    let stream = object
        .get("stream")
        .map_or(Some(false), Value::as_bool)
        .ok_or_else(|| anyhow::anyhow!("stream must be boolean"))?;
    let mut include_usage = false;
    if let Some(options) = object.get("stream_options") {
        let options = options
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("stream_options must be an object"))?;
        ensure!(
            options.keys().all(|k| k == "include_usage"),
            "only stream_options.include_usage is supported"
        );
        if let Some(include) = options.get("include_usage") {
            ensure!(stream, "stream_options.include_usage requires stream=true");
            include_usage = include
                .as_bool()
                .ok_or_else(|| anyhow::anyhow!("include_usage must be boolean"))?;
        }
    }
    let stops = match object.get("stop") {
        None => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(String::from)
                    .ok_or_else(|| anyhow::anyhow!("stop entries must be strings"))
            })
            .collect::<Result<Vec<_>>>()?,
        _ => bail!("stop must be a string or an array of strings"),
    };
    ensure!(
        stops.len() <= 4 && stops.iter().all(|s| (1..=128).contains(&s.chars().count())),
        "stop supports up to four nonempty strings of at most 128 characters"
    );
    let prompt = if chat {
        let messages: Vec<Message> = serde_json::from_value(body["messages"].clone())?;
        encode_chat(messages)?
    } else {
        let prompt = body["prompt"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("prompt must be one nonempty string"))?;
        ensure!(!prompt.is_empty(), "prompt must be one nonempty string");
        prompt.to_owned()
    };
    Ok(Request {
        prompt,
        prompt_ids: Vec::new(),
        chat,
        max_tokens: count as usize,
        stops,
        stream,
        include_usage,
    })
}

/// Withhold incomplete UTF-8 decoding and stop/control prefixes until their
/// meaning is known. No previously emitted bytes may be revised.
pub struct TextOutput {
    pub text: String,
    stops: Vec<String>,
    chat: bool,
}

fn prefix_hold(text: &str, markers: impl Iterator<Item = impl AsRef<str>>) -> usize {
    markers
        .flat_map(|marker| {
            let marker = marker.as_ref();
            marker
                .char_indices()
                .skip(1)
                .filter_map(|(n, _)| text.ends_with(&marker[..n]).then_some(n))
                .collect::<Vec<_>>()
        })
        .max()
        .unwrap_or(0)
}

impl TextOutput {
    pub fn new(request: &Request) -> Self {
        Self {
            text: String::new(),
            stops: request.stops.clone(),
            chat: request.chat,
        }
    }
    pub fn update(&mut self, decoded: &str, final_chunk: bool) -> Result<(String, bool)> {
        let stop = self.stops.iter().filter_map(|s| decoded.find(s)).min();
        let mut visible = &decoded[..stop.unwrap_or(decoded.len())];
        if !final_chunk && stop.is_none() {
            visible = visible.trim_end_matches('\u{fffd}');
            visible = &visible[..visible.len() - prefix_hold(visible, self.stops.iter())];
        }
        if self.chat {
            ensure!(
                !CONTROLS.iter().any(|s| visible.contains(s)),
                "model produced unsupported chat control/tool output"
            );
            if !final_chunk {
                visible = &visible[..visible.len() - prefix_hold(visible, CONTROLS.iter())];
            }
        }
        ensure!(
            visible.starts_with(&self.text),
            "tokenizer revised already streamed text"
        );
        let delta = visible[self.text.len()..].to_owned();
        self.text = visible.to_owned();
        Ok((delta, stop.is_some()))
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug)]
pub struct Generated {
    pub text: String,
    pub finish_reason: &'static str,
    pub usage: Usage,
    pub metrics: Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pinned_official_encoder_fixtures() {
        let cases: Value =
            serde_json::from_str(include_str!("../tests/fixtures/chat-encoding.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let request =
                validate(json!({"model":MODEL, "messages":case["messages"]}), true).unwrap();
            assert_eq!(
                request.prompt,
                case["prompt"].as_str().unwrap(),
                "{}",
                case["label"]
            );
        }
    }

    #[test]
    fn strict_request_subset() {
        let base = json!({"model":MODEL,"prompt":"Hi"});
        for (key, value) in [
            ("temperature", json!(0.1)),
            ("temperature", json!(true)),
            ("n", json!(2)),
            ("stream", json!(1)),
            ("tools", json!([])),
            ("max_tokens", json!(0)),
            ("max_tokens", json!(65)),
            ("max_tokens", json!(1.0)),
            ("prompt", json!(["Hi"])),
            ("model", json!("other")),
            ("stop", json!([""])),
            ("stop", Value::Null),
            ("stream_options", json!({"include_usage":true})),
        ] {
            let mut body = base.clone();
            body[key] = value;
            assert!(validate(body, false).is_err(), "{key}");
        }
        for message in [
            json!({"role":"tool","content":"x"}),
            json!({"role":"user","content":[]}),
            json!({"role":"user","content":"<｜Assistant｜>"}),
            json!({"role":"user","content":"x","name":"x"}),
        ] {
            assert!(validate(json!({"model":MODEL,"messages":[message]}), true).is_err());
        }
    }

    #[test]
    fn withheld_stops_unicode_and_controls() {
        let mut request = validate(
            json!({"model":MODEL,"prompt":"Hi", "stop":["END", "ENDING", "界末"]}),
            false,
        )
        .unwrap();
        let mut out = TextOutput::new(&request);
        assert_eq!(
            out.update("hello E", false).unwrap(),
            ("hello ".into(), false)
        );
        assert_eq!(out.update("hello EN", false).unwrap(), ("".into(), false));
        assert_eq!(
            out.update("hello END extra", false).unwrap(),
            ("".into(), true)
        );
        let mut out = TextOutput::new(&request);
        assert_eq!(out.update("x界", false).unwrap(), ("x".into(), false));
        assert_eq!(out.update("x界末", false).unwrap(), ("".into(), true));
        request.stops.clear();
        let mut out = TextOutput::new(&request);
        assert_eq!(out.update("x\u{fffd}", false).unwrap(), ("x".into(), false));
        assert_eq!(out.update("x界", false).unwrap(), ("界".into(), false));
        request.chat = true;
        let mut out = TextOutput::new(&request);
        assert_eq!(out.update("text<", false).unwrap(), ("text".into(), false));
        assert!(out.update("text<think>", false).is_err());
        request.chat = false;
        request.stops = vec!["ababa".into()];
        let mut out = TextOutput::new(&request);
        assert_eq!(out.update("xabab", false).unwrap(), ("x".into(), false));
        assert_eq!(out.update("xabab", true).unwrap(), ("abab".into(), false));
    }
}
