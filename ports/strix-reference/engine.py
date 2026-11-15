"""Persistent, single-owner AMD reference engine; no native Rust decoding."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(ROOT))

from encoding import encode_messages, eos_token, parse_message_from_completion_text

MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"


class Cancelled(Exception):
    pass


class InvalidRequest(ValueError):
    """Admission error, distinct from a model/storage ValueError during inference."""


def validate_request(body, chat, max_tokens=64):
    """Strict supported subset. Never silently ignore client generation options."""
    if not isinstance(body, dict):
        raise ValueError("request must be a JSON object")
    allowed = {"model", "max_tokens", "stream", "stream_options", "temperature", "n", "stop"}
    allowed.add("messages" if chat else "prompt")
    unknown = set(body) - allowed
    if unknown:
        raise ValueError(f"unsupported fields: {', '.join(sorted(unknown))}")
    if body.get("model") != MODEL_ID:
        raise ValueError(f"model must be {MODEL_ID}")
    for name, default, expected in (("temperature", 0, 0), ("n", 1, 1)):
        value = body.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
            raise ValueError(f"only {name}={expected} is supported (greedy, one completion)")
    count = body.get("max_tokens", min(16, max_tokens))
    if type(count) is not int or not 1 <= count <= max_tokens:
        raise ValueError(f"max_tokens must be an integer in [1, {max_tokens}]")
    stream = body.get("stream", False)
    if type(stream) is not bool:
        raise ValueError("stream must be boolean")
    options = body.get("stream_options", {})
    if not isinstance(options, dict) or set(options) - {"include_usage"}:
        raise ValueError("only stream_options.include_usage is supported")
    if "include_usage" in options and (not stream or type(options["include_usage"]) is not bool):
        raise ValueError("include_usage must be boolean and requires stream=true")
    stops = body.get("stop", [])
    if isinstance(stops, str):
        stops = [stops]
    if not isinstance(stops, list) or len(stops) > 4 or any(
        not isinstance(s, str) or not 1 <= len(s) <= 128 for s in stops
    ):
        raise ValueError("stop must be a string or up to four nonempty strings of at most 128 characters")
    if chat:
        messages = body.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
            raise ValueError("messages must contain 1 to 32 text messages")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user", "assistant")
                    or not isinstance(message["content"], str)):
                raise ValueError("messages support only role=system/user/assistant and string content; no images or tools")
            if any(marker in message["content"] for marker in ("<｜", "｜>", "<think>", "</think>")):
                raise ValueError("chat content may not inject model control tokens")
        if messages[-1]["role"] != "user":
            raise ValueError("the final message must be a user message")
        prompt = encode_messages(messages, thinking_mode="chat")
    else:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be one nonempty string")
    return {"prompt": prompt, "chat": chat, "max_tokens": count, "stop": stops,
            "stream": stream, "include_usage": options.get("include_usage", False)}


class TextOutput:
    """Incremental text with stop-prefix and incomplete UTF-8 withholding."""
    def __init__(self, stops, chat):
        self.stops, self.chat = stops, chat
        self.text = ""

    def update(self, decoded, final=False):
        stop_at = min((decoded.find(s) for s in self.stops if s in decoded), default=None)
        stopped = stop_at is not None
        visible = decoded[:stop_at] if stopped else decoded
        if not final and not stopped:
            # Full-token decoding can end in an incomplete UTF-8 code point.
            visible = visible.rstrip("\ufffd")
            hold = max((size for s in self.stops for size in range(1, len(s))
                        if visible.endswith(s[:size])), default=0)
            if hold:
                visible = visible[:-hold]
        if self.chat:
            # Control tokens and DSML must never leak as ordinary assistant text.
            controls = ("<｜", "｜DSML｜", "<think>", "</think>")
            if any(s in visible for s in controls):
                raise RuntimeError("model produced unsupported chat control/tool output")
            if not final:
                hold = max((size for s in controls for size in range(1, len(s))
                            if visible.endswith(s[:size])), default=0)
                if hold:
                    visible = visible[:-hold]
        if not visible.startswith(self.text):
            raise RuntimeError("tokenizer revised already streamed text")
        delta, self.text = visible[len(self.text):], visible
        return delta, stopped


def reset_request_state(model, module):
    """Reset only mutable runtime buffers, preserving RoPE/hash constants and weights."""
    import torch
    with torch.inference_mode():
        for name, buffer in model.named_buffers():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in {"window_kv_cache", "compress_kv_cache", "k_cache", "kv_state"}:
                buffer.zero_()
            elif leaf == "score_state":
                buffer.fill_(-torch.inf)
            elif name == "engram_hash.cache":
                buffer.fill_(-1)
    module.shared_attn = module.SharedAttentionRuntime()
    module.get_window_topk_idxs.cache_clear()
    module.get_dspark_topk_idxs.cache_clear()


class ReferenceEngine:
    def __init__(self, checkpoint, cache_gib=8, retain_gib=0, context=256, prefill_batch=16):
        import torch
        import model as official
        from checkpoint import build_lazy_model
        from transformers import AutoTokenizer

        started = time.perf_counter()
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("AMD HIP PyTorch required; no CPU fallback")
        torch.set_num_threads(8)
        torch.set_default_dtype(torch.bfloat16)
        self.torch, self.official = torch, official
        self.checkpoint, self.context, self.prefill_batch = Path(checkpoint), context, prefill_batch
        manifest = json.loads((self.checkpoint / "source-manifest.json").read_text())
        if manifest["repo"] != MODEL_ID or manifest["revision"] != REVISION:
            raise ValueError("requires the pinned original DeepSeek-V4.1-Flash checkpoint")
        config = json.loads((self.checkpoint / "inference/config.json").read_text())
        config.update(max_batch_size=1, max_seq_len=context, temperature=0, vision_n_layers=0, n_mtp_layers=0)
        config["compress_ratios"] = config["compress_ratios"][:config["n_layers"]]
        fields = {f.name for f in dataclasses.fields(official.ModelArgs)}
        if set(config) - fields:
            raise ValueError("unsupported model configuration")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True, trust_remote_code=False)
        self.model, self.manager = build_lazy_model(
            official.ModelArgs(**config), self.tokenizer, self.checkpoint, "cuda",
            int(cache_gib * 1024**3), int(retain_gib * 1024**3))
        self.cancel = None
        self.hooks = [layer.register_forward_pre_hook(self._check_cancel) for layer in self.model.layers]
        reset_request_state(self.model, official)
        self.identity = {"model": MODEL_ID, "revision": REVISION, "device": torch.cuda.get_device_name(),
                         "torch": torch.__version__, "hip": torch.version.hip, "platform": platform.platform(),
                         "context": context, "prefill_batch": prefill_batch,
                         "initialization_seconds": time.perf_counter() - started,
                         "encoding_sha256": hashlib.sha256((ROOT / "upstream/encoding.py").read_bytes()).hexdigest(),
                         "tokenizer_sha256": hashlib.sha256((self.checkpoint / "tokenizer.json").read_bytes()).hexdigest(),
                         "provenance_sha256": {name: hashlib.sha256((self.checkpoint / name).read_bytes()).hexdigest()
                                               for name in ("source-manifest.json", "download-verified.json",
                                                            "inference/config.json", "tokenizer_config.json")},
                         "weight_cache": self.manager.stats()}
        self.last_metrics = None

    def _check_cancel(self, *unused):
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled("client disconnected or server stopping")

    def generate(self, request, cancel, emit):
        torch = self.torch
        self.last_metrics = None
        prompt = self.tokenizer.encode(request["prompt"], add_special_tokens=not request["chat"])
        if not prompt or len(prompt) + request["max_tokens"] > self.context:
            raise InvalidRequest(f"prompt ({len(prompt)}) plus max_tokens exceeds context {self.context}")
        self.cancel = cancel
        steps, sampled, text_tokens = [], [], []
        before = self.manager.stats()
        output = TextOutput(request["stop"], request["chat"])
        started = time.perf_counter()
        reason, stop_reason = "length", "max_tokens"
        reset_request_state(self.model, self.official)
        torch.cuda.reset_peak_memory_stats()
        emit("ready", {"prompt_tokens": len(prompt)})

        def forward(tokens, position, stage):
            self._check_cancel()
            step_start = time.perf_counter()
            next_ids, logits, _ = self.model(torch.tensor([tokens], dtype=torch.long), position)
            if not torch.isfinite(logits).all().item():
                raise RuntimeError(f"nonfinite logits at {position}")
            next_id = int(next_ids.reshape(-1)[-1].item())
            steps.append({"stage": stage, "position": position, "input_tokens": tokens,
                          "next_token_id": next_id, "logits_finite": True,
                          "logits_min": logits.min().item(), "logits_max": logits.max().item(),
                          "seconds": time.perf_counter() - step_start})
            self._check_cancel()
            return next_id

        try:
            with torch.inference_mode(), torch.device("cuda"):
                # Official decode accepts one token when start_pos > 0. Batch only
                # the initial bounded chunk; never send multi-token decode chunks.
                initial = min(len(prompt), self.prefill_batch)
                next_id = forward(prompt[:initial], 0, "prefill")
                for pos in range(initial, len(prompt)):
                    next_id = forward([prompt[pos]], pos, "prefill")
                for offset in range(request["max_tokens"]):
                    self._check_cancel()
                    sampled.append(next_id)
                    if next_id == self.tokenizer.eos_token_id:
                        reason, stop_reason = "stop", "eos"
                        break
                    text_tokens.append(next_id)
                    decoded = self.tokenizer.decode(text_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    delta, stopped = output.update(decoded)
                    if delta:
                        emit("delta", delta)
                    if stopped:
                        reason, stop_reason = "stop", "stop_sequence"
                        break
                    if offset + 1 < request["max_tokens"]:
                        next_id = forward([next_id], len(prompt) + offset, "decode")
                decoded = self.tokenizer.decode(text_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                delta, _ = output.update(decoded, final=True)
                if delta:
                    emit("delta", delta)
                message = None
                if request["chat"]:
                    # The official parser requires a terminal marker. For length or
                    # user-stop truncation, append it only to parse the text fragment;
                    # finish_reason and usage still reflect the actual sampled IDs.
                    message = parse_message_from_completion_text(output.text + eos_token, thinking_mode="chat")
                    if message["tool_calls"] or message["reasoning_content"]:
                        raise RuntimeError("tools and thinking output are unsupported")
                    message = {"role": "assistant", "content": message["content"]}
                result = {"text": output.text, "message": message, "finish_reason": reason,
                          "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(sampled),
                                    "total_tokens": len(prompt) + len(sampled)}}
            return result
        finally:
            # Keep weights warm, but clear every request's mutable state even after
            # partial prefill, cancellation, malformed output, or an exception.
            reset_request_state(self.model, self.official)
            torch.cuda.synchronize()
            if any(self.manager.pins.values()) or any(self.manager._scope_stacks.values()):
                raise RuntimeError("checkpoint scopes were not released; restart the worker")
            after = self.manager.stats()
            self.last_metrics = {"prompt_token_ids": prompt, "sampled_token_ids": sampled,
                                 "text": output.text, "stop_reason": "cancelled" if cancel.is_set() else stop_reason,
                                 "cancelled": cancel.is_set(), "seconds": time.perf_counter() - started,
                                 "steps": steps, "weight_cache": after,
                                 "cache_delta": {k: after[k] - before[k] for k in
                                                 ("parameter_loads", "cache_hits", "loaded_weight_bytes",
                                                  "cache_hit_bytes", "load_seconds", "engram_rows_read")},
                                 "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                            "total_device_bytes": torch.cuda.get_device_properties(0).total_memory}}
            self.cancel = None

    def close(self):
        reset_request_state(self.model, self.official)
        for hook in self.hooks:
            hook.remove()
        self.manager.close()
