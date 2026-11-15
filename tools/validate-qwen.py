"""Bounded, reproducible acceptance for the in-process resident Qwen API.

Python standard library only. Connects to an already running loopback engine;
never starts or stops a server. Report files contain hashes/counts instead of
large prompts. These task-specific checks are not a general model benchmark.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import traceback


MODEL = "Qwen/Qwen3.8-27B"
ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
BENCHMARK_PROMPT = "Write a detailed practical handbook for maintaining wooden exterior doors in a rainy climate. Explain inspection, cleaning, finish repair, hardware adjustment, weather sealing, seasonal scheduling, and when professional repairs are needed. Give concrete steps and examples in every section. Keep writing until all sections are thoroughly covered; aim for at least 1500 words."


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def chat(prompt, max_tokens=128, **extra):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": SEED,
        "max_tokens": max_tokens,
        **extra,
    }


class Harness:
    def __init__(self, args):
        self.args = args
        self.report = {
            "schema_version": 1,
            "started_utc": utc(),
            "base_url": f"http://127.0.0.1:{args.port}",
            "suite": args.suite,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Bounded API, task-quality and timing checks; not a broad model-quality benchmark.",
            "records": [],
            "checks": [],
            "passed": False,
        }
        self.save()

    def save(self):
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.args.output.with_suffix(self.args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(self.args.output)

    def connection(self):
        return http.client.HTTPConnection("127.0.0.1", self.args.port, timeout=self.args.timeout)

    def get(self, path):
        connection = self.connection()
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def health(self):
        status, body = self.get("/health")
        check(status == 200, f"health returned {status}: {body}")
        return body

    def idle(self, timeout=120):
        started = time.perf_counter()
        while time.perf_counter() - started < timeout:
            health = self.health()
            check(health.get("state") != "failed", f"native worker failed: {health}")
            if not health.get("busy"):
                return health
            time.sleep(0.1)
        raise TimeoutError("engine did not become idle")

    def run(self, name, operation):
        print(json.dumps({"check": name, "state": "running"}), flush=True)
        started = time.perf_counter()
        try:
            detail = operation()
            record = {"name": name, "passed": True, "seconds": time.perf_counter() - started}
            if detail is not None:
                record["detail"] = detail
        except Exception as error:
            record = {
                "name": name,
                "passed": False,
                "seconds": time.perf_counter() - started,
                "error": str(error),
                "traceback": traceback.format_exc(limit=4),
            }
        self.report["checks"].append(record)
        self.save()
        print(json.dumps(record, ensure_ascii=True), flush=True)
        return record["passed"]

    def request(self, name, body, path="/v1/chat/completions", expected=200):
        connection = self.connection()
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        try:
            connection.request("POST", path, encoded, {"Content-Type": "application/json"})
            response = connection.getresponse()
            content = response.read()
            seconds = time.perf_counter() - started
            result = json.loads(content)
        finally:
            connection.close()
        record = {
            "name": name,
            "path": path,
            "utc": utc(),
            "http_status": response.status,
            "http_seconds": seconds,
            "request_sha256": hashlib.sha256(encoded).hexdigest(),
            "request_bytes": len(encoded),
            "parameters": {k: v for k, v in body.items() if k not in ("prompt", "messages")},
            "response": result,
        }
        self.report["records"].append(record)
        self.save()
        check(response.status == expected, f"{name}: HTTP {response.status}, expected {expected}: {str(result)[:1000]}")
        if response.status == 200:
            self.validate_response(result)
        return result

    @staticmethod
    def validate_response(result):
        check(result.get("model") == MODEL, "response must identify the actual Qwen model")
        check(len(result.get("choices", [])) == 1, "expected exactly one choice")
        usage = result.get("usage", {})
        check(all(isinstance(usage.get(k), int) and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")), "usage counts must be nonnegative integers")
        check(usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"], "usage arithmetic mismatch")
        check(result["choices"][0]["finish_reason"] in ("stop", "length"), "invalid finish reason")
        metrics = result.get("metrics", {})
        for name in ("prefill_ms", "decode_ms", "ttft_ms", "prefill_tps", "decode_tps"):
            check(name in metrics and math.isfinite(metrics[name]) and metrics[name] >= 0, f"missing/nonfinite timing {name}")

    def stream(self, name, body, path="/v1/chat/completions"):
        connection = self.connection()
        streamed = {**body, "stream": True, "stream_options": {"include_usage": True}}
        encoded = json.dumps(streamed, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        chunks = []
        pieces = []
        first_content = None
        last_content = None
        finish_reason = None
        usage = None
        metrics = None
        done = False
        try:
            connection.request("POST", path, encoded, {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise AssertionError(f"stream HTTP {response.status}: {response.read(2000)}")
            check(response.getheader("Content-Type", "").startswith("text/event-stream"), "missing SSE content type")
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    done = True
                    break
                event = json.loads(raw)
                check("error" not in event, f"stream error: {event}")
                check(event.get("model") == MODEL, "SSE model identity mismatch")
                elapsed = time.perf_counter() - started
                if event.get("usage"):
                    usage = event["usage"]
                if event.get("metrics"):
                    metrics = event["metrics"]
                for choice in event.get("choices", []):
                    piece = choice.get("delta", {}).get("content", "") if path.endswith("chat/completions") else choice.get("text", "")
                    if piece:
                        first_content = elapsed if first_content is None else first_content
                        last_content = elapsed
                        pieces.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                # Bounded event evidence; full generated text is retained once.
                if len(chunks) < 3 or event.get("usage") or any(c.get("finish_reason") for c in event.get("choices", [])):
                    chunks.append({"seconds": elapsed, "data": event})
        finally:
            connection.close()
        check(done, "stream did not terminate with [DONE]")
        check(usage is not None, "include_usage did not emit usage")
        check(finish_reason in ("length", "stop"), "stream has no terminal finish reason")
        text = "".join(pieces)
        if metrics is None:
            metrics = self.idle().get("last_request", {}).get("metrics", {})
        result = {
            "name": name,
            "path": path,
            "utc": utc(),
            "request_sha256": hashlib.sha256(encoded).hexdigest(),
            "request_bytes": len(encoded),
            "parameters": {k: v for k, v in streamed.items() if k not in ("prompt", "messages")},
            "text": text,
            "text_sha256": digest(text),
            "finish_reason": finish_reason,
            "usage": usage,
            "metrics": metrics,
            "client_ttft_seconds": first_content,
            "client_last_content_seconds": last_content,
            "http_seconds": time.perf_counter() - started,
            "delta_count": len(pieces),
            "bounded_event_samples": chunks,
        }
        self.report["records"].append(result)
        self.save()
        check(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "stream usage arithmetic mismatch")
        check(first_content is not None and text.strip(), "empty streamed completion")
        for key in ("prefill_ms", "decode_ms", "ttft_ms", "prefill_tps", "decode_tps"):
            check(key in metrics and math.isfinite(metrics[key]) and metrics[key] >= 0, f"stream missing metric {key}")
        return result


def answer(result):
    choice = result["choices"][0]
    return choice.get("message", {}).get("content", choice.get("text", ""))


def strict_number(value):
    return re.sub(r"[,$\s]", "", value.strip())


def test_quality(harness):
    cases = [
        ("arithmetic_invoice", "An order has 17 units at $24.50 each and a flat $12 shipping charge. What is the total? Reply with only the decimal number, two digits after the decimal point.", lambda value: strict_number(value) == "428.50", "428.50"),
        ("structured_instruction", 'Output only valid JSON with exactly two fields: "color" set to "blue" and "count" set to the integer 3. Do not use Markdown.', lambda value: json.loads(value) == {"color": "blue", "count": 3}, '{"color":"blue","count":3}'),
        ("unicode", 'Copy the following text exactly, with no extra text or quotation marks: Café — 東京 🪵', lambda value: value.strip() == "Café — 東京 🪵", "Café — 東京 🪵"),
        ("instruction_summary", "Summarize this in one sentence of at most 20 words: The warehouse received 12 doors on Monday. Three were damaged during shipping. Nine passed inspection and are ready for collection. Mention both usable and damaged quantities.", lambda value: len(value.split()) <= 20 and re.search(r"\b(nine|9)\b", value, re.I) and re.search(r"\b(three|3)\b", value, re.I), "<=20 words; usable9 damaged3"),
    ]
    for name, prompt, predicate, expected in cases:
        def run(name=name, prompt=prompt, predicate=predicate, expected=expected):
            response = harness.request(name, chat(prompt))
            value = answer(response)
            check(bool(predicate(value)), f"expected {expected}; got {value!r}")
            return {"expected": expected, "observed": value}
        harness.run(name, run)

    def multi_turn():
        body = chat("unused", 128)
        body["messages"] = [
            {"role": "system", "content": "Follow the user's inventory edits. Answer the final question with only an integer."},
            {"role": "user", "content": "The rack starts with 24 oak doors and 7 pine doors."},
            {"role": "assistant", "content": "Recorded: 24 oak doors and 7 pine doors."},
            {"role": "user", "content": "Sell 5 oak doors, then receive 3 oak doors. How many oak doors remain?"},
        ]
        response = harness.request("multi_turn_inventory", body)
        check(answer(response).strip() == "22", repr(answer(response)))
        return {"expected": "22", "observed": answer(response)}
    harness.run("multi_turn_inventory", multi_turn)

    def coding():
        response = harness.request("coding_clamp", chat("Write a Python function clamp(value, low, high) that returns value limited to the inclusive [low, high] range. Assume low <= high. Use only a function definition, return, and min/max calls; no imports, annotations, docstrings, tests, or explanation. Return only Python code, without Markdown.", 160))
        source = answer(response).strip()
        if source.startswith("```"):
            source = re.sub(r"^```(?:python)?\s*|\s*```$", "", source)
        tree = ast.parse(source)
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Call, ast.Name, ast.Load)
        check(all(isinstance(node, allowed) for node in ast.walk(tree)), "generated code exceeds the safe expression-only AST subset")
        check(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef) and tree.body[0].name == "clamp", "expected only clamp definition")
        function = tree.body[0]
        check([arg.arg for arg in function.args.args] == ["value", "low", "high"] and len(function.body) == 1, "incorrect clamp signature/body")
        check(not function.decorator_list and not function.args.defaults and not function.args.kw_defaults, "decorators/default expressions forbidden")
        check(all(not isinstance(node, ast.Call) or isinstance(node.func, ast.Name) and node.func.id in ("min", "max") and not node.keywords for node in ast.walk(tree)), "only min/max calls allowed")
        namespace = {"__builtins__": {"min": min, "max": max}}
        exec(compile(tree, "<validated-clamp>", "exec"), namespace)
        fixtures = [(-5, 0, 10, 0), (0, 0, 10, 0), (5, 0, 10, 5), (11, 0, 10, 10), (-9, -5, -1, -5), (-3, -5, -1, -3), (3.25, 0.5, 3.5, 3.25), (9, 4, 4, 4)]
        for value, low, high, expected in fixtures:
            observed = namespace["clamp"](value, low, high)
            check(observed == expected, f"clamp{value,low,high}: expected{expected}, got{observed}")
        return {"source": source, "fixtures_passed": len(fixtures), "execution_safety": "AST allowlist with only min/max calls; no imports, attributes, loops or arbitrary builtins"}
    harness.run("coding_clamp_fixtures", coding)

    def thinking():
        response = harness.request("thinking_opt_in", chat("Compute 37 multiplied by 24. Check the arithmetic, then give the final product and one short sentence explaining the calculation.", 384, enable_thinking=True))
        text = answer(response)
        has_think_tags = "<think>" in text
        final_answer = text
        if has_think_tags:
            check("</think>" in text, "thinking exceeded the bounded output budget without reaching an answer")
            final_answer = text.split("</think>", 1)[1]
        check(re.search(r"\b888\b", final_answer), f"thinking answer missing correct product888: {text!r}")
        return {"observed": text, "raw_thinking_tags_present": has_think_tags, "usage": response["usage"], "scope": "Explicit thinking request with a bounded arithmetic check; not a reasoning benchmark"}
    harness.run("thinking_opt_in_arithmetic", thinking)


def test_contract(harness):
    def invalid_options():
        invalid = [
            {"model": "wrong/model"}, {"temperature": -1}, {"temperature": "0"},
            {"max_tokens": 0}, {"max_tokens": 1.5}, {"top_p": 1.1}, {"n": 2},
            {"tools": []}, {"logprobs": True}, {"response_format": {"type": "json_object"}},
            {"presence_penalty": 2.1}, {"frequency_penalty": -2.1},
            {"stop": [""]}, {"stream_options": {"unknown": True}}, {"unsupported_option": True},
        ]
        for index, extra in enumerate(invalid):
            response = harness.request(f"reject_{index}", {**chat("Say hello", 8), **extra}, expected=400)
            check("error" in response, "rejection must have structured error")
        return {"invalid_requests_rejected": len(invalid)}
    harness.run("strict_request_validation", invalid_options)

    body = chat("Explain in four short sentences how a reusable shopping list helps a busy household.", 128)
    baseline = None
    def deterministic():
        nonlocal baseline
        baseline = harness.request("deterministic_A", body)
        harness.request("unrelated_request", chat("A bakery sold 9 rolls and then 8 more. Reply with only the total number sold.", 48))
        repeated = harness.request("deterministic_B_after_unrelated", body)
        check(answer(baseline) == answer(repeated), "greedy request isolation/repeat changed text")
        check(baseline["usage"] == repeated["usage"], "identical request usage changed")
        return {"text_sha256": digest(answer(baseline)), "usage": baseline["usage"]}
    harness.run("greedy_repeat_request_isolation", deterministic)

    def sampled_repeat():
        sampled = chat("Write a five-sentence scene in which a gardener discovers a tiny door at the base of an old tree.", 192, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, seed=SEED, presence_penalty=0.3, frequency_penalty=0.0, repeat_penalty=1.0)
        first = harness.request("sampled_seed_A", sampled)
        second = harness.request("sampled_seed_B", sampled)
        check(answer(first).strip(), "empty sampled output")
        check(answer(first) == answer(second) and first["usage"] == second["usage"], "fixed-seed sampled repeat changed output or usage")
        return {"temperature": 0.7, "seed": SEED, "text_sha256": digest(answer(first)), "usage": first["usage"]}
    harness.run("sampled_fixed_seed_repeat", sampled_repeat)

    def stream_equivalence():
        reference = baseline if baseline is not None else harness.request("stream_reference", body)
        streamed = harness.stream("stream_equivalence", body)
        check(streamed["text"] == answer(reference), "SSE/nonstream content mismatch")
        check(streamed["usage"] == reference["usage"], "SSE/nonstream usage mismatch")
        check(streamed["finish_reason"] == reference["choices"][0]["finish_reason"], "SSE finish reason mismatch")
        return {"text_sha256": digest(streamed["text"]), "usage": streamed["usage"]}
    harness.run("stream_nonstream_equivalence", stream_equivalence)

    def stop_sequence():
        reference = baseline if baseline is not None else harness.request("stop_reference", body)
        full = answer(reference)
        # Select a real multi-character interior sequence, testing suppression
        # across token/chunk boundaries without relying on instruction fidelity.
        check(len(full) >= 40, "reference too short for multi-token stop test")
        offset = len(full) // 3
        stop = full[offset:offset + 17]
        expected = full[:full.index(stop)]
        stopped = harness.request("stop_nonstream", {**body, "stop": [stop]})
        check(answer(stopped) == expected, f"stop mismatch: {answer(stopped)!r}, expected {expected!r}")
        check(stopped["choices"][0]["finish_reason"] == "stop", "stop sequence must finish with stop")
        streamed = harness.stream("stop_stream", {**body, "stop": [stop]})
        check(streamed["text"] == expected, "stream leaked stop-prefix bytes or hid preceding content")
        check(streamed["usage"] == stopped["usage"], "stopped stream usage mismatch")
        return {"stop": stop, "prefix_sha256": digest(expected), "usage": stopped["usage"]}
    harness.run("stop_buffering_stream_and_nonstream", stop_sequence)

    def raw_completion():
        raw = {"model": MODEL, "prompt": "Continue the numbered list of positive even integers.\n2, 4, 6,", "max_tokens": 32, "temperature": 0, "seed": SEED}
        response = harness.request("raw_completion", raw, "/v1/completions")
        check("8" in answer(response) and "10" in answer(response), f"raw continuation unexpected: {answer(response)!r}")
        streamed = harness.stream("raw_completion_stream", raw, "/v1/completions")
        check(streamed["text"] == answer(response) and streamed["usage"] == response["usage"], "raw streaming parity mismatch")
        return {"text": answer(response), "usage": response["usage"]}
    harness.run("raw_completion_and_stream", raw_completion)

    def busy_cancel():
        before = harness.idle()
        connection = harness.connection()
        generated = chat("Write a detailed 2000-word practical guide to organizing a community library. Include sections on cataloguing, lending, volunteers, accessibility, maintenance, and events.", 2048, stream=True)
        encoded = json.dumps(generated).encode("utf-8")
        try:
            connection.request("POST", "/v1/chat/completions", encoded, {"Content-Type": "application/json"})
            response = connection.getresponse()
            check(response.status == 200, f"cancellable stream status {response.status}")
            deadline = time.perf_counter() + 20
            while not harness.health().get("busy"):
                check(time.perf_counter() < deadline, "request never became busy")
                time.sleep(0.01)
            harness.request("busy_rejection", chat("Reply with Hello", 8), expected=429)
            check(harness.get("/readyz")[0] == 200, "busy engine should remain live/ready")
            # Read actual content before disconnecting, proving active decode
            # cancellation rather than merely cancelling a never-started job.
            saw_content = False
            while time.perf_counter() < deadline:
                line = response.readline()
                if not line:
                    break
                if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                    event = json.loads(line[5:].strip())
                    if any(c.get("delta", {}).get("content") for c in event.get("choices", [])):
                        saw_content = True
                        break
            check(saw_content, "did not receive a real generated delta before disconnect")
            response.close()
        finally:
            connection.close()
        started = time.perf_counter()
        after = harness.idle(timeout=120)
        check(after.get("cancelled_requests", 0) >= before.get("cancelled_requests", 0) + 1, "disconnect was not recorded as cancellation")
        recovery = harness.request("after_disconnect_recovery", chat("What is 19 plus 23? Answer with only the integer.", 64))
        check(answer(recovery).strip() == "42", f"cancelled request contaminated recovery: {answer(recovery)!r}")
        return {"cancellation_cleanup_seconds": time.perf_counter() - started - harness.report["records"][-1]["http_seconds"], "cancelled_before": before.get("cancelled_requests"), "cancelled_after": after.get("cancelled_requests"), "saw_content_before_disconnect": saw_content}
    harness.run("busy_disconnect_cancellation_recovery", busy_cancel)

    def prefill_cancel():
        before = harness.idle()
        connection = harness.connection()
        # Do not call getresponse(): the API intentionally waits for its first
        # native event before committing a streamed HTTP200. Disconnect while
        # this populated prompt is still being prefetched/evaluated.
        body = chat("Read these inventory notes:\n" + "The rack contains oak doors and brass hinges.\n" * 600 + "\nSummarize the inventory notes in two sentences.", 128, stream=True)
        encoded = json.dumps(body).encode("utf-8")
        try:
            connection.request("POST", "/v1/chat/completions", encoded, {"Content-Type": "application/json"})
            deadline = time.perf_counter() + 20
            while not harness.health().get("busy"):
                check(time.perf_counter() < deadline, "populated request did not become busy")
                time.sleep(0.01)
            # Move beyond admission/tokenization into the measured multi-second
            # populated prefill. Do not mistake an immediate queued cancel for
            # cancellation of an executing GPU batch.
            time.sleep(1.0)
            check(harness.health().get("busy"), "populated request finished before prefill disconnect")
        finally:
            connection.close()
        started = time.perf_counter()
        after = harness.idle(timeout=120)
        check(after.get("cancelled_requests", 0) >= before.get("cancelled_requests", 0) + 1, "prefill disconnect was not recorded as cancellation")
        cleanup = time.perf_counter() - started
        recovery = harness.request("after_prefill_disconnect_recovery", chat("What is 7 times 8? Reply with only the integer.", 48))
        check(answer(recovery).strip() == "56", "prefill cancellation contaminated recovery")
        return {"request_sha256": hashlib.sha256(encoded).hexdigest(), "request_bytes": len(encoded), "seconds_before_disconnect": 1.0, "cancellation_cleanup_seconds": cleanup, "cancelled_before": before.get("cancelled_requests"), "cancelled_after": after.get("cancelled_requests"), "scope": "Disconnect one second after busy admission during a populated prompt, without reading response headers"}
    harness.run("prefill_disconnect_cancellation_recovery", prefill_cancel)

    def overflow():
        context = harness.args.context
        # This literal is a normal vocabulary word plus a separator; empirical
        # token count is checked by a short calibration before overflowing.
        probe = {"model": MODEL, "prompt": "record " * 128, "max_tokens": 1, "temperature": 0}
        calibrated = harness.request("overflow_calibration", probe, "/v1/completions")
        per_item = calibrated["usage"]["prompt_tokens"] / 128
        repetitions = math.ceil((context + 1024) / per_item)
        rejected = harness.request("context_overflow", {**probe, "prompt": "record " * repetitions}, "/v1/completions", expected=400)
        check("error" in rejected, "overflow requires structured error")
        recovered = harness.request("after_overflow_recovery", chat("Reply with only READY.", 16))
        check(answer(recovered).strip() == "READY", "overflow corrupted next request")
        return {"configured_context": context, "probe_tokens": calibrated["usage"]["prompt_tokens"], "overflow_repetitions": repetitions}
    harness.run("context_overflow_and_recovery", overflow)


def test_benchmark(harness):
    check(not harness.health()["identity"].get("target_logits_trace", False), "diagnostic tracing invalidates benchmark timing")
    body = chat(BENCHMARK_PROMPT, harness.args.decode_tokens, temperature=harness.args.benchmark_temperature)
    warm = harness.request("benchmark_warmup", chat("Explain three practical ways to label and organize a small tool cabinet.", 96))
    check(answer(warm).strip(), "empty benchmark warmup")
    samples = []
    for index in range(harness.args.repeats):
        record = harness.stream(f"short_decode_{index + 1}", body)
        check(record["usage"]["completion_tokens"] >= 256, "decode benchmark ended before 256 tokens; speed sample is too short")
        check(record["finish_reason"] == "length", "expected controlled max-token benchmark length")
        samples.append(record)
    check(len({sample["text_sha256"] for sample in samples}) == 1, "benchmark repeats changed greedy output")
    summary = {
        "repeats": len(samples),
        "temperature": harness.args.benchmark_temperature,
        "prompt_tokens": samples[0]["usage"]["prompt_tokens"],
        "completion_tokens_each": [sample["usage"]["completion_tokens"] for sample in samples],
        "median_decode_tps": statistics.median(sample["metrics"]["decode_tps"] for sample in samples),
        "median_prefill_tps": statistics.median(sample["metrics"]["prefill_tps"] for sample in samples),
        "median_client_ttft_seconds": statistics.median(sample["client_ttft_seconds"] for sample in samples),
        "median_http_seconds": statistics.median(sample["http_seconds"] for sample in samples),
        "text_sha256": samples[0]["text_sha256"],
    }
    harness.report["benchmark_summary"] = summary
    harness.save()
    return summary


def test_long_context(harness, target_tokens):
    def records(count):
        # Stable filler; key records deliberately distinguish content retrieval
        # from a familiar fact or a mnemonic repeated in the final question.
        values = [f"Stock row {i:05d}: oak trim, quantity {(i * 17) % 89 + 1}, bin {(i * 7) % 113:03d}." for i in range(count)]
        values[0] = "Private audit key ALPHA = MAPLE-7319."
        values[len(values) // 2] = "Private audit key BETA = BRASS-4826."
        values[-1] = "Private audit key GAMMA = CEDAR-9052."
        return "\n".join(values)

    def build(count):
        return chat("Read the following stock ledger. Remember the three private audit keys; all other rows are irrelevant.\n\n" + records(count) + "\n\nWhat are the values of private audit keys ALPHA, BETA, and GAMMA? Reply with only the three values, in that order, separated by a comma and a space.", 96)

    calibration = harness.request(f"context_{target_tokens}_calibration", {**build(64), "max_tokens": 1})
    estimated_per_row = max(1, (calibration["usage"]["prompt_tokens"] - 100) / 64)
    count = max(8, int((target_tokens - 200) / estimated_per_row))
    # Avoid exceeding configured context on an estimate; every actual count is
    # retained. Adjust once if the first populated request is more than 10% low.
    body = build(count)
    response = harness.request(f"retrieval_target_{target_tokens}", body)
    observed_tokens = response["usage"]["prompt_tokens"]
    if observed_tokens < target_tokens * 0.9:
        count = int(count * (target_tokens - 150) / observed_tokens)
        body = build(count)
        response = harness.request(f"retrieval_target_{target_tokens}_adjusted", body)
        observed_tokens = response["usage"]["prompt_tokens"]
    observed = answer(response).strip()
    expected = "MAPLE-7319, BRASS-4826, CEDAR-9052"
    check(observed == expected, f"populated {observed_tokens} token retrieval failed: {observed!r}")
    check(observed_tokens >= target_tokens * 0.85, f"context populated only {observed_tokens}, target {target_tokens}")
    check(observed_tokens + response["usage"]["completion_tokens"] <= harness.args.context, "usage exceeds context")
    detail = {"target_tokens": target_tokens, "actual_prompt_tokens": observed_tokens, "rows": count, "expected": expected, "observed": observed, "metrics": response["metrics"], "scope": "One three-key retrieval at front/middle/end; not general long-context quality validation"}
    if target_tokens <= 8192:
        extended = chat(
            "Read this stock ledger, remembering the private audit keys.\n\n"
            + records(count)
            + "\n\nFirst output the three audit key values ALPHA, BETA, GAMMA in order on one line. Then write a detailed 1500-word guide to auditing a warehouse stock ledger, with concrete steps for receiving, counting, reconciling differences, checking damage, and updating records. Explain every section thoroughly.",
            harness.args.decode_tokens,
        )
        decoded = harness.stream(f"populated_{target_tokens}_decode", extended)
        check(decoded["usage"]["completion_tokens"] >= 256, "populated-context decode ended before 256 tokens")
        check(all(key in decoded["text"][:300] for key in ("MAPLE-7319", "BRASS-4826", "CEDAR-9052")), "populated-context decode did not retrieve audit keys")
        detail["populated_decode"] = {
            "usage": decoded["usage"],
            "metrics": decoded["metrics"],
            "client_ttft_seconds": decoded["client_ttft_seconds"],
            "text_sha256": decoded["text_sha256"],
        }
    return detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--suite", choices=("smoke", "quality", "contract", "acceptance", "bench", "long-context", "all"), default="acceptance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=384)
    parser.add_argument("--benchmark-temperature", type=float, default=0)
    parser.add_argument("--long-targets", type=int, nargs="+", default=[4096, 30000])
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.repeats <= 10 or not 256 <= args.decode_tokens <= 4096:
        parser.error("port/repeats/decode-tokens outside bounded range")
    if not 0 <= args.benchmark_temperature <= 2:
        parser.error("benchmark-temperature must be0..2")
    if args.output.exists():
        parser.error("output exists; choose a new report path to preserve prior evidence")
    if any(value < 256 or value > args.context - 1024 for value in args.long_targets):
        parser.error("long-targets must leave at least 1024 tokens of context headroom")
    harness = Harness(args)
    try:
        status, initial = harness.get("/readyz")
        check(status == 200 and not initial.get("busy"), f"engine not ready/idle: {status} {initial}")
        harness.report["initial_health"] = initial
        status, models = harness.get("/v1/models")
        check(status == 200 and any(model["id"] == MODEL for model in models.get("data", [])), f"model identity mismatch: {models}")
        harness.report["models"] = models
        harness.save()
        if args.suite == "smoke":
            def smoke():
                response = harness.request("smoke_arithmetic", chat("A carpenter cuts a 240 cm board into three equal pieces. How long is each piece? Reply with only the number of centimeters.", 64))
                check(answer(response).strip() == "80", repr(answer(response)))
                return {"observed": answer(response), "usage": response["usage"], "metrics": response["metrics"]}
            harness.run("smoke_arithmetic", smoke)
        if args.suite in ("quality", "acceptance", "all"):
            test_quality(harness)
        if args.suite in ("contract", "acceptance", "all"):
            test_contract(harness)
        if args.suite in ("bench", "all"):
            harness.run("warm_repeated_decode_benchmark", lambda: test_benchmark(harness))
        if args.suite in ("long-context", "all"):
            for target in args.long_targets:
                harness.run(f"populated_context_{target}_retrieval", lambda target=target: test_long_context(harness, target))
        harness.report["final_health"] = harness.idle()
        harness.report["passed"] = bool(harness.report["checks"]) and all(item["passed"] for item in harness.report["checks"])
    except Exception as error:
        harness.report["fatal_error"] = str(error)
        harness.report["traceback"] = traceback.format_exc(limit=6)
    finally:
        harness.report["finished_utc"] = utc()
        harness.save()
    print(json.dumps({"output": str(args.output.resolve()), "passed": harness.report["passed"], "checks": len(harness.report["checks"]), "failed": [item["name"] for item in harness.report["checks"] if not item["passed"]]}, ensure_ascii=True), flush=True)
    return 0 if harness.report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
