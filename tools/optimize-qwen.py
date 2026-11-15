"""Reproducible Qwen optimization checks against an already-running native engine.

Serial GPU work only; no server lifecycle changes. Reports preserve prompts by
hash plus deterministic generator inputs. Long-context checks combine randomized
front/middle/end retrieval, a ledger reconciliation, and sustained decoding.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import re
import socket
import threading
import time


spec = importlib.util.spec_from_file_location("qwen_validation", Path(__file__).with_name("validate-qwen.py"))
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def ledger(count, seed):
    rng = random.Random(seed)
    keys = ["".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(12)) for _ in range(3)]
    opening, received, shipped, damaged = [rng.randint(11, 79) for _ in range(4)]
    opening += 100
    expected = {"front": keys[0], "middle": keys[1], "end": keys[2], "closing_stock": opening + received - shipped - damaged}
    # Each target is placed at a seeded random position within a distinct depth
    # band. Target keys never appear in the final question or system message.
    positions = [rng.randint(count // 100, count // 20), rng.randint(count * 45 // 100, count * 55 // 100), rng.randint(count * 95 // 100, count * 99 // 100)]
    rows = [f"Stock row {i:06d}: oak trim, quantity {(i * 17) % 89 + 1}, bin {(i * 7) % 113:03d}." for i in range(count)]
    rows[positions[0]] = f"AUDIT FRONT: secret {keys[0]}. Ledger for special SKU ZX: opening stock {opening} units."
    rows[positions[1]] = f"AUDIT MIDDLE: secret {keys[1]}. Ledger for special SKU ZX: received {received} units and shipped {shipped} units."
    rows[positions[2]] = f"AUDIT END: secret {keys[2]}. Ledger for special SKU ZX: scrapped {damaged} damaged units. No other ZX adjustments occurred."
    prompt = (
        "Read the complete warehouse ledger. Preserve the three AUDIT secrets and reconcile the special SKU ZX using all its transactions. Other stock rows describe different products.\n\n"
        + "\n".join(rows)
        + '\n\nFirst output one JSON object on a single line with exactly these keys: "front", "middle", "end" (the three AUDIT secrets), and "closing_stock" (integer final stock of ZX after every transaction). '
        + "Then write a detailed 1500-word practical audit procedure for receiving, counting, reconciling discrepancies, damage records, and sign-off. Continue with concrete steps and examples until each area is covered."
    )
    return validation.chat(prompt, 384, presence_penalty=0), expected, positions


def extended(harness):
    args = harness.args
    body, _, _ = ledger(256, args.fixture_seed)
    calibration = harness.request("calibrate_ledger_rows", {**body, "max_tokens": 1})
    # Fixed six-digit row identifiers have stable segmentation. Leave 2% for
    # estimation error; actual prompt usage, never the requested target, is proof.
    per_row = (calibration["usage"]["prompt_tokens"] - 200) / 256
    count = max(256, int((args.target_tokens * 0.98 - 300) / per_row))
    body, expected, positions = ledger(count, args.fixture_seed)
    body["max_tokens"] = args.decode_tokens
    harness.report["fixture"] = {
        "generator": "ledger_v1", "seed": args.fixture_seed, "rows": count,
        "target_tokens": args.target_tokens, "target_positions_rows": positions,
        "target_depth_fractions": [p / count for p in positions], "expected": expected,
        "request_sha256": validation.digest(json.dumps(body, ensure_ascii=False)),
    }
    harness.save()
    stop = threading.Event()
    started = time.perf_counter()

    def progress():
        while not stop.wait(30):
            try:
                health = harness.health()
                print(json.dumps({"state": "long_context_in_progress", "elapsed_seconds": round(time.perf_counter() - started), "target_tokens": args.target_tokens, "busy": health.get("busy"), "engine_state": health.get("state")}), flush=True)
            except Exception as error:
                print(json.dumps({"state": "progress_check_error", "error": str(error)}), flush=True)

    observer = threading.Thread(target=progress, daemon=True)
    observer.start()
    try:
        result = harness.stream("random_retrieval_ledger_and_sustained_decode", body)
    finally:
        stop.set()
        observer.join(5)
    actual = result["usage"]["prompt_tokens"]
    match = re.search(r"\{[^{}]+\}", result["text"][:600])
    observed = json.loads(match.group()) if match else None
    # Save all observations before assertions, including a confidently wrong key.
    detail = {"target_tokens": args.target_tokens, "actual_prompt_tokens": actual,
              "expected": expected, "observed": observed, "usage": result["usage"],
              "metrics": result["metrics"], "client_ttft_seconds": result["client_ttft_seconds"],
              "scope": "One randomized three-depth retrieval plus multi-record arithmetic and sustained procedural output; not broad long-context quality."}
    harness.report["extended_summary"] = detail
    harness.save()
    validation.check(actual >= args.target_tokens * 0.90, f"populated length {actual} below target range")
    validation.check(actual + result["usage"]["completion_tokens"] <= args.context, "context budget exceeded")
    validation.check(observed == expected, f"retrieval/reconciliation mismatch: {observed!r} != {expected!r}")
    validation.check(result["usage"]["completion_tokens"] >= 256 and result["finish_reason"] == "length", "not a sustained decode sample")
    validation.check(len(result["text"][match.end():].split()) >= 80, "procedural output too short")
    return detail


def parity(harness):
    cases = [
        ("greedy_arithmetic", validation.chat("Compute 37 * 19 - 46. Reply with the integer only.", 32)),
        ("greedy_prose", validation.chat("Explain how a warehouse receiving count should be reconciled with a purchase order. Give concrete steps and examples.", 192, presence_penalty=0)),
        ("greedy_code", validation.chat("Write Python function normalize_sku(text) that strips spaces at either end and uppercases ASCII letters, preserving internal spaces and digits. Return only code.", 128)),
        ("sampled_penalties", validation.chat("List practical inspection steps for a wooden exterior door.", 128, temperature=0.7, top_k=20, top_p=0.8, presence_penalty=1.5, frequency_penalty=0.3, repeat_penalty=1.05)),
        ("sampled_unfiltered", validation.chat("Describe a peaceful forest workshop.", 128, temperature=0.8, top_k=0, top_p=1, presence_penalty=0)),
        ("multiturn", {**validation.chat("unused", 64), "messages": [{"role": "user", "content": "We have 24 oak doors and 16 pine doors."}, {"role": "assistant", "content": "Recorded: 24 oak and 16 pine doors."}, {"role": "user", "content": "Ship 7 oak and receive 9 pine. Reply with only remaining oak and pine counts in that order, separated by a comma."}]}),
        ("cross_token_stop", validation.chat("Write exactly: red, green, blue, orange, violet.", 96, stop="blue")),
        ("eos", validation.chat("Reply with exactly OK and nothing else.", 64)),
        ("length_one", validation.chat("Continue writing about wood grain.", 1)),
    ]
    results = {}
    for name, body in cases:
        result = harness.request(name, body)
        results[name] = {"text": validation.answer(result), "usage": result["usage"], "finish_reason": result["choices"][0]["finish_reason"], "metrics": result["metrics"]}
    harness.report["parity_outputs"] = results
    harness.save()
    if harness.args.compare:
        baseline = json.loads(harness.args.compare.read_text(encoding="utf-8-sig"))["parity_outputs"]
        comparisons = {name: {field: result[field] == baseline[name][field] for field in ("text", "usage", "finish_reason")} for name, result in results.items()}
        harness.report["parity_comparison"] = comparisons
        harness.save()
        validation.check(all(all(fields.values()) for fields in comparisons.values()), f"target parity differs: {comparisons}")
    return {"cases": len(results), "compared": str(harness.args.compare) if harness.args.compare else None}


def cancel_populated(harness):
    source = json.loads(harness.args.cancel_from.read_text(encoding="utf-8-sig"))
    fixture = source["fixture"]
    validation.check(source["passed"], "cancellation fixture must have passed populated-context acceptance")
    body, _, _ = ledger(fixture["rows"], fixture["seed"])
    body["max_tokens"] = harness.args.decode_tokens
    validation.check(validation.digest(json.dumps(body, ensure_ascii=False)) == fixture["request_sha256"], "cancellation fixture changed")
    control_body = validation.chat("Reply with only the result of 123 + 456.", 32)
    before = harness.request("before_long_prefill_cancel", control_body)
    cancelled_before = harness.health()["cancelled_requests"]
    connection = harness.connection()
    encoded = json.dumps({**body, "stream": True}).encode("utf-8")
    started = time.perf_counter()
    try:
        connection.request("POST", "/v1/chat/completions", encoded, {"Content-Type": "application/json"})
        while not harness.health()["busy"]:
            validation.check(time.perf_counter() - started < 10, "long prefill did not enter busy state")
            time.sleep(0.05)
        time.sleep(5)
        disconnected = time.perf_counter()
        connection.sock.shutdown(socket.SHUT_RDWR)
    finally:
        connection.close()
    health = harness.idle(timeout=180)
    cleanup = time.perf_counter() - disconnected
    validation.check(health["cancelled_requests"] > cancelled_before, "long prefill disconnect was not cancelled")
    after = harness.request("after_long_prefill_cancel", control_body)
    validation.check(validation.answer(before) == validation.answer(after) == "579", "cancelled recurrent state changed recovery answer")
    validation.check(before["usage"] == after["usage"], "cancel recovery changed usage")
    return {"known_fixture_prompt_tokens": source["extended_summary"]["actual_prompt_tokens"],
            "fixture_report": str(harness.args.cancel_from), "request_bytes": len(encoded),
            "request_sha256": validation.digest(encoded.decode()),
            "seconds_before_disconnect": disconnected - started,
            "cancellation_cleanup_seconds": cleanup, "recovery_answer": validation.answer(after)}


def speculative_contract(harness):
    """Check sampling/isolation within one mode, independently of cross-mode RNG."""
    cases = [
        ("filtered_penalties", dict(temperature=0.7, top_k=20, top_p=0.8, presence_penalty=1.5, frequency_penalty=0.3, repeat_penalty=1.05)),
        ("unfiltered", dict(temperature=0.8, top_k=0, top_p=1.0, presence_penalty=0)),
    ]
    details = []
    for name, parameters in cases:
        body = validation.chat("Describe practical inspection steps for a wooden exterior door. Give concrete examples.", 128, **parameters)
        first = harness.request(f"{name}_before", body)
        unrelated = validation.chat("Write a scene about a lighthouse keeper finding a silver key.", 96, temperature=0.9, seed=12345, presence_penalty=0)
        harness.request(f"{name}_interleaved_other_request", unrelated)
        repeated = harness.request(f"{name}_after", body)
        streamed = harness.stream(f"{name}_stream", body)
        text = validation.answer(first)
        validation.check(text == validation.answer(repeated) == streamed["text"], f"{name}: seeded output changed across isolation/streaming")
        validation.check(first["usage"] == repeated["usage"] == streamed["usage"], f"{name}: usage changed across isolation/streaming")
        validation.check(first["choices"][0]["finish_reason"] == repeated["choices"][0]["finish_reason"] == streamed["finish_reason"], f"{name}: termination changed")
        for result in (first, repeated, streamed):
            metrics = result["metrics"]
            validation.check(metrics["decode_tokens"] == result["usage"]["completion_tokens"] - 1, "draft or rejected tokens leaked into output accounting")
            validation.check(0 <= metrics["accepted_draft_tokens"] <= min(metrics["drafted_tokens"], metrics["decode_tokens"]), "invalid draft counters")
        details.append({"case": name, "text_sha256": validation.digest(text), "usage": first["usage"], "metrics": first["metrics"]})
    # Exercise the transition between ordinary tail decoding and a full draft
    # block, including budgets smaller than every supported draft depth.
    for budget in range(1, 10):
        result = harness.request(f"output_budget_{budget}", validation.chat("Write a detailed 1500-word maintenance handbook for exterior doors.", budget))
        validation.check(result["usage"]["completion_tokens"] == budget, f"output budget {budget} was not honored")
        validation.check(result["choices"][0]["finish_reason"] == "length", "budget boundary has wrong finish reason")
    return {"sampling_cases": details, "output_budgets": list(range(1, 10)), "scope": "Within-mode reproducibility, request isolation, streaming equivalence, output accounting and draft-tail boundaries; distribution testing is separate."}


def trace(harness):
    validation.check(harness.health()["identity"].get("target_logits_trace") is True, "trace suite requires explicit native diagnostic tracing")
    result = harness.request("greedy_logits_diagnostic", validation.chat(validation.BENCHMARK_PROMPT, harness.args.decode_tokens))
    validation.check(result["metrics"]["benchmark_timing_valid"] is False, "trace timings must be marked invalid")
    return {"text_sha256": validation.digest(validation.answer(result)), "usage": result["usage"],
            "scope": "Diagnostic output/logit comparison only; timings are not performance evidence. Native stderr contains bounded target logits."}


def deadline(harness):
    source = json.loads(harness.args.cancel_from.read_text(encoding="utf-8-sig"))
    validation.check(source["passed"], "deadline fixture must have passed extended acceptance")
    seconds = harness.health()["request_limits"]["deadline_seconds"]
    validation.check(1 <= seconds <= 10, "deadline check requires a deliberately short 1..10 second server deadline")
    fixture = source["fixture"]
    body, _, _ = ledger(fixture["rows"], fixture["seed"])
    body["max_tokens"] = harness.args.decode_tokens
    validation.check(validation.digest(json.dumps(body, ensure_ascii=False)) == fixture["request_sha256"], "deadline fixture changed")
    cancelled_before = harness.health()["cancelled_requests"]
    start = time.perf_counter()
    result = harness.request("populated_prefill_deadline", body, expected=504)
    response_seconds = time.perf_counter() - start
    validation.check("deadline" in result["error"]["message"], "unexpected timeout response")
    health = harness.idle(timeout=180)
    validation.check(health["cancelled_requests"] > cancelled_before, "deadline did not cancel native work")
    recovered = harness.request("after_deadline_recovery", validation.chat("Reply with only the result of 123 + 456.", 32))
    validation.check(validation.answer(recovered) == "579", "deadline recovery answer changed")
    return {"known_fixture_prompt_tokens": source["extended_summary"]["actual_prompt_tokens"], "deadline_seconds": seconds,
            "http_504_seconds": response_seconds, "recovery_answer": validation.answer(recovered)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("extended", "parity", "cancel", "spec-contract", "trace", "deadline"), required=True)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--timeout", type=float, default=10800)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--target-tokens", type=int, default=60000)
    parser.add_argument("--decode-tokens", type=int, default=384)
    parser.add_argument("--fixture-seed", type=int, default=20261115)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--cancel-from", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve earlier evidence")
    if args.suite == "extended" and not 4096 <= args.target_tokens <= args.context - 2048:
        parser.error("target must leave at least 2048 tokens of context headroom")
    if not 256 <= args.decode_tokens <= 2048:
        parser.error("decode-tokens must be 256..2048")
    if args.suite in ("cancel", "deadline") and not args.cancel_from:
        parser.error("cancel/deadline requires --cancel-from with an accepted extended report")
    harness = validation.Harness(args)
    harness.report["optimization_script_sha256"] = validation.digest(Path(__file__).read_text(encoding="utf-8"))
    try:
        initial = harness.idle()
        validation.check(initial["state"] == "ready", "engine is not ready")
        validation.check(initial["identity"]["context"] == args.context, "configured context mismatch")
        harness.report["initial_health"] = initial
        harness.save()
        operation = {"extended": extended, "parity": parity, "cancel": cancel_populated, "spec-contract": speculative_contract, "trace": trace, "deadline": deadline}[args.suite]
        harness.run(args.suite, lambda: operation(harness))
        harness.report["final_health"] = harness.idle()
        harness.report["passed"] = all(check["passed"] for check in harness.report["checks"])
    finally:
        harness.report["finished_utc"] = validation.utc()
        harness.save()
    print(json.dumps({"passed": harness.report["passed"], "output": str(args.output)}), flush=True)
    return 0 if harness.report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
