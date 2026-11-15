"""Original-checkpoint native HTTP acceptance; owns only its server process."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
RAW = {"model": MODEL, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}
CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Reply with only OK."}], "max_tokens": 8}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--executable", type=Path, default=ROOT / "target/release/strix.exe")
    parser.add_argument("--recovery-only", action="store_true", help="bounded delivery-build check: cancel real prefill, then original The -> token 455")
    args = parser.parse_args()
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.label):
        raise ValueError("label must be letters, digits, underscores or dashes")
    output = ROOT / "target" / args.label
    output.mkdir(exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    command = [str(args.executable), "serve-aten", str(ROOT / "target/models/DeepSeek-V4.1-Flash"), str(args.port), "8", "128"]
    result = {"utc": datetime.now(timezone.utc).isoformat(), "command": command,
              "executable_sha256": sha(args.executable), "shim_sha256": sha(os.environ["STRIX_ATEN_SHIM"]),
              "source_sha256": {str(p): sha(ROOT / p) for p in (
                  "crates/strix/src/serving.rs", "crates/strix/src/serving_contract.rs", "crates/strix/src/decoder.rs", "crates/strix/src/aten.rs",
                  "crates/strix/native/aten.cpp", "ports/strix-reference/upstream/encoding.py",
                  "tools/validate-strix-native-api.py", "tools/serve-strix.ps1")}, "requests": [], "passed": False}
    with (output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result["server_pid"] = process.pid
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=1800, trust_env=False) as client:
                start = time.monotonic()
                while time.monotonic() - start < 120:
                    if process.poll() is not None:
                        raise RuntimeError(f"server exited {process.returncode}; see {output / 'server.log'}")
                    try:
                        ready = client.get("/readyz")
                        if ready.status_code == 200:
                            assert ready.json()["pid"] == process.pid, "port belongs to another process"
                            break
                        if ready.json()["state"] == "failed":
                            raise RuntimeError(ready.text)
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.25)
                else:
                    raise TimeoutError("native readiness exceeded 120 seconds")
                result["initial_health"] = ready.json()
                models = client.get("/v1/models")
                assert models.status_code == 200 and models.json()["data"][0]["id"] == MODEL

                def health():
                    response = client.get("/health")
                    response.raise_for_status()
                    return response.json()

                def save():
                    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

                def complete(name, body=RAW, path="/v1/completions"):
                    started = time.perf_counter()
                    response = client.post(path, json=body)
                    record = {"name": name, "http_seconds": time.perf_counter() - started,
                              "http_status": response.status_code, "response": response.json()}
                    result["requests"].append(record)
                    print(json.dumps({k: v for k, v in record.items()}), flush=True)
                    assert response.status_code == 200, response.text
                    record["health_after"] = health()
                    save()
                    return response.json()

                if args.recovery_only:
                    before = health()["cancelled_requests"]
                    with client.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}) as response:
                        assert response.status_code == 200
                        for line in response.iter_lines():
                            if line.startswith("data: "):
                                time.sleep(2)
                                result["disconnect_busy_before_close"] = health()["busy"]
                                assert result["disconnect_busy_before_close"]
                                break
                    started = time.monotonic()
                    while True:
                        state = health()
                        if not state["busy"] and state["cancelled_requests"] == before + 1:
                            break
                        if time.monotonic() - started > 120:
                            raise TimeoutError("disconnect did not clean up within one bounded forward")
                        time.sleep(0.1)
                    result["disconnect_cleanup_seconds"] = time.monotonic() - started
                    result["health_after_cancel"] = state
                    answer = complete("The_after_inflight_cancel", {"model": MODEL, "prompt": "The", "max_tokens": 1})
                    assert answer["usage"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    assert answer["choices"][0]["finish_reason"] == "length"
                    result["final_health"] = health()
                    metrics = result["final_health"]["last_request"]["metrics"]
                    assert metrics["prompt_token_ids"] == [671] and metrics["sampled_token_ids"] == [455]
                    assert metrics["combined_weight_cache_peak_upper_bound_bytes"] <= metrics["total_weight_cache_budget_bytes"]
                    assert not any(Path(p).name.lower().startswith("python") or Path(p).name.lower() == "torch_python.dll" for p in metrics["loaded_modules"])
                    result["scope"] = "delivery build: in-flight native cancellation and original one-token recovery; full HTTP suite recorded separately"
                    result["passed"] = True
                    return

                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(complete, "raw_A")
                    start = time.monotonic()
                    while not health()["busy"]:
                        assert time.monotonic() - start < 10
                        time.sleep(0.02)
                    started = time.perf_counter()
                    busy_health = health()
                    result["health_while_busy"] = {"seconds": time.perf_counter() - started, "state": busy_health}
                    assert busy_health["busy"] and busy_health["state"] == "ready"
                    busy = client.post("/v1/completions", json=RAW)
                    assert busy.status_code == 429
                    result["busy_rejection"] = {"status": busy.status_code, "response": busy.json()}
                    first = future.result()
                assert first["choices"][0]["text"] == " Paris."
                assert first["choices"][0]["finish_reason"] == "stop"
                assert first["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

                def stream(name, body, path, expected, expected_usage):
                    started = time.perf_counter()
                    events = []
                    with client.stream("POST", path, json={**body, "stream": True, "stream_options": {"include_usage": True}}) as response:
                        assert response.status_code == 200, response.read()
                        assert response.headers["content-type"].startswith("text/event-stream")
                        for line in response.iter_lines():
                            if line.startswith("data: "):
                                raw = line[6:]
                                event = raw if raw == "[DONE]" else json.loads(raw)
                                events.append({"seconds": time.perf_counter() - started, "data": event})
                                print(json.dumps({"name": name, "event": events[-1]}), flush=True)
                    assert events[-1]["data"] == "[DONE]"
                    chunks = [e["data"] for e in events[:-1]]
                    assert all("error" not in event for event in chunks), chunks
                    text = "".join((e["choices"][0]["delta"].get("content", "") if path.endswith("chat/completions")
                                    else e["choices"][0]["text"]) for e in chunks if e["choices"])
                    assert text == expected, repr(text)
                    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
                    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"] == expected_usage
                    record = {"name": name, "text": text, "events": events, "health_after": health()}
                    result["requests"].append(record)
                    save()
                    return record

                stream("chat_B_stream", CHAT, "/v1/chat/completions", "OK",
                       {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11})
                repeated = stream("raw_A_after_B_stream", RAW, "/v1/completions", " Paris.", first["usage"])
                assert repeated["health_after"]["last_request"]["metrics"]["sampled_token_ids"] == [11111, 16, 1]

                before = health()["cancelled_requests"]
                with client.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}) as response:
                    assert response.status_code == 200
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            # Give the native owner time to enter the first real
                            # forward; cleanup then waits for that token boundary.
                            time.sleep(2)
                            result["disconnect_busy_before_close"] = health()["busy"]
                            assert result["disconnect_busy_before_close"]
                            break
                started = time.monotonic()
                while True:
                    state = health()
                    if not state["busy"] and state["cancelled_requests"] == before + 1:
                        break
                    if time.monotonic() - started > 120:
                        raise TimeoutError("disconnect did not cancel within one bounded forward")
                    time.sleep(0.1)
                result["disconnect_cleanup_seconds"] = time.monotonic() - started
                result["health_after_cancel"] = state
                # Recovery and stop accounting share one real request.
                stream("raw_after_cancel_user_stop", {**RAW, "stop": "."}, "/v1/completions", " Paris",
                       {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
                invalid = client.post("/v1/completions", json={**RAW, "prompt": "hello " * 130})
                assert invalid.status_code == 400 and "context" in invalid.json()["error"]["message"]
                result["context_limit"] = {"status": invalid.status_code, "response": invalid.json()}
                result["final_health"] = health()
                assert result["final_health"]["state"] == "ready" and not result["final_health"]["busy"]
                modules = result["final_health"]["last_request"]["metrics"]["loaded_modules"]
                assert not any(Path(p).name.lower().startswith("python") or Path(p).name.lower() == "torch_python.dll" for p in modules)
                result["passed"] = True
        finally:
            if process.poll() is None:
                process.terminate()  # This exact child only; releases its GPU resources.
            process.wait(timeout=30)
            result["server_stopped"] = process.poll() is not None
            for _ in range(50):
                with socket.socket() as probe:
                    probe.settimeout(0.1)
                    result["listener_closed"] = probe.connect_ex(("127.0.0.1", args.port)) != 0
                if result["listener_closed"]:
                    break
                time.sleep(0.1)
            (output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
