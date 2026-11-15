"""Opt-in actual-checkpoint HTTP acceptance. Starts/stops only its own server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
PROMPT = "The capital of France is"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retain-gib", type=float, default=0)
    parser.add_argument("--prefill-batch", type=int, default=1)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    prefix = ROOT / "target" / args.label
    audit = prefix.with_suffix(".jsonl")
    if audit.exists():
        raise RuntimeError("choose a fresh label; evidence is not overwritten")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    command = [sys.executable, "-u", str(ROOT / "ports/strix-reference/serve.py"),
               "--checkpoint", str(ROOT / "target/models/DeepSeek-V4.1-Flash"),
               "--cache-gib", "8", "--retain-gib", str(args.retain_gib),
               "--prefill-batch", str(args.prefill_batch), "--port", str(args.port),
               "--audit-log", str(audit)]
    result = {"utc": datetime.now(timezone.utc).isoformat(), "command": command, "requests": []}
    raw = {"model": MODEL, "prompt": PROMPT, "max_tokens": 8, "temperature": 0}
    with prefix.with_suffix(".log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result["server_pid"] = process.pid
        try:
            with httpx.Client(base_url=base, timeout=1800, trust_env=False) as client:
                for _ in range(120):
                    if process.poll() is not None:
                        raise RuntimeError(f"server exited {process.returncode}; inspect {prefix}.log")
                    try:
                        ready = client.get("/readyz")
                        if ready.status_code == 200:
                            # Windows venv python.exe may be a redirector parent.
                            assert process.pid in (ready.json()["pid"], ready.json()["parent_pid"]), \
                                "port belongs to another process"
                            result["listener_pid"] = ready.json()["pid"]
                            break
                        if ready.json()["state"] == "failed":
                            raise RuntimeError(ready.text)
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.5)
                else:
                    raise TimeoutError("model initialization exceeded 60 seconds")
                result["identity"] = ready.json()["identity"]
                models = client.get("/v1/models")
                assert models.status_code == 200 and models.json()["data"][0]["id"] == MODEL

                def complete(name, body=raw, path="/v1/completions"):
                    started = time.perf_counter()
                    response = client.post(path, json=body)
                    record = {"name": name, "http_status": response.status_code,
                              "http_seconds": time.perf_counter() - started, "body": response.json()}
                    result["requests"].append(record)
                    print(json.dumps(record), flush=True)
                    assert response.status_code == 200, response.text
                    return record["body"]

                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(complete, "raw_A")
                    for _ in range(100):
                        if client.get("/healthz").json()["busy"]:
                            break
                        time.sleep(0.05)
                    time.sleep(2)  # Probe while real GPU prefill is underway.
                    health_start = time.perf_counter()
                    health = client.get("/healthz")
                    result["health_while_busy"] = {"status": health.status_code,
                        "busy": health.json()["busy"], "seconds": time.perf_counter() - health_start}
                    assert health.status_code == 200 and health.json()["busy"]
                    busy = client.post("/v1/completions", json=raw)
                    result["busy_rejection"] = {"status": busy.status_code, "body": busy.json()}
                    assert busy.status_code == 429
                    first = future.result()
                assert first["choices"][0]["text"] == " Paris."
                assert first["choices"][0]["finish_reason"] == "stop"
                assert first["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

                if args.acceptance:
                    # A, unrelated B, A on one persistent engine: exact raw output
                    # and usage repeat, with no model reinitialization between jobs.
                    chat_body = {"model": MODEL, "messages": [{"role": "user", "content": "Reply with only OK."}],
                                 "max_tokens": 8, "temperature": 0, "stream": True,
                                 "stream_options": {"include_usage": True}}
                    started = time.perf_counter()
                    events = []
                    with client.stream("POST", "/v1/chat/completions", json=chat_body) as response:
                        assert response.status_code == 200
                        assert response.headers["content-type"] == "text/event-stream"
                        for line in response.iter_lines():
                            if line.startswith("data: "):
                                data = line[6:]
                                events.append({"seconds": time.perf_counter() - started,
                                               "data": data if data == "[DONE]" else json.loads(data)})
                                print(json.dumps({"chat_stream_event": events[-1]}), flush=True)
                    assert events[-1]["data"] == "[DONE]"
                    assert not any("error" in e["data"] for e in events if isinstance(e["data"], dict))
                    content = "".join(e["data"]["choices"][0]["delta"].get("content", "")
                                      for e in events[:-1] if e["data"]["choices"])
                    assert content and "<｜" not in content and "think>" not in content
                    assert events[-2]["data"]["choices"] == []
                    finish = events[-3]["data"]["choices"][0]["finish_reason"]
                    assert finish in ("stop", "length")
                    if finish == "length":
                        assert events[-2]["data"]["usage"]["completion_tokens"] == chat_body["max_tokens"]
                    record = {"name": "chat_B_stream", "text": content, "events": events}
                    result["requests"].append(record)
                    print(json.dumps(record), flush=True)
                    again = complete("raw_A_after_B")
                    assert again["choices"] == first["choices"] and again["usage"] == first["usage"]

                    # Raw stream: stop sequence held across streamed tokens; usage
                    # counts the sampled stop token, without exposing it as text.
                    started = time.perf_counter()
                    events = []
                    with client.stream("POST", "/v1/completions", json={**raw, "stream": True, "stop": ".",
                                       "stream_options": {"include_usage": True}}) as response:
                        assert response.status_code == 200
                        for line in response.iter_lines():
                            if line.startswith("data: "):
                                data = line[6:]
                                events.append({"seconds": time.perf_counter() - started,
                                               "data": data if data == "[DONE]" else json.loads(data)})
                    text = "".join(e["data"]["choices"][0]["text"] for e in events[:-1] if e["data"]["choices"])
                    assert text == " Paris" and events[-1]["data"] == "[DONE]"
                    assert events[-3]["data"]["choices"][0]["finish_reason"] == "stop"
                    assert events[-2]["data"]["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
                    result["requests"].append({"name": "raw_stream_user_stop", "text": text, "events": events})
                    limited = complete("raw_length_cap", {**raw, "max_tokens": 1})
                    assert limited["choices"][0]["text"] == " Paris"
                    assert limited["choices"][0]["finish_reason"] == "length"
                    assert limited["usage"] == {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}

                    invalid = client.post("/v1/completions", json={**raw, "prompt": "hello " * 300})
                    assert invalid.status_code == 400 and "context" in invalid.json()["error"]["message"]
                    assert client.get("/readyz").status_code == 200
                    result["context_limit_rejected"] = invalid.json()

                    # Disconnect during real prefill, then wait for layer-boundary
                    # cleanup and prove a new independent completion still works.
                    with client.stream("POST", "/v1/chat/completions", json=chat_body) as response:
                        for line in response.iter_lines():
                            if line.startswith("data: "):
                                break  # role chunk; inference is active
                    started = time.perf_counter()
                    for _ in range(120):
                        if not client.get("/healthz").json()["busy"]:
                            break
                        time.sleep(0.25)
                    else:
                        raise TimeoutError("disconnect did not clean up within 30 seconds")
                    result["disconnect_cleanup_seconds"] = time.perf_counter() - started
                    after_cancel = complete("raw_A_after_cancel")
                    assert after_cancel["choices"] == first["choices"]
                    # Ordinary chat response uses the same official output parser.
                    complete("chat_nonstream", {"model": MODEL, "messages": [{"role": "user", "content": "Reply with only OK."}],
                                                "max_tokens": 8}, "/v1/chat/completions")
                result["passed"] = True
        finally:
            # Owned subprocess only. No existing server or unrelated application is
            # touched. Windows TerminateProcess also releases this process's GPU.
            if os.name == "nt" and process.poll() is None:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            elif process.poll() is None:
                process.terminate()
            process.wait(timeout=30)
            result["server_stopped"] = process.poll() is not None
            # Windows can briefly finish a queued connect after taskkill returns.
            for _ in range(50):
                with socket.socket() as probe:
                    probe.settimeout(0.1)
                    result["listener_closed"] = probe.connect_ex(("127.0.0.1", args.port)) != 0
                if result["listener_closed"]:
                    break
                time.sleep(0.1)
            if audit.exists():
                result["audit"] = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            prefix.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {prefix}.json", flush=True)


if __name__ == "__main__":
    main()
