"""Bounded native API cache experiment on the existing original checkpoint.

Starts only its own loopback server, runs the same France request twice, and
records exact sampled IDs, all seven next-token predictions, timings and memory.
It never changes defaults or other services.
"""
import argparse
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


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-gib", type=int, choices=range(3, 33), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()
    if not args.label or not all(c.isascii() and (c.isalnum() or c in "-_") for c in args.label):
        raise ValueError("invalid artifact label")
    output = ROOT / "target" / args.label
    output.mkdir(exist_ok=False)
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    exe = ROOT / "target/release/strix.exe"
    command = [str(exe), "serve-aten", str(ROOT / "target/models/DeepSeek-V4.1-Flash"),
               str(args.port), str(args.cache_gib), "128"]
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "cache_gib": args.cache_gib,
              "executable_sha256": digest(exe), "shim_sha256": digest(os.environ["STRIX_ATEN_SHIM"]),
              "source_sha256": {name: digest(ROOT / name) for name in (
                  "crates/strix/src/serving.rs", "crates/strix/src/decoder.rs", "crates/strix/src/aten.rs",
                  "crates/strix/native/aten.cpp", "tools/benchmark-strix-cache.py")},
              "scope": "Two original France requests in one native process; filesystem cache and other system workloads are uncontrolled. No default or arithmetic changes.",
              "command": command, "requests": [], "passed": False}
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=1800, trust_env=False) as client:
                started = time.monotonic()
                while time.monotonic() - started < 120:
                    if process.poll() is not None:
                        raise RuntimeError("native server exited; see server.log")
                    try:
                        ready = client.get("/readyz")
                        if ready.status_code == 200:
                            assert ready.json()["pid"] == process.pid
                            report["identity"] = ready.json()
                            break
                        if ready.json()["state"] == "failed":
                            raise RuntimeError(ready.text)
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.25)
                else:
                    raise TimeoutError("native readiness")
                for label in ("cold_process", "repeated_request"):
                    started = time.perf_counter()
                    response = client.post("/v1/completions", json={"model": "deepseek-ai/DeepSeek-V4.1-Flash",
                        "prompt": "The capital of France is", "max_tokens": 3, "temperature": 0})
                    elapsed = time.perf_counter() - started
                    response.raise_for_status()
                    body = response.json()
                    health = client.get("/health").json()
                    metrics = health["last_request"]["metrics"]
                    assert body["choices"][0]["text"] == " Paris." and body["choices"][0]["finish_reason"] == "stop"
                    assert metrics["sampled_token_ids"] == [11111, 16, 1]
                    assert [s["input_token"] for s in metrics["steps"]] == [671, 6102, 294, 8760, 344, 11111, 16]
                    assert [s["next_token"] for s in metrics["steps"]] == [455, 294, 8760, 344, 11111, 16, 1]
                    assert all(s["logits_finite"] for s in metrics["steps"])
                    record = {"label": label, "http_seconds": elapsed, "response": body, "health": health}
                    report["requests"].append(record)
                    save()
                    print(json.dumps({"label": label, "seconds": elapsed, "cache": metrics["weight_cache"],
                                      "memory": metrics["aten_memory"]}), flush=True)
                report["passed"] = True
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=30)
            report["server_stopped"] = process.poll() is not None
            with socket.socket() as check:
                check.settimeout(1)
                report["listener_closed"] = check.connect_ex(("127.0.0.1", args.port)) != 0
            save()


if __name__ == "__main__":
    main()
