#!/usr/bin/env python3
"""Download exactly the pinned Qwen GGUF, with resumable HTTP and SHA256 verification.

Uses only Python's standard library. Weights and the partial download stay in the
repository's ignored target directory; no Hugging Face cache copy is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
LOCKS = {"q4": "model.lock.json", "q8": "model-q8.lock.json"}


def selected_lock(quant: str | None = None, path: Path | None = None) -> tuple[Path, dict]:
    """Select one receipt-bound lock without changing the existing Q4 lock."""
    lock_path = path or ROOT / "ports/qwen-resident" / LOCKS[quant or "q8"]
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    filename = lock["filename"]
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("lock filename must be a single GGUF filename")
    if not filename.endswith(".gguf") or lock["size_bytes"] <= 0:
        raise ValueError("invalid pinned GGUF filename or size")
    if len(lock["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in lock["sha256"]):
        raise ValueError("lock requires a lowercase SHA256")
    return lock_path, lock


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, lock: dict) -> str:
    actual_size = path.stat().st_size
    if actual_size != lock["size_bytes"]:
        raise ValueError(f"size mismatch: {actual_size} != {lock['size_bytes']}")
    print(f"Verifying SHA256: {path}", flush=True)
    digest = sha256(path)
    if digest != lock["sha256"]:
        raise ValueError(f"SHA256 mismatch: {digest} != {lock['sha256']}; retained for inspection")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--attempts", type=int, default=12)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--quant", choices=LOCKS, help="pinned quantization (default: q8)")
    selector.add_argument("--lock", type=Path, help="explicit pinned model lock JSON")
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    lock_path, lock = selected_lock(args.quant, args.lock)
    dest = ROOT / "target/models/Qwen3.8-27B" / lock["filename"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    expected = lock["size_bytes"]
    if not dest.exists():
        if args.verify_only:
            raise FileNotFoundError(dest)
        url = f"https://huggingface.co/{lock['repository']}/resolve/{lock['revision']}/{lock['filename']}"
        started = time.monotonic()
        initial = partial.stat().st_size if partial.exists() else 0
        for attempt in range(1, args.attempts + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected:
                raise ValueError(f"partial file exceeds pinned size: {partial}")
            if offset == expected:
                break
            try:
                request = urllib.request.Request(
                    f"{url}?download=true&request_time={time.time_ns()}",
                    headers={"Range": f"bytes={offset}-", "User-Agent": "qwen-resident-pinned-download/1"},
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status == 206:
                        prefix = f"bytes {offset}-"
                        if not content_range.startswith(prefix) or not content_range.endswith(f"/{expected}"):
                            raise ValueError(f"unexpected Content-Range: {content_range}")
                    elif response.status != 200 or offset != 0:
                        raise ValueError(f"server did not honor resume at {offset}: HTTP {response.status}")
                    reported = response.headers.get("Content-Length")
                    if reported and int(reported) != expected - offset:
                        raise ValueError(f"unexpected Content-Length: {reported}")
                    print(f"Download attempt {attempt}: resuming at {offset:,}/{expected:,} bytes", flush=True)
                    with partial.open("ab") as output:
                        last_report = time.monotonic()
                        while block := response.read(8 * 1024 * 1024):
                            if offset + len(block) > expected:
                                raise ValueError("response exceeds pinned size")
                            output.write(block)
                            offset += len(block)
                            now = time.monotonic()
                            if now - last_report >= 10:
                                speed = (offset - initial) / (now - started) / 1_000_000
                                print(f"{offset:,}/{expected:,} bytes ({offset / expected:.1%}); {speed:.1f} MB/s", flush=True)
                                last_report = now
                        output.flush()
                        os.fsync(output.fileno())
                if offset != expected:
                    raise OSError(f"early EOF at {offset}")
                break
            except (OSError, ValueError) as error:
                print(f"Attempt {attempt} interrupted: {error}", flush=True)
                if attempt == args.attempts:
                    raise
                time.sleep(min(attempt * 2, 20))
        digest = verify(partial, lock)
        partial.replace(dest)
    else:
        digest = verify(dest, lock)
    receipt = {
        "schema_version": 1,
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": lock["model_id"],
        "repository": lock["repository"],
        "revision": lock["revision"],
        "filename": lock["filename"],
        "size_bytes": dest.stat().st_size,
        "sha256": digest,
        "file_mtime_ns": dest.stat().st_mtime_ns,
        "file_mtime_100ns": dest.stat().st_mtime_ns // 100,
        "lock_sha256": sha256(lock_path),
    }
    receipt_path = dest.with_suffix(dest.suffix + ".verified.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
