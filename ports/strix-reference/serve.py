"""Loopback OpenAI-compatible API for the persistent AMD PyTorch reference."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import threading
import time
import traceback
import uuid

from engine import Cancelled, InvalidRequest, MODEL_ID, ReferenceEngine, validate_request


def error(message, kind="invalid_request_error", code=None):
    return {"error": {"message": message, "type": kind, "param": None, "code": code}}


@dataclass
class Job:
    request: dict
    loop: asyncio.AbstractEventLoop
    id: str = field(default_factory=lambda: "cmpl-" + uuid.uuid4().hex)
    created: int = field(default_factory=lambda: int(time.time()))
    cancel: threading.Event = field(default_factory=threading.Event)
    events: asyncio.Queue = field(default_factory=asyncio.Queue)

    def emit(self, kind, value):
        # At most max_tokens deltas plus ready/result: the request token cap also
        # bounds this queue when a slow consumer is applying TCP backpressure.
        if not self.cancel.is_set():
            self.loop.call_soon_threadsafe(self.events.put_nowait, (kind, value))


class Worker:
    """One model and one owning thread. No waiting inference queue; busy => 429."""
    def __init__(self, factory, audit_log=None):
        self.factory, self.audit_log = factory, audit_log
        self.lock = threading.Lock()
        self.jobs = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.active = None
        self.state, self.failure, self.identity = "starting", None, None
        self.thread = threading.Thread(target=self._run, name="amd-reference-worker", daemon=False)

    def start(self):
        self.thread.start()

    def status(self):
        with self.lock:
            return {"engine": "pytorch-hip-reference", "state": self.state,
                    "pid": os.getpid(),
                    "parent_pid": os.getppid(),
                    "busy": self.active is not None, "queue_capacity": 0,
                    "error": self.failure, "identity": self.identity}

    def submit(self, job):
        with self.lock:
            if self.state != "ready":
                return 503
            if self.active is not None:
                return 429
            self.active = job
            self.jobs.put_nowait(job)
        return 200

    def _audit(self, value):
        if self.audit_log:
            with Path(self.audit_log).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **value}) + "\n")

    def _run(self):
        engine = None
        try:
            engine = self.factory()
            self._audit({"event": "initialized", "identity": engine.identity})
            with self.lock:
                self.identity, self.state = engine.identity, "ready"
            while not self.stopping.is_set():
                try:
                    job = self.jobs.get(timeout=0.1)
                except queue.Empty:
                    continue
                outcome, value = "result", None
                try:
                    engine.last_metrics = None
                    if job.cancel.is_set():
                        raise Cancelled()
                    value = engine.generate(job.request, job.cancel, job.emit)
                except Cancelled:
                    outcome = "cancelled"
                except InvalidRequest as exc:
                    outcome, value = "invalid", error(str(exc))
                except Exception as exc:
                    traceback.print_exc()
                    outcome, value = "error", error(str(exc), "engine_error")
                    with self.lock:
                        self.state, self.failure = "failed", str(exc)
                finally:
                    self._audit({"event": "request", "id": job.id, "chat": job.request["chat"],
                                 "outcome": outcome, "result": value,
                                 "metrics": engine.last_metrics})
                    # generate() has completed its finally/reset before admission
                    # reopens. A disconnected request remains busy until cleanup.
                    with self.lock:
                        self.active = None
                job.emit(outcome, value)
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.state, self.failure = "failed", str(exc)
                active, self.active = self.active, None
            if active is not None:
                active.emit("error", error(str(exc), "engine_error"))
        finally:
            if engine is not None:
                engine.close()

    def stop(self):
        with self.lock:
            self.state = "stopping"
            self.stopping.set()
            if self.active is not None:
                self.active.cancel.set()
        self.thread.join()


class LocalAPI:
    def __init__(self, worker, max_tokens=64, max_body=65536, request_timeout=1800):
        self.worker, self.max_tokens = worker, max_tokens
        self.max_body, self.request_timeout = max_body, request_timeout

    async def _json(self, send, status, data, extra=()):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()), *extra]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    self.worker.start()
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    await asyncio.to_thread(self.worker.stop)
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        path, method = scope["path"], scope["method"]
        headers = dict(scope.get("headers", []))
        if b"origin" in headers:
            await self._json(send, 403, error("browser cross-origin access is disabled"))
            return
        if method == "GET" and path in ("/healthz", "/readyz"):
            state = self.worker.status()
            status = 200 if path == "/healthz" or state["state"] == "ready" else 503
            await self._json(send, status, state)
            return
        if method == "GET" and path == "/v1/models":
            await self._json(send, 200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "deepseek-ai"}]})
            return
        if method != "POST" or path not in ("/v1/completions", "/v1/chat/completions"):
            await self._json(send, 404, error("unknown endpoint"))
            return
        if headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower() != b"application/json":
            await self._json(send, 415, error("Content-Type must be application/json"))
            return
        if self.worker.status()["busy"]:
            await self._json(send, 429, error("one inference is active; no waiting queue", "rate_limit_error", "engine_busy"),
                             [(b"retry-after", b"5")])
            return
        body = bytearray()
        try:
            async with asyncio.timeout(10):
                while True:
                    event = await receive()
                    if event["type"] == "http.disconnect":
                        return
                    body.extend(event.get("body", b""))
                    if len(body) > self.max_body:
                        await self._json(send, 413, error(f"request body exceeds {self.max_body} bytes"))
                        return
                    if not event.get("more_body", False):
                        break
            request = validate_request(json.loads(body), path == "/v1/chat/completions", self.max_tokens)
        except (ValueError, UnicodeError) as exc:
            await self._json(send, 400, error(str(exc)))
            return
        except TimeoutError:
            await self._json(send, 408, error("request body timed out"))
            return
        job = Job(request, asyncio.get_running_loop())
        status = self.worker.submit(job)
        if status != 200:
            await self._json(send, status, error("engine busy" if status == 429 else "engine not ready",
                                              "rate_limit_error" if status == 429 else "engine_error"),
                             [(b"retry-after", b"5")])
            return

        async def disconnected():
            while True:
                if (await receive())["type"] == "http.disconnect":
                    job.cancel.set()
                    return

        watch = asyncio.create_task(disconnected())
        pending = None
        streaming = False
        common = {"id": ("chat" if request["chat"] else "") + job.id,
                  "created": job.created, "model": MODEL_ID}

        def chunk(delta=None, reason=None, usage=None):
            choice = {"index": 0, "finish_reason": reason}
            choice.update({"delta": delta or {}} if request["chat"] else {"text": delta or "", "logprobs": None})
            return {**common, "object": "chat.completion.chunk" if request["chat"] else "text_completion",
                    "choices": [] if usage is not None else [choice],
                    **({"usage": usage} if usage is not None or request["include_usage"] else {})}

        async def sse(data):
            encoded = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            await send({"type": "http.response.body", "body": ("data: " + encoded + "\n\n").encode("utf-8"),
                        "more_body": True})

        try:
            async with asyncio.timeout(self.request_timeout):
                while True:
                    if pending is None:
                        pending = asyncio.create_task(job.events.get())
                    done, _ = await asyncio.wait((pending, watch), timeout=10, return_when=asyncio.FIRST_COMPLETED)
                    if watch in done:
                        return
                    if not done:
                        if streaming:
                            await send({"type": "http.response.body", "body": b": working\n\n", "more_body": True})
                        continue
                    kind, value = pending.result()
                    pending = None
                    if kind == "ready" and request["stream"]:
                        await send({"type": "http.response.start", "status": 200, "headers": [
                            (b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache"),
                            (b"x-accel-buffering", b"no")]})
                        streaming = True
                        if request["chat"]:
                            await sse(chunk({"role": "assistant", "content": ""}))
                    elif kind == "delta" and streaming:
                        await sse(chunk({"content": value} if request["chat"] else value))
                    elif kind in ("invalid", "error"):
                        if streaming:
                            await sse(value)
                            await sse("[DONE]")
                            await send({"type": "http.response.body", "body": b""})
                        else:
                            await self._json(send, 400 if kind == "invalid" else 500, value)
                        return
                    elif kind == "result":
                        if streaming:
                            await sse(chunk(reason=value["finish_reason"]))
                            if request["include_usage"]:
                                await sse(chunk(usage=value["usage"]))
                            await sse("[DONE]")
                            await send({"type": "http.response.body", "body": b""})
                        else:
                            choice = {"index": 0, "finish_reason": value["finish_reason"]}
                            choice.update({"message": value["message"]} if request["chat"]
                                          else {"text": value["text"], "logprobs": None})
                            await self._json(send, 200, {**common,
                                "object": "chat.completion" if request["chat"] else "text_completion",
                                "choices": [choice], "usage": value["usage"]})
                        return
        except TimeoutError:
            payload = error("inference deadline exceeded; cancellation requested", "engine_error", "request_timeout")
            if streaming:
                await sse(payload)
                await sse("[DONE]")
                await send({"type": "http.response.body", "body": b""})
            else:
                await self._json(send, 504, payload)
        except OSError:
            pass  # Connection reset during send; finally cancels the worker.
        finally:
            job.cancel.set()
            watch.cancel()
            if pending is not None:
                pending.cancel()
            await asyncio.gather(watch, *([pending] if pending else []), return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--cache-gib", type=float, default=8)
    parser.add_argument("--retain-gib", type=float, default=0)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--prefill-batch", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--audit-log", type=Path, help="optional local JSONL with prompt token IDs and generated text")
    args = parser.parse_args()
    if not (1 <= args.port <= 65535 and 0 < args.cache_gib <= 32
            and 0 <= args.retain_gib <= args.cache_gib / 2 and 1 <= args.context <= 512
            and 1 <= args.prefill_batch <= min(32, args.context)
            and 1 <= args.max_tokens <= min(256, args.context) and 1 <= args.request_timeout <= 7200):
        parser.error("invalid limits: cache <=32 GiB, retain <=half cache, context <=512, prefill <=32, tokens <=256")
    worker = Worker(lambda: ReferenceEngine(args.checkpoint, args.cache_gib, args.retain_gib,
                                            args.context, args.prefill_batch), args.audit_log)
    import uvicorn
    # Exactly one process owns upstream globals. Never expose or use --workers.
    uvicorn.run(LocalAPI(worker, args.max_tokens, request_timeout=args.request_timeout),
                host="127.0.0.1", port=args.port, workers=1, access_log=False,
                timeout_keep_alive=5, timeout_graceful_shutdown=35)


if __name__ == "__main__":
    main()
