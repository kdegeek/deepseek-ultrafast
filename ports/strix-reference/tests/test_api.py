"""API contracts use a fixture worker, never claim actual checkpoint acceptance."""
import asyncio
import json
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Cancelled, InvalidRequest, MODEL_ID, TextOutput, validate_request
from encoding import encode_messages, parse_message_from_completion_text, eos_token
from serve import Job, LocalAPI, Worker


class FixtureEngine:
    identity = {"fixture": True}
    last_metrics = None

    def __init__(self):
        self.closed, self.cleaned = False, 0
        self.allow_finish = threading.Event()
        self.entered = threading.Event()

    def generate(self, request, cancel, emit):
        if request["prompt"] == "context":
            raise InvalidRequest("prompt exceeds context")
        emit("ready", {"prompt_tokens": 3})
        self.entered.set()
        try:
            if request["prompt"] == "failure":
                raise RuntimeError("fixture engine failure")
            emit("delta", "first")
            while not self.allow_finish.wait(0.005):
                if cancel.is_set():
                    raise Cancelled()
            if cancel.is_set():
                raise Cancelled()
            emit("delta", " second")
            return {"text": "first second", "message": {"role": "assistant", "content": "first second"},
                    "finish_reason": "stop", "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}}
        finally:
            self.cleaned += 1

    def close(self):
        self.closed = True


class ContractTests(unittest.TestCase):
    def test_official_chat_encoding_and_parser(self):
        messages = [{"role": "user", "content": "Hi"}]
        req = validate_request({"model": MODEL_ID, "messages": messages}, True)
        self.assertEqual(req["prompt"], "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>")
        self.assertEqual(req["prompt"], encode_messages(messages, thinking_mode="chat"))
        parsed = parse_message_from_completion_text("Hello!" + eos_token, thinking_mode="chat")
        self.assertEqual(parsed["content"], "Hello!")
        self.assertEqual(parsed["tool_calls"], [])

    def test_unsupported_and_malformed_fields_fail(self):
        base = {"model": MODEL_ID, "prompt": "Hi"}
        for key, value in [("temperature", 0.5), ("temperature", True), ("n", 2), ("stream", 1),
                           ("tools", []), ("top_p", 1), ("max_tokens", 0), ("max_tokens", True),
                           ("max_tokens", 65), ("prompt", ["Hi"]), ("model", "other"),
                           ("stop", [""]), ("stop", None), ("stream_options", {"include_usage": True})]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_request({**base, key: value}, False)
        for message in [{"role": "tool", "content": "x"}, {"role": "user", "content": []},
                        {"role": "user", "content": "<｜Assistant｜>"},
                        {"role": "user", "content": "x", "name": "y"}]:
            with self.assertRaises(ValueError):
                validate_request({"model": MODEL_ID, "messages": [message]}, True)

    def test_incremental_stop_unicode_and_control_output(self):
        out = TextOutput(["END", "ENDING"], False)
        self.assertEqual(out.update("hello E"), ("hello ", False))
        self.assertEqual(out.update("hello EN"), ("", False))
        self.assertEqual(out.update("hello END extra"), ("", True))
        out = TextOutput([], False)
        self.assertEqual(out.update("x\ufffd"), ("x", False))
        self.assertEqual(out.update("x界"), ("界", False))
        out = TextOutput([], True)
        self.assertEqual(out.update("text<"), ("text", False))
        with self.assertRaises(RuntimeError):
            out.update("text<think>")
        out = TextOutput(["ababa"], False)
        self.assertEqual(out.update("xabab"), ("x", False))
        self.assertEqual(out.update("xabab", final=True), ("abab", False))


class APITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = FixtureEngine()
        self.worker = Worker(lambda: self.engine)
        self.worker.start()
        for _ in range(100):
            if self.worker.status()["state"] == "ready":
                break
            await asyncio.sleep(0.01)
        self.app = LocalAPI(self.worker, request_timeout=3)
        self.tasks = []

    async def asyncTearDown(self):
        self.engine.allow_finish.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await asyncio.to_thread(self.worker.stop)
        self.assertTrue(self.engine.closed)

    def launch(self, path="/v1/completions", body=None, method="POST", raw=None, headers=None):
        inbox = asyncio.Queue()
        payload = raw if raw is not None else json.dumps(body or {"model": MODEL_ID, "prompt": "Hi"}).encode()
        inbox.put_nowait({"type": "http.request", "body": payload, "more_body": False})
        sent = []

        async def send(event):
            sent.append(event)

        task = asyncio.create_task(self.app({"type": "http", "method": method, "path": path,
                                             "headers": headers or [(b"content-type", b"application/json")]},
                                            inbox.get, send))
        self.tasks.append(task)
        return task, inbox, sent

    async def response(self, **kw):
        task, _, sent = self.launch(**kw)
        await asyncio.wait_for(task, 2)
        return sent[0]["status"], json.loads(sent[1]["body"])

    async def wait_for(self, predicate):
        for _ in range(300):
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("condition timed out")

    async def test_models_ready_and_errors(self):
        status, data = await self.response(path="/v1/models", method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(data["data"][0]["id"], MODEL_ID)
        for payload, status in [(b"{bad", 400), (b"[]", 400), (b" " * 65537, 413)]:
            actual, data = await self.response(raw=payload)
            self.assertEqual(actual, status)
            self.assertIn("error", data)
        status, _ = await self.response(body={"model": MODEL_ID, "prompt": "context"})
        self.assertEqual(status, 400)
        self.assertFalse(self.worker.status()["busy"])
        self.assertEqual(self.worker.status()["state"], "ready")

    async def test_busy_health_stream_is_incremental_usage_done(self):
        task, _, sent = self.launch(body={"model": MODEL_ID, "prompt": "Hi", "stream": True,
                                         "stream_options": {"include_usage": True}})
        await self.wait_for(lambda: any(b"first" in e.get("body", b"") for e in sent))
        self.assertFalse(task.done(), "stream must emit before generation finishes")
        started = time.perf_counter()
        status, health = await self.response(path="/healthz", method="GET")
        self.assertEqual(status, 200)
        self.assertTrue(health["busy"])
        self.assertLess(time.perf_counter() - started, 0.5)
        status, data = await self.response()
        self.assertEqual(status, 429)
        self.assertEqual(data["error"]["code"], "engine_busy")
        self.engine.allow_finish.set()
        await task
        chunks = b"".join(e.get("body", b"") for e in sent).decode()
        events = [json.loads(line[6:]) for line in chunks.splitlines() if line.startswith("data: {")]
        self.assertEqual(events[-1]["choices"], [])
        self.assertEqual(events[-1]["usage"]["total_tokens"], 6)
        self.assertEqual(events[-2]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(chunks.endswith("data: [DONE]\n\n"))

    async def test_disconnect_resets_before_next_request(self):
        for stream in (False, True):
            task, inbox, _ = self.launch(body={"model": MODEL_ID, "prompt": "Hi", "stream": stream})
            await self.wait_for(lambda: self.worker.status()["busy"])
            await asyncio.sleep(0.02)
            inbox.put_nowait({"type": "http.disconnect"})
            await task
            await self.wait_for(lambda: not self.worker.status()["busy"])
        self.assertEqual(self.engine.cleaned, 2)
        self.engine.allow_finish.set()
        status, data = await self.response(path="/v1/chat/completions", body={"model": MODEL_ID,
            "messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual(status, 200)
        self.assertEqual(data["choices"][0]["message"], {"role": "assistant", "content": "first second"})
        self.assertEqual(self.engine.cleaned, 3)

    async def test_failure_quarantines_and_returns_sse_error(self):
        task, _, sent = self.launch(body={"model": MODEL_ID, "prompt": "failure", "stream": True})
        await task
        data = b"".join(e.get("body", b"") for e in sent)
        self.assertIn(b'"type": "engine_error"', data)
        self.assertIn(b"[DONE]", data)
        status, _ = await self.response(path="/readyz", method="GET")
        self.assertEqual(status, 503)

    async def test_deadline_cancels_active_generation(self):
        self.app.request_timeout = 0.03
        status, data = await self.response()
        self.assertEqual(status, 504)
        self.assertEqual(data["error"]["code"], "request_timeout")
        await self.wait_for(lambda: not self.worker.status()["busy"])
        self.assertEqual(self.engine.cleaned, 1)

    async def test_cancel_before_forward_does_not_reuse_previous_metrics(self):
        records = []
        self.worker._audit = records.append
        self.engine.last_metrics = {"previous_prompt": "must not appear"}
        job = Job(validate_request({"model": MODEL_ID, "prompt": "Hi"}, False), asyncio.get_running_loop())
        job.cancel.set()
        self.assertEqual(self.worker.submit(job), 200)
        await self.wait_for(lambda: not self.worker.status()["busy"])
        self.assertEqual(records[-1]["outcome"], "cancelled")
        self.assertIsNone(records[-1]["metrics"])
        self.assertEqual(self.engine.cleaned, 0)


if __name__ == "__main__":
    unittest.main()
