#!/usr/bin/env python3
"""Offline pinned-download regressions; never accesses the network or real models."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("download_qwen", Path(__file__).with_name("download-qwen.py"))
DOWNLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD)


class Response(io.BytesIO):
    def __init__(self, data: bytes, status: int, headers: dict):
        super().__init__(data)
        self.status = status
        self.headers = headers


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(DOWNLOAD, "ROOT", self.root).start()
        self.locks = self.root / "ports/qwen-resident"
        self.locks.mkdir(parents=True)
        self.models = self.root / "target/models/Qwen3.8-27B"
        self.models.mkdir(parents=True)
        self.data = {"q4": b"Q4 fixture contents", "q8": b"Q8 different fixture contents"}
        for quant, filename in DOWNLOAD.LOCKS.items():
            lock = {
                "filename": f"model-{quant}.gguf", "size_bytes": len(self.data[quant]),
                "sha256": hashlib.sha256(self.data[quant]).hexdigest(),
                "model_id": "test", "repository": "fixture/model", "revision": "a" * 40,
            }
            (self.locks / filename).write_text(json.dumps(lock), encoding="utf-8")

    def invoke(self, quant, response=None, extra=()):
        with patch("sys.argv", ["download-qwen.py", "--quant", quant, "--attempts", "1", *extra]), \
             patch.object(DOWNLOAD.urllib.request, "urlopen", return_value=response) as request, \
             contextlib.redirect_stdout(io.StringIO()):
            DOWNLOAD.main()
        return request

    def test_two_quants_have_independent_files_and_receipts(self):
        first = self.data["q4"]
        self.invoke("q4", Response(first, 200, {"Content-Length": str(len(first))}))
        receipt_path = self.models / "model-q4.gguf.verified.json"
        before = receipt_path.read_bytes()
        second = self.data["q8"]
        self.invoke("q8", Response(second, 200, {"Content-Length": str(len(second))}))
        self.assertEqual((self.models / "model-q4.gguf").read_bytes(), first)
        self.assertEqual((self.models / "model-q8.gguf").read_bytes(), second)
        self.assertEqual(receipt_path.read_bytes(), before)
        self.assertEqual(len(list(self.models.iterdir())), 4)

    def test_resume_hashes_complete_file_and_renames_verified_partial(self):
        data = self.data["q8"]
        partial = self.models / "model-q8.gguf.part"
        partial.write_bytes(data[:5])
        request = self.invoke("q8", Response(data[5:], 206, {
            "Content-Range": f"bytes 5-{len(data)-1}/{len(data)}", "Content-Length": str(len(data)-5),
        }))
        self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=5-")
        self.assertEqual((self.models / "model-q8.gguf").read_bytes(), data)
        self.assertFalse(partial.exists())
        receipt = json.loads((self.models / "model-q8.gguf.verified.json").read_text())
        self.assertEqual(receipt["lock_sha256"], DOWNLOAD.sha256(self.locks / "model-q8.lock.json"))

    def test_ignored_resume_is_rejected_without_overwrite(self):
        data = self.data["q8"]
        partial = self.models / "model-q8.gguf.part"
        partial.write_bytes(data[:5])
        with self.assertRaisesRegex(ValueError, "did not honor resume"):
            self.invoke("q8", Response(data, 200, {"Content-Length": str(len(data))}))
        self.assertEqual(partial.read_bytes(), data[:5])

    def test_bad_hash_is_retained_without_receipt(self):
        data = b"X" * len(self.data["q8"])
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.invoke("q8", Response(data, 200, {"Content-Length": str(len(data))}))
        self.assertEqual((self.models / "model-q8.gguf.part").read_bytes(), data)
        self.assertFalse((self.models / "model-q8.gguf").exists())
        self.assertFalse((self.models / "model-q8.gguf.verified.json").exists())

    def test_existing_unverified_file_is_not_destroyed(self):
        path = self.models / "model-q8.gguf"
        path.write_bytes(b"do not overwrite")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            self.invoke("q8")
        self.assertEqual(path.read_bytes(), b"do not overwrite")

    def test_explicit_lock_and_path_traversal_rejection(self):
        path = self.locks / "model-q8.lock.json"
        selected, lock = DOWNLOAD.selected_lock(path=path)
        self.assertEqual(selected, path)
        self.assertEqual(lock["filename"], "model-q8.gguf")
        lock["filename"] = "../outside.gguf"
        path.write_text(json.dumps(lock), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "single GGUF filename"):
            DOWNLOAD.selected_lock(path=path)

    def test_default_selects_q8_without_changing_explicit_q4(self):
        self.assertEqual(DOWNLOAD.selected_lock()[1]["filename"], "model-q8.gguf")
        self.assertEqual(DOWNLOAD.selected_lock(quant="q4")[1]["filename"], "model-q4.gguf")


if __name__ == "__main__":
    unittest.main()
