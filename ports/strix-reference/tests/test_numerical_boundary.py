"""Independent exact regression for the native attention scaling correction."""
import unittest
from pathlib import Path
import sys
import torch
import json
import tempfile
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel import sparse_attn


class NumericalBoundary(unittest.TestCase):
    def test_float32_json_roundtrip_is_not_numerical_error(self):
        from component_oracle import float32_metrics, metrics
        expanded = torch.tensor([.1], dtype=torch.float32).tolist()
        self.assertEqual(float32_metrics([.1], expanded)["different"], 0)
        # The independent float64 calibration must retain extra precision.
        self.assertEqual(metrics([.1], expanded)["different"], 1)

    def test_attention_scalar_multiply_boundary(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                query = torch.zeros(1, 1, 1, 512, dtype=torch.bfloat16, device=device)
                kv = torch.zeros(1, 1, 512, dtype=torch.bfloat16, device=device)
                query[..., 0], kv[..., 0] = 12., 1.
                sink = torch.tensor([0.5225175023078918], dtype=torch.float32, device=device)
                indices = torch.tensor([[[0]]], dtype=torch.int32, device=device)
                out = sparse_attn(query, kv, sink, indices, 512 ** -0.5)
                self.assertEqual(out[..., 0].item(), .5)
                self.assertTrue(bool((out[..., 1:] == 0).all()))

    def test_all_logits_report_reads_native_utf8_and_rejects_truncation(self):
        from validate_numerics import compare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native, hip, output = root / "native", root / "hip", root / "report"
            (native / "chat").mkdir(parents=True)
            (hip / "chat").mkdir(parents=True)
            (root / "inference").mkdir()
            output.mkdir()
            (root / "inference/config.json").write_text('{"vocab_size":3}', encoding="utf-8")
            binary = root / "fixture-executable"
            binary.write_bytes(b"not executed: compare existing artifacts only")
            case = {"label":"chat", "prompt":"<｜User｜>OK", "tokens":[2], "prompt_tokens":[2], "next_tokens":[1]}
            summary = {"prompt":case["prompt"], "input_tokens":[2], "prompt_tokens":[2], "next_tokens":[1],
                       "weight_cache":{}, "peak_read_bytes":0}
            (native / "chat/summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
            for directory in (native, hip):
                (directory / "chat/p000-logits.json").write_text('[0,2,1]', encoding="utf-8")
            args = SimpleNamespace(output=output, native=binary, native_output=native, checkpoint=root)
            compare(args, [case], {"fixture":True}, hip)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["cases"]["chat"]["prompt"], case["prompt"])
            self.assertEqual(report["cases"]["chat"]["positions"][0]["count"], 3)
            (native / "chat/p000-logits.json").write_text('[0,2]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete original vocabulary"):
                compare(args, [case], {"fixture":True}, hip)
