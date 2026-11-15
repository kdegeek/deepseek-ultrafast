"""Bounded checkpoint fixtures plus an optional actual released expert matrix."""
import json
from itertools import product
from pathlib import Path
import sys
import struct
import tempfile
import unittest

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(ROOT))
import checkpoint
import model as official
from run import tiny_args


def parameter(shape, dtype=torch.float32):
    return torch.nn.Parameter(torch.empty(shape, device="meta", dtype=dtype), requires_grad=False)


class QuantLinear(torch.nn.Module):
    def __init__(self, rows=2, cols=32):
        super().__init__()
        self.weight = parameter((rows, cols // 2), torch.float4_e2m1fn_x2)
        self.scale = parameter((rows, cols // 32), torch.float8_e8m0fnu)

    def forward(self):
        assert self.weight.scale is self.scale
        return self.weight.view(torch.uint8).clone(), self.scale.view(torch.uint8).clone()


class ParallelEngramEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_embeddings, self.dim, self.block_size = 17, 64, 32
        self.vocab_start_idx, self.vocab_end_idx = 0, 17
        self.weight = parameter((17, 64), torch.float8_e4m3fn)
        self.scale = parameter((17, 2), torch.float8_e8m0fnu)

    def forward(self, indices):
        raise AssertionError("lazy row implementation must replace this forward")


class TinyTokenizer:
    backend_tokenizer = property(lambda self: self)

    def __len__(self):
        return 128

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"token{i}" for i in ids)

    def id_to_token(self, token_id):
        return f"token{token_id}"


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, tensors):
        save_file(tensors, self.directory / "model.safetensors")

    def test_index_missing_shard_and_traversal(self):
        path = self.directory / "model.safetensors.index.json"
        for filename, error in [("missing.safetensors", FileNotFoundError), ("../escape.safetensors", ValueError)]:
            path.write_text(json.dumps({"weight_map": {"weight": filename}}))
            with self.assertRaises(error):
                checkpoint.CheckpointStore(self.directory)

    def test_invalid_safetensors_rejected(self):
        (self.directory / "bad.safetensors").write_bytes(b"not safetensors")
        with self.assertRaises(Exception):
            checkpoint.CheckpointStore(self.directory)

    def test_lru_reload_matches_eager_with_one_matrix_budget(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False, device="meta"),
                                    torch.nn.Linear(4, 4, bias=False, device="meta"))
        a = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
        b = torch.eye(4) * 2
        self.save({"0.weight": a, "1.weight": b})
        manager = checkpoint.attach_hooks(model, self.directory, "cpu", 64)
        try:
            x = torch.tensor([[1., 2., 3., 4.]])
            for _ in range(2):
                torch.testing.assert_close(model(x), x @ a.T @ b.T)
            self.assertEqual(manager.stats()["parameter_loads"], 4)
            self.assertEqual(manager.peak_resident_bytes, 64)
            self.assertTrue(model[0].weight.is_meta)
            self.assertFalse(any(manager.pins.values()))
        finally:
            manager.close()
        self.assertTrue(model[1].weight.is_meta)

    def test_indexed_shards_load_under_device_context_with_eviction(self):
        weight_map = {}
        for index in range(40):
            name, filename = f"{index}.weight", f"part-{index:02d}.safetensors"
            weight_map[name] = filename
            value = torch.eye(4, dtype=torch.float32) * (2 if index == 39 else 1)
            save_file({name: value}, self.directory / filename)
        (self.directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() and torch.version.hip else [])
        for device in devices:
            with self.subTest(device=device):
                model = torch.nn.Sequential(*(torch.nn.Linear(4, 4, bias=False, dtype=torch.float32, device="meta")
                                              for _ in range(40)))
                manager = checkpoint.attach_hooks(model, self.directory, device, 64)
                try:
                    with torch.inference_mode(), torch.device(device):
                        x = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float32)
                        for _ in range(2):
                            torch.testing.assert_close(model(x), x * 2, rtol=0, atol=0)
                    self.assertEqual(manager.loads, 80)
                    self.assertEqual(manager.peak_resident_bytes, 64)
                    self.assertFalse(any(manager.pins.values()))
                finally:
                    manager.close()

    def test_budget_error_unpins_scope(self):
        model = QuantLinear()
        self.save({"weight": torch.zeros(2, 16, dtype=torch.int8),
                   "scale": torch.full((2, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)})
        manager = checkpoint.attach_hooks(model, self.directory, "cpu", 32)
        try:
            with self.assertRaises(MemoryError):
                model()
            self.assertFalse(any(manager.pins.values()))
            self.assertLessEqual(manager.resident_bytes, 32)
        finally:
            manager.close()

    def test_shape_mismatch_before_copy(self):
        self.save({"weight": torch.zeros(3, 4)})
        model = torch.nn.Linear(4, 4, bias=False, device="meta")
        manager = checkpoint.attach_hooks(model, self.directory, "cpu", 1024)
        try:
            with self.assertRaisesRegex(ValueError, "checkpoint shape"):
                model(torch.zeros(1, 4))
            self.assertEqual(manager.resident_bytes, 0)
            self.assertFalse(any(manager.pins.values()))
        finally:
            manager.close()

    def test_packed_fp4_scales_and_device_copy(self):
        raw = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
        scale = torch.tensor([[127], [128]], dtype=torch.uint8)
        self.save({"weight": raw.view(torch.int8), "scale": scale.view(torch.float8_e8m0fnu)})
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() and torch.version.hip else [])
        for device in devices:
            with self.subTest(device=device):
                model = QuantLinear()
                manager = checkpoint.attach_hooks(model, self.directory, device, 34)
                try:
                    actual, actual_scale = model()
                    self.assertTrue(torch.equal(actual.cpu(), raw))
                    self.assertTrue(torch.equal(actual_scale.cpu(), scale))
                    self.assertEqual(manager.resident_bytes, 34)
                    self.assertEqual(model.weight.dtype, torch.float4_e2m1fn_x2)
                finally:
                    manager.close()

    def test_direct_wo_a_access_dequantizes_raw_checkpoint(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wo_a = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16, device="meta")

            def forward(self, x):
                return x @ self.wo_a.weight.T

        model = torch.nn.Sequential(Attention())
        weights = (torch.arange(4096).reshape(64, 64) % 7 - 3).to(torch.float8_e4m3fn)
        scales = torch.tensor([[127, 128], [129, 126]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
        self.save({"0.wo_a.weight": weights, "0.wo_a.scale": scales})
        manager = checkpoint.attach_hooks(model, self.directory, "cpu", 8192)
        try:
            x = torch.ones(1, 64, dtype=torch.bfloat16)
            dense = (weights.float().reshape(2, 32, 2, 32) * scales.float()[:, None, :, None]).reshape(64, 64).bfloat16()
            torch.testing.assert_close(model(x), x @ dense.T, rtol=0, atol=0)
        finally:
            manager.close()

    def test_engram_fetches_unique_rows_only_and_masks(self):
        model = torch.nn.Sequential(ParallelEngramEmbedding())
        values = (torch.arange(17 * 64).reshape(17, 64) % 7).to(torch.float8_e4m3fn)
        scales = torch.full((17, 2), 128, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        self.save({"0.weight": values, "0.scale": scales})
        manager = checkpoint.attach_hooks(model, self.directory, "cpu", 1)
        try:
            manager.store.tensor = lambda name: self.fail("Engram must never request the entire tensor")
            ids = torch.tensor([[4, 4, 9, -1, 17]])
            actual = model(ids)
            expected = (values.float() * 2).bfloat16()[torch.tensor([[4, 4, 9, 0, 0]])]
            expected[:, 3:] = 0
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(manager.engram_rows, 3)
            self.assertEqual(manager.resident_bytes, 0)
            self.assertTrue(model[0].weight.is_meta)
        finally:
            manager.close()

    def test_actual_original_expert_sample(self):
        sample = ROOT.parents[1] / "target" / "strix-checkpoint-sample"
        if not (sample / "expert-sample.safetensors").exists():
            self.skipTest("downloaded original expert sample is unavailable")
        model = torch.nn.Module()
        owner = model
        for component in ("layers", "6", "ffn", "experts", "0"):
            child = torch.nn.Module()
            owner.add_module(component, child)
            owner = child
        expert = QuantLinear(2304, 5120)
        owner.add_module("w1", expert)
        manager = checkpoint.attach_hooks(model, sample, "cpu", 2304 * (2560 + 160))
        try:
            actual, scales = expert()
            self.assertEqual(actual.shape, (2304, 2560))
            self.assertEqual(scales.shape, (2304, 160))
            source = manager.store.tensor("layers.6.ffn.experts.0.w1.weight")
            self.assertTrue(torch.equal(actual, source.view(torch.uint8)))
        finally:
            manager.close()

    def test_actual_engram_high_offset_rows_under_gpu_default(self):
        directory = ROOT.parents[1] / "target" / "models" / "DeepSeek-V4.1-Flash"
        if not (directory / "download-verified.json").exists():
            self.skipTest("verified original checkpoint is unavailable")
        store = checkpoint.CheckpointStore(directory)
        device = "cuda" if torch.cuda.is_available() and torch.version.hip else "cpu"
        try:
            for name in ("layers.1.engram.embed.weight", "layers.1.engram.embed.scale"):
                path = directory / store.weight_map[name]
                with path.open("rb") as stream:
                    header_length = struct.unpack("<Q", stream.read(8))[0]
                    metadata = json.loads(stream.read(header_length))[name]
                    rows, columns = metadata["shape"]
                    ids = [0, rows // 2, rows - 1, rows // 2]
                    expected = []
                    for row in ids:
                        # Both original Engram weights and scales are one byte.
                        stream.seek(8 + header_length + metadata["data_offsets"][0] + row * columns)
                        expected.append(stream.read(columns))
                with torch.device(device):
                    actual = store.rows(name, ids)
                self.assertEqual(actual.device.type, "cpu")
                self.assertEqual(tuple(actual.shape), (len(ids), columns))
                self.assertEqual(actual.view(torch.uint8).numpy().tobytes(), b"".join(expected))
        finally:
            store.close()

    def test_full_tiny_architecture_matches_eager_prefill_and_decode(self):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(7)
        official.precompute_freqs_cis.cache_clear()
        official.get_window_topk_idxs.cache_clear()
        try:
            args = tiny_args()
            # Exercise the actual hash state and Engram path, not only a fake table.
            args.engram_layer_ids = (1,)
            args.engram_num_embeddings = (8,)
            args.engram_vocab_size = 3  # two disjoint prime buckets: 3 + 5
            args.engram_max_ngram_size = 2
            args.engram_n_heads = 2
            args.engram_head_dim = 32
            args.engram_compressed_vocab_size = 128
            tokenizer = TinyTokenizer()
            eager = official.Transformer(args, tokenizer).eval()
            with torch.no_grad():
                for name, value in eager.named_parameters():
                    if value.dtype == torch.float8_e8m0fnu:
                        value.view(torch.uint8).fill_(123)
                    elif value.dtype == torch.float4_e2m1fn_x2:
                        value.view(torch.uint8).copy_(torch.randint(0, 256, value.shape, dtype=torch.uint8))
                    elif name.endswith("norm.weight") or name == "norm.weight":
                        value.fill_(1)
                    elif name.endswith("bias") or name.endswith("attn_sink"):
                        value.zero_()
                    elif "hc_" in name:
                        value.copy_(torch.randn(value.shape, dtype=torch.float32) * 0.02)
                    else:
                        value.copy_(torch.randn(value.shape, dtype=torch.float32) * 0.05)
            state = {name: value.detach().contiguous().view(torch.int8) if value.dtype == torch.float4_e2m1fn_x2
                     else value.detach().contiguous() for name, value in eager.named_parameters()}
            self.save(state)
            tokens, decode = torch.tensor([[4, 9, 12, 7]]), torch.tensor([[5]])
            with torch.inference_mode():
                expected_prefill = eager(tokens, 0)[1].clone()
                expected_decode = eager(decode, 4)[1].clone()
            devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() and torch.version.hip else [])
            for device, incremental in product(devices, (False, True)):
                with self.subTest(device=device, incremental=incremental):
                    official.precompute_freqs_cis.cache_clear()
                    official.get_window_topk_idxs.cache_clear()
                    lazy, manager = checkpoint.build_lazy_model(args, tokenizer, self.directory, device, 128 * 1024)
                    try:
                        self.assertTrue(all(value.is_meta for value in lazy.parameters()))
                        self.assertFalse(any(value.is_meta for value in lazy.buffers()))
                        for name, value in lazy.named_buffers():
                            if name.endswith("score_state"):
                                self.assertTrue(torch.isneginf(value).all())
                        with torch.inference_mode(), torch.device(device):
                            if incremental:
                                # The bounded runner prefills one token at a time,
                                # including the first incomplete compression group.
                                for position in range(tokens.shape[1]):
                                    actual_prefill = lazy(tokens[:, position:position + 1].to(device), position)[1].cpu()
                            else:
                                actual_prefill = lazy(tokens.to(device), 0)[1].cpu()
                            actual_decode = lazy(decode.to(device), 4)[1].cpu()
                        tolerance = {"rtol": 0, "atol": 0} if device == "cpu" and not incremental else {"rtol": 0.03, "atol": 0.01}
                        torch.testing.assert_close(actual_prefill, expected_prefill, **tolerance)
                        torch.testing.assert_close(actual_decode, expected_decode, **tolerance)
                        self.assertTrue(torch.equal(actual_prefill.argmax(-1), expected_prefill.argmax(-1)))
                        self.assertTrue(torch.equal(actual_decode.argmax(-1), expected_decode.argmax(-1)))
                        self.assertLessEqual(manager.peak_resident_bytes, 128 * 1024)
                        self.assertGreater(manager.loads, len(manager.cache))
                        self.assertGreater(manager.engram_rows, 0)
                        self.assertFalse(any(manager.pins.values()))
                    finally:
                        manager.close()
        finally:
            torch.set_default_dtype(previous)
            official.precompute_freqs_cis.cache_clear()
            official.get_window_topk_idxs.cache_clear()


if __name__ == "__main__":
    torch.set_num_threads(8)
    unittest.main()
