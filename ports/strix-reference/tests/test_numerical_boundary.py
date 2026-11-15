"""Independent exact regression for the native attention scaling correction."""
import unittest
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel import sparse_attn


class NumericalBoundary(unittest.TestCase):
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
