"""CPU and real AMD tests for the PyTorch kernel replacements.

Run: python -m unittest discover -s ports/strix-reference/tests -v
GPU tests require ROCm and are explicitly skipped if it is unavailable.
"""
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import unittest
import torch

spec = importlib.util.spec_from_file_location("strix_reference_kernel", Path(__file__).resolve().parents[1] / "kernel.py")
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)


def fp8_cpu(tensor):
    return tensor.cpu().view(torch.float8_e4m3fn).float()


def scale_cpu(tensor):
    if tensor.dtype == torch.float8_e8m0fnu:
        return torch.tensor([2.0 ** (v - 127) for v in tensor.cpu().view(torch.uint8).flatten().tolist()], dtype=torch.float32).reshape(tensor.shape)
    return tensor.cpu().float()


def fp4_cpu(tensor):
    table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    result = []
    for byte in tensor.cpu().view(torch.uint8).flatten().tolist():
        for code in (byte & 15, byte >> 4):
            result.append(table[code & 7] * (-1 if code & 8 else 1))
    return torch.tensor(result, dtype=torch.float32).reshape(*tensor.shape[:-1], tensor.shape[-1] * 2)


class KernelCases:
    device = "cpu"

    def test_fp8_all_codes_and_midpoint_rounding(self):
        positive = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        midpoint = (positive[:-1] + positive[1:]) * 0.5
        values = torch.cat([positive, -positive, midpoint, -midpoint, torch.tensor([-1000.0, 1000.0])])
        expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        actual = kernel._fp8_encode(values.to(self.device)).view(torch.uint8).cpu()
        self.assertTrue(torch.equal(actual, expected))
        decoded = kernel._fp8_decode(actual.to(self.device).view(torch.float8_e4m3fn)).cpu()
        torch.testing.assert_close(decoded, expected.view(torch.float8_e4m3fn).float(), rtol=0, atol=0)

    def test_act_quant_shapes_scales_and_inplace_alias(self):
        torch.manual_seed(7)
        original = torch.randn(2, 3, 64, dtype=torch.float32).to(torch.bfloat16)
        original[0, 0] = 0
        blocks = original.float().reshape(2, 3, 2, 32)
        for rounded in (False, True):
            amax = blocks.abs().amax(-1).clamp_min(1e-4)
            scale = amax * (1.0 / 448.0)
            if rounded:
                scale = torch.exp2(torch.ceil(torch.log2(scale)))
            expected = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
            dtype = torch.float8_e8m0fnu if rounded else torch.float32
            actual, scales = kernel.act_quant(original.to(self.device), 32, "ue8m0" if rounded else None, dtype)
            self.assertEqual(actual.shape, original.shape)
            self.assertEqual(scales.shape, (2, 3, 2))
            self.assertEqual(actual.dtype, torch.float8_e4m3fn)
            self.assertEqual(scales.dtype, dtype)
            self.assertTrue(torch.equal(actual.cpu().view(torch.uint8), expected.reshape(original.shape).view(torch.uint8)))
            torch.testing.assert_close(scale_cpu(scales), scale, rtol=0, atol=0)
            inplace = original.to(self.device).clone()
            pointer = inplace.data_ptr()
            returned = kernel.act_quant(inplace, 32, "ue8m0" if rounded else None, dtype, True)
            self.assertEqual(returned.data_ptr(), pointer)
            expected_dequant = (expected.float() * scale[..., None]).reshape(original.shape).to(original.dtype)
            torch.testing.assert_close(inplace.cpu(), expected_dequant, rtol=0, atol=0)

    def test_fp4_quantization_ties_shapes_and_inplace(self):
        x = torch.tensor([0.0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -0.25, -0.75, -1.25, -1.75, -2.5, -5.0] * 2)
        x = torch.stack([x, x * 4, torch.zeros_like(x)]).to(torch.bfloat16)
        magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        for dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
            quant, scales = kernel.fp4_act_quant(x.to(self.device), 32, scale_dtype=dtype)
            self.assertEqual(quant.shape, (3, 16))
            self.assertEqual(quant.dtype, torch.float4_e2m1fn_x2)
            self.assertEqual(scales.shape, (3, 1))
            self.assertEqual(scales.dtype, dtype)
            expected_bytes = []
            expected_scale = []
            for row in x.float().tolist():
                if dtype == torch.float8_e8m0fnu:
                    s = 2.0 ** math.ceil(math.log2(max(max(abs(v) for v in row), 6 * 2.0 ** -126) / 6))
                else:
                    s = torch.tensor(max(max(abs(v) for v in row), 6 * 2.0 ** -9) / 6).to(torch.float8_e4m3fn).float().item()
                expected_scale.append(s)
                codes = []
                for value in row:
                    a = min(abs(value / s), 6)
                    code = min(range(8), key=lambda i: (abs(a - magnitudes[i]), i % 2))
                    codes.append(code | (8 if math.copysign(1, value) < 0 else 0))
                expected_bytes.append([codes[i] | codes[i + 1] << 4 for i in range(0, 32, 2)])
            self.assertEqual(quant.cpu().view(torch.uint8).tolist(), expected_bytes)
            torch.testing.assert_close(scale_cpu(scales).flatten(), torch.tensor(expected_scale), rtol=0, atol=0)
            inplace = x.to(self.device).clone()
            returned = kernel.fp4_act_quant(inplace, 32, True, dtype)
            self.assertEqual(returned.data_ptr(), inplace.data_ptr())
            torch.testing.assert_close(inplace.cpu(), (fp4_cpu(quant) * scale_cpu(scales)).to(x.dtype), rtol=0, atol=0)

    def test_fp4_compressed_kv_block16(self):
        x = torch.zeros((2, 16), dtype=torch.bfloat16, device=self.device)
        q, s = kernel.fp4_act_quant(x, 16, scale_dtype=torch.float8_e4m3fn)
        self.assertEqual(q.shape, (2, 8))
        self.assertTrue(bool((scale_cpu(s) > 0).all()))
        self.assertTrue(bool((fp4_cpu(q) == 0).all()))

    def test_fp8_gemm_block_scaling_and_output_dtype(self):
        torch.manual_seed(8)
        for block in (32, 128):
            k, n = block * 2, block + 1
            a = torch.randn(2, 3, k).to(torch.float8_e4m3fn)
            b = torch.randn(n, k).to(torch.float8_e4m3fn)
            a_s = torch.rand(2, 3, 2) + 0.1
            b_s = torch.rand(2, 2) + 0.1
            av = a.float() * a_s.repeat_interleave(block, dim=-1)
            bv = b.float() * b_s.repeat_interleave(block, dim=0)[:n].repeat_interleave(block, dim=-1)
            expected = av.double() @ bv.double().T
            actual = kernel.fp8_gemm(a.to(self.device), a_s.to(self.device), b.to(self.device), b_s.to(self.device), block_size=block)
            self.assertEqual(actual.shape, (2, 3, n))
            self.assertEqual(actual.dtype, torch.get_default_dtype())
            torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-4, atol=2e-5)
            previous = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                result = kernel.fp8_gemm(a.to(self.device), a_s.to(self.device), b.to(self.device), b_s.to(self.device), block_size=block)
                self.assertEqual(result.dtype, torch.bfloat16)
                torch.testing.assert_close(result.cpu().float(), expected.float(), rtol=0.008, atol=0.008)
            finally:
                torch.set_default_dtype(previous)

    def test_fp4_gemm_with_e8m0_scales(self):
        torch.manual_seed(9)
        for group in (32, 128):
            k, n = 256, 259
            a = torch.randn(2, 2, k).to(torch.float8_e4m3fn)
            a_s = torch.randint(122, 130, (2, 2, k // group), dtype=torch.uint8).view(torch.float8_e8m0fnu)
            b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            b_s = torch.randint(121, 129, (n, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
            av = a.float() * scale_cpu(a_s).repeat_interleave(group, dim=-1)
            bv = fp4_cpu(b) * scale_cpu(b_s).repeat_interleave(32, dim=-1)
            expected = av.double() @ bv.double().T
            actual = kernel.fp4_gemm(a.to(self.device), a_s.to(self.device), b.to(self.device), b_s.to(self.device), torch.float8_e8m0fnu, group)
            self.assertEqual(actual.shape, (2, 2, n))
            torch.testing.assert_close(actual.cpu().double(), expected, rtol=3e-4, atol=8e-5)

    def test_sparse_attention_padding_duplicates_chunks_and_sink(self):
        torch.manual_seed(10)
        q = torch.randn(2, 17, 3, 8).to(torch.bfloat16)
        kv = torch.randn(2, 75, 8).to(torch.bfloat16)
        sink = torch.tensor([-1.0, 0.0, 3.0])
        indices = torch.randint(0, 75, (2, 17, 70), dtype=torch.int32)
        indices[0, 0] = -1
        indices[1, :, ::3] = -1
        scale = 8 ** -0.5
        expected = torch.zeros_like(q, dtype=torch.float64)
        for batch in range(2):
            for seq in range(17):
                selected = kv[batch, indices[batch, seq][indices[batch, seq] >= 0].long()].double()
                for head in range(3):
                    logits = selected @ q[batch, seq, head].double() * scale
                    probabilities = torch.softmax(torch.cat([logits, sink[head:head + 1].double()]), dim=0)[:-1]
                    expected[batch, seq, head] = probabilities @ selected
        actual = kernel.sparse_attn(q.to(self.device), kv.to(self.device), sink.to(self.device), indices.to(self.device), scale)
        self.assertEqual(actual.shape, q.shape)
        self.assertEqual(actual.dtype, q.dtype)
        # Upstream stages unnormalized probabilities through BF16 per64 block.
        torch.testing.assert_close(actual.cpu().double(), expected, rtol=0.02, atol=0.004)
        self.assertTrue(bool((actual[0, 0] == 0).all()))
        dominant = kernel.sparse_attn(q[:1, :1].to(self.device), kv[:1].to(self.device), torch.full((3,), 1000.0, device=self.device), indices[:1, 1:2].to(self.device), scale)
        self.assertTrue(bool((dominant == 0).all()))

    def test_sparse_attention_invalid_index_and_empty_selection(self):
        q = torch.zeros((1, 1, 2, 4), dtype=torch.bfloat16, device=self.device)
        kv = torch.zeros((1, 2, 4), dtype=torch.bfloat16, device=self.device)
        sink = torch.zeros(2, device=self.device)
        for value in (-2, 2):
            with self.assertRaises(ValueError):
                kernel.sparse_attn(q, kv, sink, torch.full((1, 1, 1), value, dtype=torch.int32, device=self.device), 0.5)
        out = kernel.sparse_attn(q, kv, sink, torch.empty((1, 1, 0), dtype=torch.int32, device=self.device), 0.5)
        self.assertTrue(bool((out == 0).all()))
        out = kernel.sparse_attn(q, kv, torch.full_like(sink, -1e35), torch.full((1, 1, 3), -1, dtype=torch.int32, device=self.device), 0.5)
        self.assertTrue(bool((out == 0).all()))

    def test_sinkhorn_matches_direct_equations(self):
        torch.manual_seed(11)
        mixes = torch.randn(2, 3, 24)
        scales = torch.tensor([0.3, 0.5, 0.8])
        base = torch.randn(24)
        for iterations in (1, 4, 20):
            pre, post, comb = kernel.hc_split_sinkhorn(mixes.to(self.device), scales.to(self.device), base.to(self.device), 4, iterations, 1e-6)
            z = mixes.double()
            expected_pre = 1 / (1 + torch.exp(-(z[..., :4] * scales[0] + base[:4]))) + 1e-6
            expected_post = 2 / (1 + torch.exp(-(z[..., 4:8] * scales[1] + base[4:8])))
            matrix = torch.exp((z[..., 8:] * scales[2] + base[8:]).reshape(2, 3, 4, 4))
            matrix = matrix / matrix.sum(-1, keepdim=True) + 1e-6
            matrix = matrix / (matrix.sum(-2, keepdim=True) + 1e-6)
            for _ in range(iterations - 1):
                matrix /= matrix.sum(-1, keepdim=True) + 1e-6
                matrix /= matrix.sum(-2, keepdim=True) + 1e-6
            for actual, expected in ((pre, expected_pre), (post, expected_post), (comb, matrix)):
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-5, atol=1e-6)
            self.assertEqual(comb.shape, (2, 3, 4, 4))

    def test_bad_shapes_and_nonfinite_quantization_fail(self):
        with self.assertRaises(ValueError):
            kernel.act_quant(torch.zeros(7, device=self.device), 32)
        with self.assertRaises(ValueError):
            kernel.act_quant(torch.full((32,), float("nan"), device=self.device), 32)
        with self.assertRaises(ValueError):
            kernel.fp4_act_quant(torch.zeros(32, device=self.device), 32, scale_dtype=torch.float32)
        with self.assertRaises(ValueError):
            kernel.hc_split_sinkhorn(torch.zeros(1, 1, 24, device=self.device), torch.ones(3, device=self.device), torch.zeros(24, device=self.device), sinkhorn_iters=0)


class CPU(KernelCases, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "requires an actual AMD ROCm GPU")
class AMD(KernelCases, unittest.TestCase):
    device = "cuda"

    def test_real_published_expert_matrix(self):
        sample_dir = Path(__file__).resolve().parents[3] / "target" / "strix-checkpoint-sample"
        sample_path = sample_dir / "expert-sample.safetensors"
        if not sample_path.exists():
            self.skipTest("pinned real checkpoint sample has not been downloaded")
        raw = sample_path.read_bytes()
        manifest = json.loads((sample_dir / "manifest.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(manifest["revision"], "314113f8a7dadb388215396ba5fbee2c610a04d6")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["sample_sha256"])
        header_size, = struct.unpack("<Q", raw[:8])
        header = json.loads(raw[8:8 + header_size])

        def tensor(name):
            metadata = header[name]
            start, end = metadata["data_offsets"]
            return torch.frombuffer(bytearray(raw[8 + header_size + start:8 + header_size + end]), dtype=torch.uint8).reshape(metadata["shape"])

        packed = tensor("layers.6.ffn.experts.0.w1.weight")
        scales = tensor("layers.6.ffn.experts.0.w1.scale")
        self.assertEqual(packed.shape, (2304, 2560))
        self.assertEqual(scales.shape, (2304, 160))
        torch.manual_seed(91)
        x = torch.randn(1, 5120).to(torch.bfloat16)
        a, a_s = kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        # Independent direct CPU dequantization followed by a full FP64 dot.
        codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
        magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        weights = magnitudes[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
        weights *= torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)
        activation = a.float() * scale_cpu(a_s).repeat_interleave(32, dim=-1)
        expected = activation.double() @ weights.double().T
        gpu_inputs = (a.cuda(), a_s.cuda(), packed.cuda().view(torch.float4_e2m1fn_x2), scales.cuda().view(torch.float8_e8m0fnu))
        torch.cuda.synchronize()
        start = time.perf_counter()
        actual = kernel.fp4_gemm(*gpu_inputs, torch.float8_e8m0fnu, 32)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        error = (actual.cpu().double() - expected).abs().max().item()
        torch.testing.assert_close(actual.cpu().double(), expected, rtol=3e-4, atol=1e-4)
        print(f"\nReal 2304x5120 FP4 expert on {torch.cuda.get_device_name()}: {seconds:.4f}s, max_abs_error={error:.8g}")


if __name__ == "__main__":
    unittest.main()
