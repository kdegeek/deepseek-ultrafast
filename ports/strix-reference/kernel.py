"""Pure PyTorch replacements for the pinned DeepSeek TileLang kernels.

This is a correctness backend, not a performance replacement for Blackwell.
Shell FP8/FP4 dtypes are transported as bytes; arithmetic uses FP32 on CPU or
ROCm. GEMMs dequantize bounded tiles and sparse attention uses online softmax.
Equations/API follow DeepSeek-V4.1-Flash revision
314113f8a7dadb388215396ba5fbee2c610a04d6, inference/kernel.py.
"""
from typing import Optional
import math
import torch

_FP8 = torch.float8_e4m3fn
_FP4 = torch.float4_e2m1fn_x2
_E8M0 = torch.float8_e8m0fnu
_FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP8_VALUES = [m / 512 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
               for e in range(16) for m in range(8)][:-1]


def _same_device(*tensors):
    if any(t.device != tensors[0].device for t in tensors[1:]):
        raise ValueError("all tensors must be on the same device")


def _require_finite(x, name):
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} must contain finite values")


def _table(values, x):
    return torch.tensor(values, dtype=torch.float32, device=x.device)


def _encode_small_float(x, values, sign_bit):
    """Saturating round-to-nearest-even; preserves negative zero."""
    table = _table(values, x)
    midpoint = (table[:-1] + table[1:]) * 0.5
    a = x.abs().clamp(max=values[-1]).contiguous()
    code = torch.bucketize(a, midpoint)
    tie = a == midpoint[code.clamp(max=midpoint.numel() - 1)]
    code = code + (tie & ((code & 1) != 0)).to(code.dtype)
    return (code.to(torch.uint8) | (torch.signbit(x).to(torch.uint8) * sign_bit))


def _fp8_encode(x):
    return _encode_small_float(x, _FP8_VALUES, 128).view(_FP8)


def _fp8_decode(x):
    if x.dtype != _FP8:
        raise ValueError("expected float8_e4m3fn tensor")
    code = x.contiguous().view(torch.uint8).long()
    table = _table(_FP8_VALUES + [float("nan")], x)
    return table[code & 127] * torch.where((code & 128) != 0, -1.0, 1.0)


def _fp4_decode(x):
    if x.dtype not in (_FP4, torch.int8, torch.uint8):
        raise ValueError("expected packed float4_e2m1fn_x2 bytes")
    raw = x.contiguous().view(torch.uint8).long()
    code = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    return _table(_FP4_VALUES, x)[code & 7] * torch.where((code & 8) != 0, -1.0, 1.0)


def _scale_decode(x):
    if x.dtype == _E8M0:
        code = x.contiguous().view(torch.uint8).int()
        values = torch.ldexp(torch.ones_like(code, dtype=torch.float32), code - 127)
        return torch.where(code == 255, float("nan"), values)
    if x.dtype == _FP8:
        return _fp8_decode(x)
    if x.dtype not in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
        raise ValueError("unsupported scale dtype")
    return x.float()


def _round_scale(amax, reciprocal):
    # Same FP32 bit-level ceil(log2) and power-of-two operation as TileLang.
    bits = (amax * reciprocal).contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0).int()
    return ((exponent + 127) << 23).contiguous().view(torch.float32)


def _scale_encode(scale, dtype):
    if dtype == torch.float32:
        return scale
    if dtype == _FP8:
        return _fp8_encode(scale)
    if dtype == _E8M0:
        # E8M0 has no mantissa. TileLang's conversion rounds positive scales
        # upward, which is exact for the power-of-two MXFP path.
        bits = scale.contiguous().view(torch.int32)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).int()
        return exponent.clamp(0, 254).to(torch.uint8).view(_E8M0)
    raise ValueError("unsupported quantization scale dtype")


def _quant_input(x, block_size):
    if x.ndim < 1 or block_size <= 0 or x.shape[-1] == 0 or x.shape[-1] % block_size:
        raise ValueError("last dimension must be a positive multiple of block_size")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("quantization requires BF16, F16 or F32 input")
    _require_finite(x, "input")
    return x.float().reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
              scale_dtype: torch.dtype = torch.float32, inplace: bool = False):
    z = _quant_input(x, block_size)
    if scale_dtype not in (torch.float32, _E8M0):
        raise ValueError("FP8 activation scales must be float32 or E8M0")
    amax = z.abs().amax(dim=-1).clamp_min(1e-4)
    scale = _round_scale(amax, 1.0 / 448.0) if scale_fmt is not None else amax * (1.0 / 448.0)
    quant = _fp8_encode((z / scale.unsqueeze(-1)).clamp(-448, 448))
    if inplace:
        x.copy_((_fp8_decode(quant) * scale.unsqueeze(-1)).reshape(x.shape).to(x.dtype))
        return x
    return quant.reshape(x.shape), _scale_encode(scale, scale_dtype)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False,
                  scale_dtype: torch.dtype = _E8M0):
    z = _quant_input(x, block_size)
    if block_size % 2 or scale_dtype not in (_E8M0, _FP8):
        raise ValueError("FP4 requires an even block size and E8M0 or E4M3 scales")
    amax = z.abs().amax(dim=-1)
    if scale_dtype == _FP8:
        scale = _fp8_decode(_fp8_encode(amax.clamp_min(6 * 2.0 ** -9) / 6.0))
    else:
        scale = _round_scale(amax.clamp_min(6 * 2.0 ** -126), 1.0 / 6.0)
    code = _encode_small_float((z / scale.unsqueeze(-1)).clamp(-6, 6), _FP4_VALUES, 8)
    code = code.reshape(x.shape)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous().view(_FP4)
    if inplace:
        values = _fp4_decode(packed).reshape(z.shape) * scale.unsqueeze(-1)
        x.copy_(values.reshape(x.shape).to(x.dtype))
        return x
    return packed, _scale_encode(scale, scale_dtype)


def _gemm_validate(a, a_s, b, b_s, group_size, fp4):
    _same_device(a, a_s, b, b_s)
    if any(not x.is_contiguous() for x in (a, a_s, b, b_s)):
        raise ValueError("GEMM tensors must be contiguous")
    if a.ndim < 1 or b.ndim != 2 or group_size not in (32, 128):
        raise ValueError("invalid GEMM dimensions or scale block size")
    k = a.shape[-1]
    if k == 0 or k % group_size or b.shape[-1] != k // (2 if fp4 else 1):
        raise ValueError("GEMM reduction dimensions do not match")
    m, n = a.numel() // k, b.shape[0]
    expected_b = (n, k // 32) if fp4 else ((n + group_size - 1) // group_size, k // group_size)
    if a_s.numel() != m * (k // group_size) or tuple(b_s.shape) != expected_b:
        raise ValueError("GEMM scale shapes do not match quantization blocks")
    return m, n, k


def _gemm(a, a_s, b, b_s, group_size, fp4):
    m, n, k = _gemm_validate(a, a_s, b, b_s, group_size, fp4)
    a = a.reshape(m, k)
    a_scale = _scale_decode(a_s).reshape(m, k // group_size)
    b_scale = _scale_decode(b_s)
    _require_finite(a_scale, "activation scales")
    _require_finite(b_scale, "weight scales")
    out = torch.empty((m, n), dtype=torch.get_default_dtype(), device=a.device)
    # A full-K slab avoids launching one tiny GEMM per32 coefficients. Each
    # decoded operand is at most16MiB (except an individual oversized row), and
    # only the requested rows of the checkpoint weight are converted at once.
    max_rows = max(1, (16 * 1024 * 1024) // (4 * k))
    input_rows, weight_rows = min(128, max_rows), min(256, max_rows)
    for column in range(0, n, weight_rows):
        stop = min(column + weight_rows, n)
        if fp4:
            bv = _fp4_decode(b[column:stop])
            bv = (bv.reshape(stop - column, k // 32, 32) * b_scale[column:stop, :, None]).reshape(stop - column, k)
        else:
            bv = _fp8_decode(b[column:stop])
            bs = b_scale[torch.arange(column, stop, device=b.device) // group_size]
            bv = (bv.reshape(stop - column, k // group_size, group_size) * bs[..., None]).reshape(stop - column, k)
        for row in range(0, m, input_rows):
            rows = min(input_rows, m - row)
            av = _fp8_decode(a[row:row + rows])
            av = (av.reshape(rows, k // group_size, group_size) * a_scale[row:row + rows, :, None]).reshape(rows, k)
            out[row:row + rows, column:stop] = (av @ bv.T).to(out.dtype)
    return out


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
             scale_dtype: torch.dtype = torch.float32, block_size: int = 128):
    shape = (*a.shape[:-1], b.shape[0])
    return _gemm(a, a_s, b, b_s, block_size, False).reshape(shape)


def fp4_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
             scale_dtype: torch.dtype = torch.float32, act_block_size: int = 128):
    shape = (*a.shape[:-1], b.shape[0])
    return _gemm(a, a_s, b, b_s, act_block_size, True).reshape(shape)


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                topk_idxs: torch.Tensor, softmax_scale: float):
    _same_device(q, kv, attn_sink, topk_idxs)
    if q.ndim != 4 or kv.ndim != 3 or topk_idxs.ndim != 3:
        raise ValueError("attention expects Q[B,S,H,D], KV[B,N,D], indices[B,S,K]")
    batch, seq, heads, dim = q.shape
    if kv.shape[0] != batch or kv.shape[2] != dim or topk_idxs.shape[:2] != (batch, seq) or attn_sink.shape != (heads,):
        raise ValueError("attention dimensions do not match")
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        raise ValueError("attention indices must be integers")
    if bool(((topk_idxs < -1) | (topk_idxs >= kv.shape[1])).any()):
        raise ValueError("attention indices must be -1 or valid KV positions")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("softmax scale must be positive and finite")
    output = torch.empty_like(q)
    if kv.shape[1] == 0 or topk_idxs.shape[-1] == 0:
        return output.zero_()
    for b in range(batch):
        for start in range(0, seq, 16):
            query = q[b, start:start + 16].float()
            count = query.shape[0]
            maximum = torch.full((count, heads), -1e30, dtype=torch.float32, device=q.device)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros((count, heads, dim), dtype=torch.float32, device=q.device)
            for offset in range(0, topk_idxs.shape[-1], 64):
                index = topk_idxs[b, start:start + 16, offset:offset + 64].long()
                valid = index >= 0
                values = kv[b, index.clamp_min(0)].float().masked_fill(~valid[..., None], 0)
                score = (query @ values.transpose(-1, -2)) * softmax_scale
                score.masked_fill_(~valid[:, None, :], -float("inf"))
                new_max = torch.maximum(maximum, score.amax(dim=-1))
                correction = torch.exp(maximum - new_max)
                probability = torch.exp(score - new_max[..., None])
                denominator = denominator * correction + probability.sum(dim=-1)
                # Match upstream's BF16 probability staging before the value GEMM.
                numerator = numerator * correction[..., None] + probability.to(torch.bfloat16).float() @ values
                maximum = new_max
            # Include the learned sink with zero value; stable even when all
            # indices are padding or sink logits greatly exceed the KV logits.
            final_max = torch.maximum(maximum, attn_sink.float()[None, :])
            correction = torch.exp(maximum - final_max)
            denominator = denominator * correction + torch.exp(attn_sink.float()[None, :] - final_max)
            numerator = numerator * correction[..., None]
            normalized = numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)[..., None]
            output[b, start:start + count] = normalized.to(q.dtype)
    return output


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                      hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    _same_device(mixes, hc_scale, hc_base)
    width = (2 + hc_mult) * hc_mult
    if hc_mult <= 0 or sinkhorn_iters < 1 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Sinkhorn dimensions, iterations and epsilon must be positive")
    if mixes.ndim != 3 or mixes.shape[-1] != width or hc_scale.shape != (3,) or hc_base.shape != (width,):
        raise ValueError("Sinkhorn tensor dimensions do not match")
    z, scale, base = mixes.float(), hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(z[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(z[..., hc_mult:2 * hc_mult] * scale[1] + base[hc_mult:2 * hc_mult])
    comb = (z[..., 2 * hc_mult:] * scale[2] + base[2 * hc_mult:]).reshape(*z.shape[:-1], hc_mult, hc_mult)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre.to(mixes.dtype), post.to(mixes.dtype), comb.to(mixes.dtype)
