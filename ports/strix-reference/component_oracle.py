"""Identical-input HIP/native component calibration; never changes the oracle.

Uses the original checkpoint and pinned modules. Evidence directories must be new.
The Rust example is validation-only and executes every learned matrix on Vulkan.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import struct

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(ROOT))


def metrics(actual, expected):
    import numpy as np
    a, b = np.asarray(actual, dtype=np.float64).reshape(-1), np.asarray(expected, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("comparison requires finite vectors of identical shape")
    return {"count": a.size, "different": int(np.count_nonzero(a != b)),
            "max_abs": float(np.max(np.abs(a - b), initial=0)),
            "relative_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))}


def bounded_rows(store, name, ids):
    """Validation-only bounded byte reads, bypassing Windows torch mmap storage."""
    import math
    import torch
    name = store.resolve(name)
    path = store.directory / store.weight_map[name]
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        if header_size > 16 * 1024**2:
            raise ValueError("oversized safetensors header")
        meta = json.loads(f.read(header_size))[name]
        shape = meta["shape"]
        begin, end = meta["data_offsets"]
        dtype, width = {"BF16":(torch.bfloat16,2), "F32":(torch.float32,4), "I8":(torch.int8,1),
                        "F8_E4M3":(torch.float8_e4m3fn,1), "F8_E8M0":(torch.float8_e8m0fnu,1)}[meta["dtype"]]
        if len(shape) != 2:
            raise ValueError("expected row matrix")
        stride = shape[1]*width
        if (not ids or len(ids)*stride > 1024**2 or end-begin != math.prod(shape)*width or
                any(i<0 or i>=shape[0] for i in ids)):
            raise ValueError("invalid bounded row read")
        data = bytearray()
        for row in ids:
            f.seek(8+header_size+begin+row*stride)
            raw=f.read(stride)
            if len(raw)!=stride:
                raise EOFError("checkpoint row truncated")
            data.extend(raw)
    with torch.device("cpu"):
        return torch.frombuffer(data,dtype=torch.uint8).clone().view(dtype).reshape(len(ids),shape[1])


def run(args):
    import torch
    import model as official
    from checkpoint import CheckpointStore
    from engine import ReferenceEngine
    args.output.mkdir(parents=True, exist_ok=False)
    engine = ReferenceEngine(args.checkpoint, context=128, prefill_batch=1)
    store = CheckpointStore(args.checkpoint)
    requests, expected, doubles = [], {}, {}

    if args.bounded_rows:
        # Diagnostic-only recovery for the Windows mmap/storage access violation.
        # Same checkpoint bytes and dtypes; no math or reference files changed.
        engine.manager.store.rows = lambda name, ids: bounded_rows(engine.manager.store, name, ids)

    def values(t):
        return (t.detach() if t.dtype == torch.float64 else t.detach().float()).flatten().cpu().tolist()

    def read(label):
        return torch.tensor(json.loads((args.trace / f"p000-{label}.json").read_text()),
                            dtype=torch.bfloat16, device="cuda")

    def weight(name):
        return store.tensor(name).cuda()

    def add(label, op, result, **kwargs):
        requests.append({"label": label, "op": op, **kwargs})
        expected[label] = values(result) if isinstance(result, torch.Tensor) else result

    handles = []
    stage = ""
    serial = 0
    # Post-hooks run after the loader has released weights; only inputs/results
    # are copied, never an entire Engram table or decoded model.
    for name, module in engine.model.named_modules():
        if isinstance(module, official.Gate):
            def gate_hook(m, a, out, name=name):
                weights, indices = out
                logits = torch.nn.functional.linear(a[0].float(), m.weight.float())
                add(f"{stage}/{name}", "router", {"logits":values(logits), "weights":values(weights), "indices":indices.flatten().tolist()},
                    name=name+".weight", input=values(a[0]), bias=values(m.bias), topk=m.topk, scale=m.route_scale)
            handles.append(module.register_forward_hook(gate_hook))
        elif isinstance(module, official.Linear):
            def linear_hook(m, a, out, name=name):
                label = f"{stage}/{name}"
                add(label, "linear", out, name=name + ".weight", input=values(a[0]),
                    round=out.dtype == torch.bfloat16)
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, official.RMSNorm):
            def norm_hook(m, a, out, name=name):
                x = a[0].float()
                add(f"{stage}/{name}", "rms",
                    {"inv": float(torch.rsqrt(x.square().mean(-1) + m.eps).item()), "output": values(out)},
                    input=values(a[0]), gamma=values(weight(name + ".weight")), eps=m.eps)
            handles.append(module.register_forward_hook(norm_hook))
        elif isinstance(module, official.Engram):
            def engram_hook(m, a, out, name=name):
                add(f"{stage}/gate", "engram", out, input=values(a[0]),
                    kv=expected[f"{stage}/{name}.wkv"], q=values(m.q_weight), k=values(m.k_weight), dim=m.dim, eps=m.eps)
            handles.append(module.register_forward_hook(engram_hook))

    original_sparse = official.sparse_attn
    def sparse(q, kv, sink, indices, scale):
        result = original_sparse(q, kv, sink, indices, scale)
        selected = [values(kv[0, i]) if i >= 0 else [] for i in indices.flatten().tolist()]
        add(f"{stage}/sparse", "attention", result, query=values(q), kv=selected,
            sinks=values(sink), dim=q.shape[-1])
        return result
    official.sparse_attn = sparse
    original_rope = official.apply_rotary_emb
    original_act = official.act_quant
    original_fp4 = official.fp4_act_quant
    original_einsum = torch.einsum
    def label(kind):
        nonlocal serial
        serial += 1
        return f"{stage}/{kind}-{serial}"

    def rotary(x, freqs, inverse=False):
        before = values(x)
        result = original_rope(x, freqs, inverse)
        add(label("rope"), "rope", x, input=before, cos=values(freqs.real), sin=values(freqs.imag), inverse=inverse)
        return result

    def act(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        from kernel import _fp8_decode, _scale_decode
        before = values(x)
        result = original_act(x, block_size, scale_fmt, scale_dtype, inplace)
        out = x if inplace else (_fp8_decode(result[0]).reshape(-1, block_size) * _scale_decode(result[1]).reshape(-1, 1))
        add(label("quant-fp8"), "quant", out, input=before, bits=8, group=block_size, e4m3=False)
        return result

    def fp4(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        assert inplace, "component capture only expects in-place index/KV quantization"
        before = values(x)
        result = original_fp4(x, block_size, inplace, scale_dtype)
        add(label("quant-fp4"), "quant", x, input=before, bits=4, group=block_size, e4m3=scale_dtype == torch.float8_e4m3fn)
        return result

    def einsum(equation, *operands):
        result = original_einsum(equation, *operands)
        if equation == "bsgd,grd->bsgr":
            x, w = operands
            for group in range(w.shape[0]):
                add(label("wo-a"), "range", result[0,0,group], name=f"layers.{layer}.attn.wo_a.weight",
                    input=values(x[0,0,group]), first=group*w.shape[1], rows=w.shape[1], round=True)
        return result
    official.apply_rotary_emb, official.act_quant, official.fp4_act_quant = rotary, act, fp4
    torch.einsum = einsum
    try:
        with torch.inference_mode(), torch.device("cuda"):
            hash_ids = engine.model.engram_hash(torch.tensor([[args.token]], dtype=torch.long), 0)
            for layer in args.layers:
                block = engine.model.layers[layer]
                dim = block.ffn.dim
                residual = (bounded_rows(store, "embed.weight", [args.token]).cuda().repeat(4, 1) if layer == 0 else
                            read(f"l{layer:02}-engram" if block.engram is not None else f"l{layer-1:02}-output").reshape(4, dim))
                if block.engram is not None:
                    stage = f"l{layer:02}-engram-p0"
                    # Historical traces stored ids as FP32, losing low bits above
                    # 2**24. Recompute from the pinned integer hash, never replay
                    # rounded trace ids as checkpoint row addresses.
                    hashes = hash_ids[:, :, block.engram.layer_hash_index, :]
                    block.engram(read(f"l{layer-1:02}-output").reshape(1,1,4,dim), hashes)
                for sub in ("attn", "ffn"):
                    x = residual if sub == "attn" else read(f"l{layer:02}-post-attn").reshape(4, dim)
                    prefix = f"layers.{layer}.hc_{sub}"
                    w, scale, base = (weight(prefix + suffix) for suffix in ("_fn", "_scale", "_base"))
                    flat = x.float().flatten().reshape(1, 1, -1)
                    raw = torch.nn.functional.linear(flat, w)
                    inv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + block.norm_eps)
                    pre, post, comb = official.hc_split_sinkhorn(raw * inv, scale, base)
                    add(prefix, "mix", {"raw": values(raw), "inv": inv.item(), "z": values(raw*inv),
                                        "pre":values(pre), "post":values(post), "comb":values(comb)},
                        prefix=prefix, input=values(x))
                    doubles[prefix] = values(torch.nn.functional.linear(flat.cpu().double(), w.cpu().double()))
                    subout = read(f"l{layer:02}-{sub}").reshape(1, 1, dim)
                    result = official.Block.hc_post(None, subout, x.reshape(1,1,4,dim), post, comb)
                    add(prefix + "/post", "post", result, input=values(subout), residual=values(x), post=values(post), comb=values(comb))
                    premix = ([1.,0.,0.,0.] if layer == 0 else json.loads((args.trace / f"p000-layers.{layer-1}.hc_ffn-pre.json").read_text())) if sub=="attn" else json.loads((args.trace / f"p000-layers.{layer}.hc_attn-pre.json").read_text())
                    collapsed = official.Block.hc_pre(None, x.reshape(1,1,4,dim), torch.tensor(premix, dtype=torch.float32).reshape(1,1,4))
                    add(prefix + "/pre", "pre", collapsed, input=values(x), pre=premix)
                    stage = f"l{layer:02}-{sub}-collapse"
                    getattr(block, sub + "_norm")(collapsed)
                for sub in ("attn", "ffn"):
                    # Keep component inputs identical, regardless of earlier native error.
                    module = getattr(block, sub)
                    x = read(f"l{layer:02}-{sub}-input").reshape(1, 1, dim)
                    for pos in ([0, 1] if sub == "attn" and layer in (2, 20) else [0]):
                        stage = f"l{layer:02}-{sub}-p{pos}"
                        result = module(x.clone(), pos) if sub == "attn" else module(x.clone())
                        expected[stage + "/component"] = values(result)
                print(f"collected layer {layer}", flush=True)
    finally:
        official.sparse_attn = original_sparse
        official.apply_rotary_emb, official.act_quant, official.fp4_act_quant = original_rope, original_act, original_fp4
        torch.einsum = original_einsum
        for h in handles:
            h.remove()
        engine.close()
        store.close()
    (args.output / "requests.json").write_text(json.dumps(requests))
    (args.output / "expected.json").write_text(json.dumps(expected))
    (args.output / "double.json").write_text(json.dumps(doubles))
    (args.output / "identity.json").write_text(json.dumps({"identity":engine.identity,
        "layers":args.layers, "trace":str(args.trace), "input_token":args.token,
        "row_reader":"bounded read/seek diagnostic recovery" if args.bounded_rows else "unchanged reference loader"}, indent=2))
    subprocess.run([str(args.probe), str(args.checkpoint), str(args.output / "requests.json"),
                    str(args.output / "actual.json")], check=True)
    actual = json.loads((args.output / "actual.json").read_text())
    report = {}
    for label, value in actual.items():
        target = expected[label]
        if isinstance(value, dict):
            report[label] = {key: metrics(val, target[key]) for key, val in value.items()}
        else:
            report[label] = metrics(value, target)
        if label in doubles:
            report[label]["native_vs_f64"] = metrics(value["raw"], doubles[label])
            report[label]["hip_vs_f64"] = metrics(target["raw"], doubles[label])
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--probe", type=Path, default=Path("target/release/examples/numerical_probe.exe"))
    p.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 20])
    p.add_argument("--bounded-rows", action="store_true")
    p.add_argument("--token", type=int, default=671, help="token at position zero of the supplied trace (default: The)")
    run(p.parse_args())
