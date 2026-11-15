"""Independent CPU fixtures and optional original HIP traces for the Rust port.

This is validation tooling only. The native executable never invokes it.
Pinned upstream files and the existing reference engine remain unmodified.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(ROOT))


def fixtures(output):
    import torch
    import kernel
    from model import Block
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(2)
    torch.manual_seed(739)
    def values(t):
        return t.float().flatten().tolist()
    result = {"source": "pinned official equations + independent PyTorch CPU reference kernels",
              "revision": "314113f8a7dadb388215396ba5fbee2c610a04d6", "quantization": []}
    x = (torch.randn(96) * 3).bfloat16()
    for kind in ("fp8", "fp4_e8m0", "fp4_e4m3"):
        y = x.clone()
        if kind == "fp8":
            kernel.act_quant(y, 32, "ue8m0", torch.float8_e8m0fnu, True)
        else:
            kernel.fp4_act_quant(y, 16 if kind == "fp4_e4m3" else 32, True,
                                 torch.float8_e4m3fn if kind == "fp4_e4m3" else torch.float8_e8m0fnu)
        result["quantization"].append({"kind": kind, "input": values(x), "output": values(y)})
    z, scale, base = torch.randn(1, 1, 24).float(), torch.randn(3).float(), torch.randn(24).float()
    pre, post, comb = kernel.hc_split_sinkhorn(z, scale, base)
    result["sinkhorn"] = {"z": values(z), "scale": values(scale), "base": values(base),
                          "pre": values(pre), "post": values(post), "comb": values(comb)}
    h, sub = torch.randn(1, 1, 4, 32).bfloat16(), torch.randn(1, 1, 32).bfloat16()
    result["hc"] = {"residual": values(h), "input": values(sub),
                    "collapsed": values(Block.hc_pre(None, h, pre)),
                    "output": values(Block.hc_post(None, sub, h, post, comb))}
    query, kv, sink = torch.randn(1, 1, 2, 32), torch.randn(1, 131, 32), torch.randn(2).float()
    indices = torch.tensor([[[-1] * 125 + list(range(6))]], dtype=torch.int32)
    out = kernel.sparse_attn(query, kv, sink, indices, 32 ** -0.5)
    result["attention"] = {"query": values(query), "kv": values(kv), "sinks": values(sink),
                           "indices": indices.flatten().tolist(), "output": values(out)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


def trace(checkpoint, output, prompt, count):
    import torch
    from engine import ReferenceEngine
    engine = ReferenceEngine(checkpoint, context=128, prefill_batch=1)
    output.mkdir(parents=True, exist_ok=True)
    position = 0
    def save(label, tensor):
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = tensor.float()
        (output / f"p{position:03}-{label}.json").write_text(json.dumps(tensor.flatten().cpu().tolist()))
    handles = []
    for i, layer in enumerate(engine.model.layers):
        for name in ("attn", "ffn"):
            module = getattr(layer, name)
            label = f"l{i:02}-{name}"
            handles.append(module.register_forward_pre_hook(lambda m, a, label=label: save(label + "-input", a[0])))
            handles.append(module.register_forward_hook(lambda m, a, o, label=label: save(label, o)))
        handles.append(layer.register_forward_hook(lambda m, a, o, i=i: save(f"l{i:02}-output", o[0])))
        if layer.engram is not None:
            handles.append(layer.engram.register_forward_pre_hook(lambda m, a, i=i: save(f"l{i:02}-engram-hashes", a[1])))
            handles.append(layer.engram.register_forward_hook(lambda m, a, o, i=i: save(f"l{i:02}-engram", o)))
    handles.append(engine.model.head.register_forward_pre_hook(lambda m, a: save("head-input", a[0])))
    tokens = engine.tokenizer.encode(prompt, add_special_tokens=False)
    summary = {"identity": engine.identity, "prompt_tokens": tokens, "next_tokens": []}
    try:
        with torch.inference_mode(), torch.device("cuda"):
            for position, token in enumerate(tokens[:count]):
                ids, logits, _ = engine.model(torch.tensor([[token]], dtype=torch.long), position)
                save("logits", logits)
                summary["next_tokens"].append(int(ids.item()))
                print(f"oracle position {position}: next {ids.item()}", flush=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        for h in handles:
            h.remove()
        engine.close()


def reduced(output):
    """A complete five-layer CPU oracle, with Engram and both KV ratios."""
    import dataclasses
    import torch
    from safetensors.torch import save_file
    from run import tiny_args
    from model import Transformer
    sys.path.insert(0, str(ROOT / "tests"))
    from test_checkpoint import TinyTokenizer
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(2)
    torch.manual_seed(2304)
    args = tiny_args()
    args.engram_layer_ids = (1,)
    args.engram_num_embeddings = (8,)
    args.engram_vocab_size = 3
    args.engram_max_ngram_size = 2
    args.engram_n_heads = 2
    args.engram_head_dim = 32
    args.engram_compressed_vocab_size = 128
    model = Transformer(args, TinyTokenizer()).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dtype == torch.float8_e8m0fnu:
                p.view(torch.uint8).fill_(123)
            elif p.dtype == torch.float4_e2m1fn_x2:
                p.view(torch.uint8).copy_(torch.randint(0, 256, p.shape, dtype=torch.uint8))
            elif name.endswith("norm.weight") or name == "norm.weight":
                p.fill_(1)
            elif name.endswith("bias") or name.endswith("attn_sink"):
                p.zero_()
            elif "hc_" in name:
                p.copy_(torch.randn(p.shape, dtype=torch.float32) * .02)
            else:
                p.copy_((torch.randn(p.shape, dtype=torch.float32) * .05).bfloat16())
    output.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, p in model.state_dict().items():
        p = p.contiguous()
        if p.dtype == torch.float4_e2m1fn_x2:
            p = p.view(torch.int8)
        if name == "head.weight":
            p = p.bfloat16()
        tensors[name] = p
    save_file(tensors, output / "fixture.safetensors")
    (output / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "fixture.safetensors" for n in tensors}}))
    (output / "config.json").write_text(json.dumps(dataclasses.asdict(args)))
    result = {"tokens": [4, 9, 12, 7], "logits": [], "greedy": [],
              "source": "independent original CPU Transformer; sequential prefill; five layers; synthetic weights",
              "rtol": .03, "atol": .01,
              "tolerance_source": "existing ports/strix-reference/run.py reduced architecture smoke gate"}
    result["hashes"] = {name: getattr(model.engram_hash, name).flatten().tolist()
                        for name in ("token_map", "primes", "offsets", "multipliers")}
    result["hashes"]["pad_id"] = model.engram_hash.pad_id
    with torch.inference_mode():
        for position, token in enumerate(result["tokens"]):
            ids, logits, _ = model(torch.tensor([[token]], dtype=torch.long), position)
            result["logits"].append(logits.float().flatten().tolist())
            result["greedy"].append(ids.item())
    (output / "expected.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--reduced", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--positions", type=int, default=2)
    args = parser.parse_args()
    if args.fixtures:
        fixtures(args.fixtures)
    elif args.reduced:
        reduced(args.reduced)
    elif args.checkpoint and args.trace:
        trace(args.checkpoint, args.trace, args.prompt, args.positions)
    else:
        parser.error("use --fixtures or --checkpoint with --trace")
