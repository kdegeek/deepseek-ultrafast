"""Memory-bounded AMD correctness reference for the original DeepSeek V4.1 model."""
import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Local AMD kernels take precedence over the unmodified upstream model sources.
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(ROOT))

import torch
import model as model_module
from model import ModelArgs, Transformer


def tiny_args():
    return ModelArgs(max_batch_size=1, max_seq_len=32, vocab_size=128, dim=64,
                     moe_inter_dim=64, n_layers=5, n_mtp_layers=0, n_heads=2,
                     n_routed_experts=4, n_activated_experts=2, dtype="fp8",
                     expert_dtype="fp4", q_lora_rank=32, head_dim=32,
                     rope_head_dim=16, o_groups=2, o_lora_rank=32,
                     window_size=8, compress_ratios=(0, 2, 2, 1, 1), kv_source_layers=(1, 3),
                     index_source_layers=(1, 3, 4), index_n_heads=2, index_head_dim=32,
                     candidate_source_layer=3, candidate_topk_blocks=2, candidate_block_size=2,
                     index_topk=4, norm_eps=1e-6, hc_mult=4, temperature=0)


def smoke(device):
    """Whole reduced architecture prefill/decode, comparing AMD to CPU, not model quality."""
    torch.manual_seed(2304)
    torch.set_default_dtype(torch.bfloat16)
    cpu_model = Transformer(tiny_args()).eval()
    with torch.no_grad():
        for name, parameter in cpu_model.named_parameters():
            if parameter.dtype == torch.float8_e8m0fnu:
                parameter.view(torch.uint8).fill_(123)
            elif parameter.dtype == torch.float4_e2m1fn_x2:
                parameter.view(torch.uint8).copy_(torch.randint(0, 256, parameter.shape, dtype=torch.uint8))
            elif name.endswith("norm.weight") or name == "norm.weight":
                parameter.fill_(1.0)
            elif name.endswith("bias") or name.endswith("attn_sink"):
                parameter.zero_()
            elif "hc_" in name:
                parameter.copy_(torch.randn(parameter.shape, dtype=torch.float32) * 0.02)
            else:
                parameter.copy_(torch.randn(parameter.shape, dtype=torch.float32) * 0.05)
    state = {k: v.clone() for k, v in cpu_model.state_dict().items()}
    tokens = torch.tensor([[4, 9, 12, 7]], dtype=torch.long)
    with torch.inference_mode(), torch.device("cpu"):
        cpu_prefill = cpu_model(tokens, 0)[1].float().clone()
        cpu_decode = cpu_model(torch.tensor([[5]], dtype=torch.long), 4)[1].float().clone()
    # A fresh model resets global shared-attention caches between devices.
    model_module.precompute_freqs_cis.cache_clear()
    model_module.get_window_topk_idxs.cache_clear()
    with torch.device(device):
        gpu_model = Transformer(tiny_args()).eval()
        # Shell dtypes may not support copy kernels on AMD. Copy packed bytes,
        # preserving each Parameter object and its custom .scale attachment.
        with torch.no_grad():
            for name, parameter in gpu_model.named_parameters():
                parameter.view(torch.uint8).copy_(state[name].view(torch.uint8).to(device))
        with torch.inference_mode():
            gpu_prefill = gpu_model(tokens.to(device), 0)[1].float().cpu()
            gpu_decode = gpu_model(torch.tensor([[5]], dtype=torch.long), 4)[1].float().cpu()
    checks = []
    for name, expected, actual in [("prefill", cpu_prefill, gpu_prefill), ("decode", cpu_decode, gpu_decode)]:
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"{name}: non-finite logits")
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)
        if not torch.equal(actual.argmax(-1), expected.argmax(-1)):
            raise RuntimeError(f"{name}: CPU/GPU greedy token differs")
        checks.append({"stage": name, "max_abs_error": (actual-expected).abs().max().item(),
                       "top_token_cpu": expected.argmax(-1).item(), "top_token_gpu": actual.argmax(-1).item()})
    return {"fixture": "synthetic reduced architecture", "device": torch.cuda.get_device_name() if device == "cuda" else device,
            "checks": checks, "passed": True, "model_quality_verified": False}


def generate(args):
    from transformers import AutoTokenizer
    from checkpoint import build_lazy_model

    initialized = time.perf_counter()
    config = json.loads((args.checkpoint / "inference" / "config.json").read_text())
    config.update(max_batch_size=1, max_seq_len=args.context, temperature=0,
                  vision_n_layers=0, n_mtp_layers=0)
    # The reference executes target decoding only; DSpark remains unported.
    config["compress_ratios"] = config["compress_ratios"][:config["n_layers"]]
    fields = {f.name for f in dataclasses.fields(ModelArgs)}
    unknown = set(config) - fields
    if unknown:
        raise ValueError(f"unsupported model config fields: {sorted(unknown)}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    prompt = tokenizer.encode(args.prompt, add_special_tokens=True)
    if not prompt or len(prompt) + args.tokens > args.context:
        raise ValueError("prompt is empty or prompt plus completion exceeds context")
    model_args = ModelArgs(**config)
    print("Building lazy original-model reference", file=sys.stderr, flush=True)
    model, manager = build_lazy_model(model_args, tokenizer, args.checkpoint, args.device, int(args.cache_gib * 1024**3))
    generated = []
    steps = []
    stop_reason = "max_tokens"
    started = time.perf_counter()
    print(f"Model ready after {started - initialized:.2f}s; {len(prompt)} prompt tokens", file=sys.stderr, flush=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def forward(token, position, stage):
        step_started = time.perf_counter()
        next_ids, logits, _ = model(torch.tensor([[token]], dtype=torch.long), position)
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"non-finite {stage} logits at position {position}")
        # Reading the extrema synchronizes the GPU before recording elapsed time.
        minimum, maximum = logits.min().item(), logits.max().item()
        next_id = int(next_ids.item())
        seconds = time.perf_counter() - step_started
        steps.append({"stage": stage, "position": position, "input_token_id": token,
                      "next_token_id": next_id, "elapsed_seconds": seconds,
                      "logits_min": minimum, "logits_max": maximum, "logits_finite": True})
        print(f"{stage} position {position}: {seconds:.2f}s; finite logits [{minimum:.4f}, {maximum:.4f}]",
              file=sys.stderr, flush=True)
        return next_id

    try:
        with torch.inference_mode(), torch.device(args.device):
            # Token-by-token prefill bounds activation and selected-expert memory.
            for position, token in enumerate(prompt):
                next_id = forward(token, position, "prefill")
            for offset in range(args.tokens):
                if next_id == tokenizer.eos_token_id:
                    stop_reason = "eos"
                    break
                generated.append(next_id)
                print(f"decode {offset + 1}/{args.tokens}: {tokenizer.decode([next_id])!r}", file=sys.stderr, flush=True)
                if offset + 1 < args.tokens:
                    next_id = forward(next_id, len(prompt) + offset, "decode")
        generation_seconds = time.perf_counter() - started
        cache_stats = manager.stats()
        device_memory = ({"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                          "total_device_bytes": torch.cuda.get_device_properties(0).total_memory}
                         if args.device == "cuda" else None)
    finally:
        manager.close()
    provenance = {}
    for name in ("source-manifest.json", "download-verified.json", "inference/config.json", "tokenizer.json", "tokenizer_config.json"):
        path = args.checkpoint / name
        if path.is_file():
            provenance[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"checked_utc": datetime.now(timezone.utc).isoformat(),
            "device": args.device, "device_name": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
            "torch_version": torch.__version__, "hip_version": torch.version.hip,
            "platform": platform.platform(), "model": str(args.checkpoint), "prompt": args.prompt,
            "text": tokenizer.decode(generated), "tokens": generated,
            "prompt_tokens": prompt, "stop_reason": stop_reason,
            "initialization_seconds": started - initialized,
            "elapsed_seconds_including_prefill": generation_seconds,
            "mode": "target-only reference, raw completion", "weight_cache": cache_stats,
            "device_memory": device_memory, "forward_steps": steps,
            "all_logits_finite": all(step["logits_finite"] for step in steps),
            "model_layers": model_args.n_layers, "context_limit": args.context,
            "provenance_sha256": provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "generate"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="PyTorch uses cuda for AMD HIP devices too")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--cache-gib", type=float, default=8)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output", type=Path, help="also save the result JSON to this path")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.device == "cuda" and (not torch.cuda.is_available() or not torch.version.hip):
        parser.error("AMD HIP PyTorch is required; no CPU fallback is performed")
    if args.command == "generate" and args.checkpoint is None:
        parser.error("generate requires --checkpoint")
    if not 1 <= args.tokens <= args.context or not 1 <= args.context <= 4096 or not 0 < args.cache_gib <= 32:
        parser.error("require 1 <= tokens <= context <= 4096 and 0 < cache-gib <= 32")
    torch.set_default_dtype(torch.bfloat16)
    result = smoke(args.device) if args.command == "smoke" else generate(args)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
