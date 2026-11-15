"""Measure isolated, explicitly labelled precision perturbations of the HIP oracle.

The baseline uses unchanged pinned code. Variants are diagnostics, never acceptance
oracles. Only layer 0's attention HC projection is changed in the f64 variant.
"""
import argparse
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(ROOT))


def run(args):
    import torch
    from engine import ReferenceEngine, reset_request_state
    from component_oracle import metrics
    import model as official
    args.output.mkdir(parents=True, exist_ok=False)
    engine = ReferenceEngine(args.checkpoint, context=128, prefill_batch=1)
    layer = engine.model.layers[0]
    original = layer.hc_mixes
    evidence, baselines = {}, {}
    raw_double = None

    def hc_double(self, x, fn, scale, base):
        nonlocal raw_double
        if fn is not self.hc_attn_fn:
            return original(x, fn, scale, base)
        flat = x.flatten(2).float()
        raw = torch.nn.functional.linear(flat.cpu().double(), fn.cpu().double())
        raw_double = raw.flatten().tolist()
        # Only the reduction is perturbed; rounding to FP32 and all subsequent
        # HIP RMS, sigmoid, Sinkhorn, BF16 boundaries remain the pinned equations.
        raw = raw.float().to(x.device)
        inv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        return official.hc_split_sinkhorn(raw * inv, scale, base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    try:
        with torch.inference_mode(), torch.device("cuda"):
            for variant in ("baseline", "baseline_repeat", "hc0_attention_f64_projection"):
                reset_request_state(engine.model, official)
                if variant.endswith("f64_projection"):
                    layer.hc_mixes = types.MethodType(hc_double, layer)
                tokens = engine.tokenizer.encode(args.prompt, add_special_tokens=False)
                for position, token in enumerate(tokens):
                    ids, logits, _ = engine.model(torch.tensor([[token]], dtype=torch.long), position)
                    values = logits.float().flatten().cpu().tolist()
                    key = f"p{position:03}"
                    (args.output / f"{variant}-{key}-logits.json").write_text(json.dumps(values))
                    if variant == "baseline":
                        baselines[key] = values
                    evidence[f"{variant}-{key}"] = {"next_token": ids.item(), **metrics(values, baselines[key])}
                    print(variant, key, evidence[f"{variant}-{key}"], flush=True)
                layer.hc_mixes = original
        (args.output / "report.json").write_text(json.dumps({"identity":engine.identity, "prompt":args.prompt,
            "variants":evidence, "f64_hc0_projection":raw_double,
            "scope":"One HC projection reduction replaced for diagnosis only; baseline/oracle files unchanged."}, indent=2))
    finally:
        layer.hc_mixes = original
        engine.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt", default="The")
    run(p.parse_args())
