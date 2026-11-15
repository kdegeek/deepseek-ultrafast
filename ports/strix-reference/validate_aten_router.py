"""Diagnose ATen top-k ties on identical original router inputs, without fixes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from component_oracle import float32_metrics


def run(args):
    import torch
    args.output.mkdir(parents=True, exist_ok=False)
    requests = json.loads((args.components / "requests.json").read_text())
    expected = json.loads((args.components / "expected.json").read_text())
    request = next(r for r in requests if r["label"] == args.label)
    if request["op"] != "router":
        raise ValueError("expected an original-input router request")
    logits, bias = expected[args.label]["logits"], request["bias"]
    n = request["topk"]
    native_requests = [{"label": f"repeat-{i:03}", "op": "router_stages", "input": logits,
                        "bias": bias, "topk": n, "scale": request["scale"]} for i in range(100)]
    (args.output / "requests.json").write_text(json.dumps(native_requests))
    # Independent HIP oracle first; no model, math, sort order or precision override.
    with torch.inference_mode():
        x = torch.tensor([logits], dtype=torch.float32, device="cuda")
        b = torch.tensor(bias, dtype=torch.float32, device="cuda").reshape(1, -1)
        scores = torch.nn.functional.softplus(x).sqrt()
        corrected = scores + b
        repeats = [corrected.topk(n, dim=-1)[1].flatten().cpu().tolist() for _ in range(100)]
        score_values, corrected_values = scores.flatten().cpu().tolist(), corrected.flatten().cpu().tolist()
        cutoff = corrected.topk(n, dim=-1)[0][0, -1].item()
        ties = (corrected == cutoff).nonzero()[:, 1].cpu().tolist()
        tensor_hash = hashlib.sha256(corrected.cpu().numpy().tobytes()).hexdigest()
    subprocess.run([str(args.probe), str(args.checkpoint), str(args.output / "requests.json"),
                    str(args.output / "native")], check=True)
    native = json.loads((args.output / "native/actual.json").read_text())
    comparisons = [{"scores": float32_metrics(v["scores"], score_values),
                    "corrected": float32_metrics(v["corrected"], corrected_values)} for v in native.values()]
    counts = lambda rows: [{"indices": list(key), "count": count} for key, count in Counter(map(tuple, rows)).items()]
    report = {"original_label": args.label, "original_components": str(args.components),
              "requests_sha256": hashlib.sha256((args.components / "requests.json").read_bytes()).hexdigest(),
              "corrected_tensor_sha256": tensor_hash, "shape": [1, len(logits)], "dtype": "FP32",
              "cutoff": cutoff, "cutoff_tied_experts": ties,
              "hip_topk_repeats": counts(repeats),
              "native_topk_repeats": counts([list(map(int, v["indices"])) for v in native.values()]),
              "all_native_scores_bit_exact": all(m[s]["different"] == 0 for m in comparisons for s in m),
              "comparisons": comparisons,
              "interpretation": "Observed tied selections are reported, never forced to match a saved oracle; no tolerance or deterministic tie policy introduced.",
              "primary_documentation": "https://docs.pytorch.org/docs/2.11/generated/torch.topk.html"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("components", "checkpoint", "probe", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--label", required=True)
    run(p.parse_args())
