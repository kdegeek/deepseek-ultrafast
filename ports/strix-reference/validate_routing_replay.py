"""Capture unchanged HIP routes and compare a conditional native tie replay.

No model source, output, routing policy, or tolerance is changed. Forward hooks
observe chosen routes; replay resamples native ATen top-k only at exact ties.
This proves conditional numerical fidelity, not deterministic production output.
"""
import argparse
import json
from pathlib import Path
import time

from validate_aten import read, sha
from component_oracle import float32_metrics


def bit_metrics(actual, expected):
    import numpy as np
    result = float32_metrics(actual, expected)
    a, b = (np.asarray(v, dtype=np.float32) for v in (actual, expected))
    result["different_bits"] = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
    return result


def validate_coverage(tokens, routes):
    if not 1 <= len(tokens) <= 128:
        raise ValueError("empty or oversized token case")
    coverage = [(r["position"], r["layer"]) for r in routes]
    expected = {(p, layer) for p in range(len(tokens)) for layer in range(40)}
    if (len(coverage) != len(expected) or set(coverage) != expected
            or not all(r["logits_scores_and_weights_exact"] is True for r in routes)):
        raise ValueError("incomplete or duplicated exact routing evidence")


def capture(args):
    import torch
    from engine import ReferenceEngine
    args.output.mkdir(parents=True, exist_ok=False)
    case = next(c for c in read(args.cases) if c["label"] == args.label)
    engine = ReferenceEngine(args.checkpoint, context=128, prefill_batch=1)
    handles = []
    position = 0
    values = lambda t: t.detach().float().flatten().cpu().tolist()
    try:
        if engine.tokenizer.encode(case["prompt"], add_special_tokens=False) != case["prompt_tokens"]:
            raise ValueError("tokenization differs")
        def hook(layer):
            def record(module, inputs, output):
                weights, indices = output
                logits = torch.nn.functional.linear(inputs[0].float(), module.weight.float()) / module.gate_temp
                scores = torch.nn.functional.softplus(logits).sqrt()
                corrected = scores + module.bias
                gathered = scores.gather(1, indices)
                gathered = gathered / (gathered.sum(-1, keepdim=True) + 1e-20) * module.route_scale
                if not torch.equal(gathered, weights):
                    raise ValueError("observer did not reproduce actual gate weights")
                path = args.output / f"p{position:03}-l{layer:02}-route.json"
                path.write_text(json.dumps({"logits": values(logits), "scores": values(scores),
                    "corrected": values(corrected), "indices": indices.flatten().cpu().tolist(),
                    "weights": values(weights)}), encoding="utf-8")
            return record
        for i, layer in enumerate(engine.model.layers):
            handles.append(layer.ffn.gate.register_forward_hook(hook(i)))
        started, next_tokens = time.perf_counter(), []
        with torch.inference_mode(), torch.device("cuda"):
            for position, token in enumerate(case["tokens"]):
                ids, logits, _ = engine.model(torch.tensor([[token]], dtype=torch.long), position)
                if not torch.isfinite(logits).all().item():
                    raise ValueError("nonfinite reference output")
                (args.output / f"p{position:03}-logits.json").write_text(json.dumps(values(logits)), encoding="utf-8")
                next_tokens.append(int(ids.item()))
                print(f"reference position {position}: {ids.item()} ({time.perf_counter()-started:.1f}s)", flush=True)
        report = {"identity": engine.identity, "case": case, "next_tokens": next_tokens,
                  "seconds": time.perf_counter() - started, "cases_sha256": sha(args.cases),
                  "source_sha256": {str(p): sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                  "official_source_sha256": {str(p): sha(p) for p in sorted((Path(__file__).parent / "upstream").glob("*.py"))}}
        (args.output / "reference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        for handle in handles:
            handle.remove()
        engine.close()


def compare(args):
    reference = read(args.reference / "reference.json")
    native = read(args.native / "summary.json")
    if reference["case"] != native["case"]:
        raise ValueError("input cases differ")
    positions = []
    for i in range(len(reference["case"]["tokens"])):
        name = f"p{i:03}-logits.json"
        actual, expected = read(args.native / name), read(args.reference / name)
        if len(actual) != 129280 or len(expected) != 129280:
            raise ValueError("incorrect vocabulary length")
        positions.append({"position": i, **bit_metrics(actual, expected),
                          "reference_sha256": sha(args.reference / name), "native_sha256": sha(args.native / name)})
    routes = native["routes"]
    validate_coverage(reference["case"]["tokens"], routes)
    report = {"method": "conditional legal-topk branch replay using repeated unchanged native ATen topk, not a production tie policy",
              "deterministic_unconditional_parity": False,
              "all_logits_exact": all(p["different_bits"] == 0 for p in positions),
              "positions": positions, "routes_checked": len(routes),
              "resampled_routes": [r for r in routes if r["attempts"] > 1],
              "reference": str(args.reference), "native": str(args.native),
              "reference_identity": reference["identity"], "native_identity": native["runtime"],
              "next_tokens_reference": reference["next_tokens"], "next_tokens_native": native["next_tokens"],
              "executable_sha256": sha(args.executable), "shim_sha256": sha(args.shim),
              "recording_sha256": {p.name: sha(p) for p in sorted(args.reference.glob("*.json"))},
              "native_summary_sha256": sha(args.native / "summary.json"),
              "source_sha256": {str(p): sha(p) for p in [Path(__file__), Path("crates/strix/src/routing_diagnostic.rs"),
                     Path("crates/strix/src/decoder.rs"), Path("crates/strix/examples/aten_route_replay.rs")]}}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("all_logits_exact", "routes_checked", "resampled_routes")}))
    if not report["all_logits_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    for name in ("checkpoint", "cases", "output"):
        cap.add_argument("--" + name, type=Path, required=True)
    cap.add_argument("--label", default="chat")
    comp = sub.add_parser("compare")
    for name in ("reference", "native", "executable", "shim", "output"):
        comp.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args)
    else:
        compare(args)
