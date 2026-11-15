"""Separate-process validation of the optional Rust/C++ ATen path.

The native executable never imports or starts Python. This tool checks unchanged
golden inputs, provenance and all FP32 logits; it creates new evidence only.
No numeric tolerance or alternate acceptance oracle is introduced.
"""
import argparse
import hashlib
import json
from pathlib import Path

from component_oracle import float32_metrics


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_components(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.reference / "requests.json") != sha(args.native / "requests.json"):
        raise ValueError("component input bytes differ from the original requests")
    actual, expected = read(args.native / "actual.json"), read(args.reference / "expected.json")
    if not actual or not actual.keys() <= expected.keys():
        raise ValueError("component labels do not match the original oracle")
    def compare_value(a, b):
        if isinstance(b, dict):
            if not isinstance(a, dict) or a.keys() != b.keys():
                raise ValueError("component fields differ")
            return {k: compare_value(a[k], b[k]) for k in b}
        return float32_metrics(a, b)
    def exact(value):
        if "different" in value:
            return value["different"] == 0
        return all(exact(v) for v in value.values())
    comparisons = {k: compare_value(actual[k], expected[k]) for k in actual}
    report = {"component_count": len(comparisons), "all_exact": all(exact(v) for v in comparisons.values()),
              "unexecuted_labels": sorted(expected.keys() - actual.keys()), "comparisons": comparisons,
              "requests_sha256": sha(args.reference / "requests.json"),
              "expected_sha256": sha(args.reference / "expected.json"),
              "actual_sha256": sha(args.native / "actual.json"),
              "executable_sha256": sha(args.executable), "shim_sha256": sha(args.shim),
              "native_identity": read(args.native / "identity.json")}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("component_count", "all_exact")}))


def compare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = read(args.reference / "cases.json")
    if args.labels:
        if not set(args.labels) <= {case["label"] for case in cases}:
            raise ValueError("unknown requested comparison label")
        cases = [case for case in cases if case["label"] in args.labels]
    identity = read(args.identity)
    identity = identity.get("identity", identity)
    for name, expected in identity["provenance_sha256"].items():
        if sha(args.checkpoint / name) != expected:
            raise ValueError(f"changed checkpoint provenance: {name}")
    if sha(args.checkpoint / "tokenizer.json") != identity["tokenizer_sha256"]:
        raise ValueError("changed tokenizer")
    vocab = read(args.checkpoint / "inference/config.json")["vocab_size"]
    report = {"identity": identity, "cases": {}, "all_exact": True,
              "native_directory": str(args.native), "reference_directory": str(args.reference),
              "inputs_sha256": sha(args.reference / "cases.json"),
              "selected_labels": [case["label"] for case in cases],
              "executable_sha256": sha(args.executable), "shim_sha256": sha(args.shim),
              "method": "exact FP32 recovery from JSON; errors computed in FP64; no tolerance",
              "source_sha256": {str(p): sha(p) for p in sorted(Path("ports/strix-reference").glob("*.py"))},
              "official_source_sha256": {str(p): sha(p) for p in sorted(Path("ports/strix-reference/upstream").glob("*.py"))},
              "golden_logits_sha256": {}, "native_logits_sha256": {}}
    for case in cases:
        label = case["label"]
        native = read(args.native / label / "summary.json")
        if native["input_tokens"] != case["tokens"] or native["prompt_tokens"] != case["prompt_tokens"]:
            raise ValueError(f"input/tokenization mismatch: {label}")
        positions = []
        for position, token in enumerate(case["tokens"]):
            name = f"{label}/p{position:03}-logits.json"
            expected_path, actual_path = args.reference / "hip" / name, args.native / name
            expected, actual = read(expected_path), read(actual_path)
            if len(actual) != vocab or len(expected) != vocab:
                raise ValueError("full original vocabulary required")
            expected_greedy = max(range(vocab), key=lambda i: (expected[i], -i))
            if expected_greedy != case["next_tokens"][position]:
                raise ValueError("reference cases disagree with saved golden logits")
            stats = float32_metrics(actual, expected)
            greedy = max(range(vocab), key=lambda i: (actual[i], -i))
            if greedy != native["next_tokens"][position]:
                raise ValueError("native summary disagrees with its saved logits")
            match = greedy == case["next_tokens"][position]
            report["all_exact"] &= stats["different"] == 0 and match
            positions.append({"position": position, "input_token": token, "hip_next": case["next_tokens"][position],
                              "native_next": greedy, "greedy_match": match, **stats})
            report["golden_logits_sha256"][name] = sha(expected_path)
            report["native_logits_sha256"][name] = sha(actual_path)
        report["cases"][label] = {"positions": positions, "native_summary": native}
    report["position_count"] = sum(len(c["positions"]) for c in report["cases"].values())
    report["greedy_matches"] = sum(p["greedy_match"] for c in report["cases"].values() for p in c["positions"])
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("all_exact", "position_count", "greedy_matches")}))


def long_reference(args):
    """Fresh, unchanged original reference beyond the previous ten positions."""
    import sys
    import torch
    from engine import ReferenceEngine
    sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))
    args.output.mkdir(parents=True, exist_ok=False)
    engine = ReferenceEngine(args.checkpoint, context=128, prefill_batch=1)
    prompt = ("The capital of France is Paris. The capital of Germany is Berlin. "
              "The capital of Italy is Rome. The capital of Spain is Madrid. "
              "The capital of Portugal is Lisbon. These cities are in Europe.")
    tokens = engine.tokenizer.encode(prompt, add_special_tokens=False)[:34]
    prompt = engine.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if engine.tokenizer.encode(prompt, add_special_tokens=False) != tokens or len(tokens) != 34:
        raise ValueError("long prompt did not roundtrip exactly")
    case = {"label": "geography34", "prompt": prompt, "tokens": tokens,
            "prompt_tokens": tokens, "next_tokens": []}
    directory = args.output / "hip" / case["label"]
    directory.mkdir(parents=True)
    try:
        with torch.inference_mode(), torch.device("cuda"):
            for position, token in enumerate(tokens):
                ids, logits, _ = engine.model(torch.tensor([[token]], dtype=torch.long), position)
                case["next_tokens"].append(ids.item())
                (directory / f"p{position:03}-logits.json").write_text(json.dumps(logits.float().flatten().cpu().tolist()))
                print(f"HIP geography34 position {position}: next {ids.item()}", flush=True)
        (args.output / "cases.json").write_text(json.dumps([case], ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output / "identity.json").write_text(json.dumps(engine.identity, indent=2), encoding="utf-8")
    finally:
        engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compare", "compare-components", "long-reference"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--shim", type=Path)
    parser.add_argument("--labels", nargs="+", help="explicitly compare only these named cases")
    args = parser.parse_args()
    if args.mode in ("compare", "compare-components"):
        required = ("reference", "native", "executable", "shim")
        if args.mode == "compare":
            required += ("identity",)
        if any(getattr(args, name) is None for name in required):
            parser.error("comparison needs " + " ".join("--" + n for n in required))
        (compare if args.mode == "compare" else compare_components)(args)
    else:
        long_reference(args)
