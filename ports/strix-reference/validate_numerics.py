"""Unchanged original HIP versus native teacher-forced, all-vocabulary validation.

Writes new evidence only. Exact token and quantitative full-logit results are
reported separately; completion matches cannot promote numerical acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(ROOT))


def compare(args, cases, identity, hip_root):
    from component_oracle import metrics
    (args.output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    binary_hash = hashlib.sha256(args.native.read_bytes()).hexdigest()
    subprocess.run([str(args.native), str(args.checkpoint), str(args.output / "cases.json"), str(args.output / "native")], check=True)
    report = {"identity":identity, "teacher_forced":True, "cases":{}, "hip_logits":str(hip_root),
              "native_executable_sha256":binary_hash,
              "acceptance":"Exact next-token matches and all-logit errors reported independently; no new tolerance."}
    for case in cases:
        label = case["label"]
        native = json.loads((args.output / "native" / label / "summary.json").read_text())
        positions = []
        for position in range(len(case["tokens"])):
            name = f"p{position:03}-logits.json"
            hip = json.loads((hip_root / label / name).read_text())
            actual = json.loads((args.output / "native" / label / name).read_text())
            positions.append({"position":position, "input_token":case["tokens"][position],
                              "hip_next":case["next_tokens"][position], "native_next":native["next_tokens"][position],
                              "greedy_match":case["next_tokens"][position] == native["next_tokens"][position],
                              **metrics(actual, hip)})
        report["cases"][label] = {"prompt":case["prompt"], "positions":positions, "native_memory":native["weight_cache"],
                                  "peak_read_bytes":native["peak_read_bytes"]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    if args.reuse_reference:
        cases = json.loads((args.reuse_reference / "cases.json").read_text(encoding="utf-8"))
        identity = json.loads((args.reuse_reference / "report.json").read_text())["identity"]
        return compare(args, cases, identity, args.reuse_reference / "hip")
    import torch
    from engine import ReferenceEngine, reset_request_state
    from encoding import encode_messages
    engine = ReferenceEngine(args.checkpoint, context=128, prefill_batch=1)
    cases = []
    specifications = [
        ("the", "The", []),
        ("france", "The capital of France is", [11111, 16]),
        ("chat", encode_messages([{"role":"user", "content":"Reply with only OK."}], thinking_mode="chat"), [11932]),
        ("arithmetic", "2 + 2 =", []),
        ("story", "Once upon a time", []),
        ("hello", "Hello", []),
    ]
    try:
        with torch.inference_mode(), torch.device("cuda"):
            for label, prompt, continuation in specifications:
                reset_request_state(engine.model, engine.official)
                encoded = engine.tokenizer.encode(prompt, add_special_tokens=False)
                tokens = encoded + continuation
                case = {"label":label, "prompt":prompt, "tokens":tokens, "prompt_tokens":encoded, "next_tokens":[]}
                cases.append(case)
                directory = args.output / "hip" / label
                directory.mkdir(parents=True)
                for position, token in enumerate(tokens):
                    ids, logits, _ = engine.model(torch.tensor([[token]], dtype=torch.long), position)
                    case["next_tokens"].append(ids.item())
                    (directory / f"p{position:03}-logits.json").write_text(json.dumps(logits.float().flatten().cpu().tolist()))
                    print(f"HIP {label} position {position}: next {ids.item()}", flush=True)
                (directory / "summary.json").write_text(json.dumps(case, indent=2))
    finally:
        engine.close()
    compare(args, cases, engine.identity, args.output / "hip")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--native", type=Path, default=Path("target/release/examples/teacher_force.exe"))
    p.add_argument("--reuse-reference", type=Path, help="reuse a completed, unchanged HIP run; write fresh native evidence")
    run(p.parse_args())
