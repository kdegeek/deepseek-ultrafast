"""Independent CPU float64 calibration of disagreeing BF16 projection rows."""
import argparse
import json
from pathlib import Path
import torch
from component_oracle import bounded_rows, metrics
from checkpoint import CheckpointStore
from kernel import act_quant, _fp8_decode, _fp4_decode


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    requests = json.loads((args.components / "requests.json").read_text())
    actual = json.loads((args.components / "actual.json").read_text())
    expected = json.loads((args.components / "expected.json").read_text())
    store = CheckpointStore(args.checkpoint)
    report = {"rows":[], "hyperconnection":{}}
    try:
        for request in requests:
            label, op = request["label"], request["op"]
            if op not in ("linear", "range") or not request["round"]:
                continue
            a, b = actual[label], expected[label]
            different = [i for i in range(len(a)) if a[i] != b[i]]
            if not different:
                continue
            rows = [i+request.get("first",0) for i in different]
            name = request["name"]
            w = bounded_rows(store, name, rows)
            x = torch.tensor(request["input"], dtype=torch.float32)
            if w.dtype in (torch.float8_e4m3fn, torch.int8):
                scale_rows = rows if w.dtype == torch.int8 else [i//32 for i in rows]
                scale = bounded_rows(store, name[:-6]+"scale", scale_rows).float()
                w = _fp4_decode(w) if w.dtype==torch.int8 else _fp8_decode(w)
                w = (w.reshape(len(rows), -1, 32) * scale[...,None]).flatten(1)
                if not name.endswith(".wo_a.weight"):
                    act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, True)
            raw = (w.double() @ x.double()).tolist()
            rounded = torch.tensor(raw, dtype=torch.float64).bfloat16().float().tolist()
            for i, row, exact, nearest in zip(different, rows, raw, rounded):
                midpoint = (a[i]+b[i])/2
                report["rows"].append({"label":label, "row":row, "native":a[i], "hip":b[i],
                    "f64":exact, "f64_bf16":nearest, "distance_from_midpoint":exact-midpoint,
                    "native_matches_f64_bf16":a[i]==nearest, "hip_matches_f64_bf16":b[i]==nearest})
        # Hold the attention output and residual fixed; vary only measured HC
        # coefficients, then observe the actual BF16 and FP8 stage crossings.
        r = next(r for r in requests if r["label"]=="layers.0.hc_attn/post")
        x = torch.tensor(r["residual"],dtype=torch.float32).reshape(4,-1)
        y = torch.tensor(r["input"],dtype=torch.float32)
        stages = {}
        gamma = store.tensor("layers.0.ffn_norm.weight").float()
        for kind, mix in (("native",actual["layers.0.hc_attn"]),("hip",expected["layers.0.hc_attn"])):
            post = torch.tensor(mix["post"],dtype=torch.float32)
            comb = torch.tensor(mix["comb"],dtype=torch.float32).reshape(4,4)
            pre = torch.tensor(mix["pre"],dtype=torch.float32)
            raw = post[:,None]*y[None,:]+(comb[:,:,None]*x[:,None,:]).sum(0)
            rounded = raw.bfloat16().float()
            collapsed = (pre[:,None]*rounded).sum(0).bfloat16().float()
            norm = (collapsed*torch.rsqrt(collapsed.square().mean()+1e-20)*gamma).bfloat16().float()
            quant = act_quant(norm.clone(),32,"ue8m0",torch.float8_e8m0fnu,True)
            stages[kind] = {"post_fp32":raw.flatten().tolist(), "post_bf16":rounded.flatten().tolist(),
                            "collapse_bf16":collapsed.tolist(), "norm_bf16":norm.tolist(), "quant_fp8":quant.tolist()}
        report["hyperconnection"] = {key:metrics(stages["native"][key],stages["hip"][key]) for key in stages["hip"]}
    finally:
        store.close()
    args.output.write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--components",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
