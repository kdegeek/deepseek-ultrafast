"""Calibrate identical-input native Engram gates against HIP and CPU float64."""
import argparse
import json
from pathlib import Path
import subprocess
import torch
from component_oracle import metrics


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    requests=json.loads((args.components / "requests.json").read_text())
    requests=[r for r in requests if r["op"]=="engram"]
    expected, doubles = {}, {}
    for r in requests:
        r["gates_only"]=True
        for device,dtype,target in (("cuda",torch.float32,expected),("cpu",torch.float64,doubles)):
            dim=r["dim"]
            h=torch.tensor(r["input"],device=device,dtype=dtype).reshape(-1,dim)
            key=torch.tensor(r["kv"][:h.numel()],device=device,dtype=dtype).reshape_as(h)
            q=torch.tensor(r["q"],device=device,dtype=dtype).reshape_as(h)
            k=torch.tensor(r["k"],device=device,dtype=dtype).reshape_as(h)
            rstd=torch.rsqrt(h.square().mean(-1)+r["eps"])*torch.rsqrt(key.square().mean(-1)+r["eps"])
            dot=(h*(q*k)*key).sum(-1)*rstd*dim**-.5
            gate=torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(),dot))
            target[r["label"]]=gate.cpu().tolist()
    (args.output / "requests.json").write_text(json.dumps(requests))
    subprocess.run([str(args.probe),str(args.checkpoint),str(args.output / "requests.json"),str(args.output / "actual.json")],check=True)
    actual=json.loads((args.output / "actual.json").read_text())
    report={k:{"native":a,"hip":expected[k],"f64":doubles[k],
        "native_vs_hip":metrics(a,expected[k]),"native_vs_f64":metrics(a,doubles[k]),
        "hip_vs_f64":metrics(expected[k],doubles[k])} for k,a in actual.items()}
    (args.output / "report.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--components",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--probe",type=Path,default=Path("target/release/examples/numerical_probe.exe"))
    run(p.parse_args())
