"""Re-score fine-tuned runs at full fp32 precision from their saved best.pt (no retraining).

Why: with fp16 autocast, the final test scoring of 7 cross-dataset runs (FER2013 -> DFERCD) overflowed
and produced NaN posteriors after training had finished normally. To keep one evaluation protocol for
every run, all runs are re-scored in fp32 with this script. Model selection is unchanged: best.pt is the
checkpoint with the highest validation macro-F1, chosen during training.

    python -m rmbff.ft_eval --index $SLURM_ARRAY_TASK_ID     # one run (array 0-89)
    python -m rmbff.ft_eval --list                           # print the run order
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch

from .finetune import E2E, loaders
from .metrics import compute_metrics

PROTO = {"fer_main": "fer_official", "cross_fer2cmd": "fer_to_cmd", "cmd_group": "cmd_group"}
BR = {"m": ["m"], "v": ["v"], "r": ["r"], "s": ["s"], "mv": ["m", "v"], "mrsv": ["m", "r", "s", "v"]}


def runs(exp="*"):
    """Main fine-tuning runs (90). The branch-failure runs are scored by slurm_ft_failure.sh itself."""
    return sorted(os.path.dirname(p) for p in glob.glob(f"outputs/runs_ft/{exp}/*/best.pt") if "fer_robust_mv" not in p)


def config_for(d):
    """The run's own config.json if it was written, else rebuilt exactly as slurm_finetune.sh built it."""
    f = f"{d}/config.json"
    if os.path.exists(f):
        return json.load(open(f))
    exp, name = d.split("/")[-2], d.split("/")[-1]
    m = re.match(r"branches-(\w+?)_(seed|fold)-(\d+)$", name)
    tag, unit, k = m.group(1), m.group(2), int(m.group(3))
    hp = json.load(open("outputs/selection.json"))["hp"]
    x = hp.get(tag, hp["mrsv"])
    return dict(protocol=PROTO[exp], fold=k if unit == "fold" else 0, seed=k if unit == "seed" else 0,
                branches=BR[tag], loss=x["loss"], fusion="concat",
                branch_drop=float(x["branch_drop"]) if len(tag) > 1 else 0.0,
                epochs=30, bs=64, lr_backbone=5e-5, lr_head=1e-3, wd=0.05, label_smoothing=0.0, patience=6,
                workers=8, dedup=False, resume=True, out=d, rebuilt_config=True)


@torch.no_grad()
def infer32(model, dl, dev, mask=None):
    """fp32 posteriors; mask={"v": 0.0} removes a branch at test time (branch-failure analysis)."""
    model.eval(); P = []
    for x, _ in dl:
        x = x.to(dev, non_blocking=True)
        logits = model(x) if mask is None else model.head({k: m(x) for k, m in model.backbones.items()}, mask=mask)
        P.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", help="score this run directory instead of --index")
    ap.add_argument("--drop", nargs="+", help="branch(es) removed at test time; writes test_probs_no<B>.npy only")
    a = ap.parse_args()
    R = runs()
    if a.list:
        for i, d in enumerate(R):
            print(i, d)
        print("runs:", len(R)); return
    d = a.run.rstrip("/") if a.run else R[a.index]
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # strict fp32
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = config_for(d)
    sp, dl = loaders(cfg, 128, 8)
    classes = sp["classes"]
    model = E2E(sorted(cfg["branches"]), len(classes), fusion=cfg.get("fusion", "concat"),
                branch_drop=cfg.get("branch_drop", 0.0)).to(dev)
    model.load_state_dict(torch.load(f"{d}/best.pt", map_location=dev))
    if a.drop:
        tag = "".join(sorted(a.drop)).upper()
        pt = infer32(model, dl["test"], dev, mask={b: 0.0 for b in a.drop})
        assert np.isfinite(pt).all(), f"non-finite posteriors: {d}"
        np.save(f"{d}/test_probs_no{tag}.npy", pt)
        mt = compute_metrics(sp["test"]["y"], pt, classes)
        json.dump(dict(dropped=a.drop, test=mt), open(f"{d}/metrics_no{tag}.json", "w"), indent=1)
        print("FP32 no-" + tag, d, {k: round(mt[k], 4) for k in ("acc", "macro_f1")}); return
    pv, pt = infer32(model, dl["val"], dev), infer32(model, dl["test"], dev)
    assert np.isfinite(pv).all() and np.isfinite(pt).all(), f"non-finite posteriors in fp32: {d}"
    ck = torch.load(f"{d}/last.pt", map_location="cpu", weights_only=False)
    curves = ck["curves"]
    if os.path.exists(f"{d}/metrics.json") and not os.path.exists(f"{d}/metrics_fp16.json"):
        os.rename(f"{d}/metrics.json", f"{d}/metrics_fp16.json")   # keep the original fp16 scoring for audit
    res = dict(val=compute_metrics(sp["val"]["y"], pv, classes), test=compute_metrics(sp["test"]["y"], pt, classes),
               classes=classes, epochs_run=len(curves), train_seconds=curves[-1]["seconds"],
               n_train=int(len(sp["train"]["y"])), n_val=int(len(sp["val"]["y"])), n_test=int(len(sp["test"]["y"])),
               gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", torch=torch.__version__,
               eval_precision="fp32", best_epoch=int(np.argmax([c["val_macro_f1"] for c in curves]) + 1))
    pd.DataFrame(curves).to_csv(f"{d}/curves.csv", index=False)
    np.save(f"{d}/val_probs.npy", pv); np.save(f"{d}/test_probs.npy", pt)
    np.save(f"{d}/test_y.npy", sp["test"]["y"]); np.save(f"{d}/test_rows.npy", sp["test"]["rows"])
    json.dump(cfg, open(f"{d}/config.json", "w"), indent=1)
    json.dump(res, open(f"{d}/metrics.json", "w"), indent=1)
    print("FP32", d, {k: round(res["test"][k], 4) for k in ("acc", "macro_f1", "bal_acc", "ece")})


if __name__ == "__main__":
    main()
