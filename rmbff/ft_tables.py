"""Recognition tables for the fine-tuned (Regime B) runs, in the same format as the frozen-backbone tables.

    RMBFF_RUNS=outputs/runs_ft python -m rmbff.ft_tables
Writes tables/T_main_fer2013_ft.csv, tables/FT_vs_frozen.csv and tables/FT_failure_acc.csv.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from .metrics import compute_metrics

FT = os.environ.get("RMBFF_RUNS", "outputs/runs_ft")
FZ = "outputs/runs"
K = ["acc", "macro_f1", "bal_acc", "auroc_macro", "ece"]
NAMES = {"m": "MobileNetV3-L", "v": "ViT-S/16", "r": "RegNetY-1.6GF", "s": "SE-ResNet-50", "mv": "RMB-FF-MV",
         "mrsv": "RMB-FF (4 branches)", "late": "Late fusion (4 models)"}


def ms(x):
    x = 100 * np.asarray(x, float)
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"


def runs(root, exp, b):
    return sorted(glob.glob(f"{root}/{exp}/branches-{b}_*"))


def metrics(root, exp, b):
    if b == "late":   # mean of the four single-backbone posteriors, per seed / fold
        out = []
        for dm in runs(root, exp, "m"):
            unit = dm.split("_")[-1]
            P = np.mean([np.load(f"{root}/{exp}/branches-{x}_{unit}/test_probs.npy") for x in "mvrs"], 0)
            y = np.load(f"{dm}/test_y.npy"); cl = json.load(open(f"{dm}/metrics.json"))["classes"]
            out.append(compute_metrics(y, P, cl))
        return out
    return [json.load(open(f"{d}/metrics.json"))["test"] for d in runs(root, exp, b)]


def main():
    rows = []
    for b in ["m", "v", "r", "s", "mv", "mrsv", "late"]:
        M = metrics(FT, "fer_main", b)
        rows.append(dict(branches=b, model=NAMES[b], n=len(M), **{k: ms([m[k] for m in M]) for k in K}))
    pd.DataFrame(rows).to_csv("tables/T_main_fer2013_ft.csv", index=False)
    rows = []
    for exp in ["fer_main", "cross_fer2cmd", "cmd_group"]:
        for b in ["m", "v", "r", "s", "mv", "mrsv", "late"]:
            a, f = metrics(FZ, exp, b), metrics(FT, exp, b)
            rows.append(dict(experiment=exp, branches=b, model=NAMES[b], n=len(f),
                             frozen_acc=ms([m["acc"] for m in a]), ft_acc=ms([m["acc"] for m in f]),
                             frozen_f1=ms([m["macro_f1"] for m in a]), ft_f1=ms([m["macro_f1"] for m in f]),
                             frozen_ece=ms([m["ece"] for m in a]), ft_ece=ms([m["ece"] for m in f])))
    pd.DataFrame(rows).to_csv("tables/FT_vs_frozen.csv", index=False)
    rows = []
    for q in ["0.0", "0.25"]:
        ds = sorted(glob.glob(f"{FT}/fer_robust_mv/branches-mv_bd-{q}_seed-*"))
        h = [json.load(open(f"{d}/metrics.json"))["test"]["acc"] for d in ds]
        n = [json.load(open(f"{d}/metrics_noV.json"))["test"]["acc"] for d in ds]
        rows.append(dict(q=float(q), n=len(ds), healthy_acc=ms(h), noV_acc=ms(n)))
    pd.DataFrame(rows).to_csv("tables/FT_failure_acc.csv", index=False)
    print(pd.read_csv("tables/T_main_fer2013_ft.csv").to_string()); print(pd.read_csv("tables/FT_vs_frozen.csv").to_string())
    print(pd.read_csv("tables/FT_failure_acc.csv").to_string())


if __name__ == "__main__":
    main()
