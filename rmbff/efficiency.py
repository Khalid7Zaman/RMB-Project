"""Computational cost of single backbones and RMB-FF variants (end-to-end, 224x224 input).

Reports parameters, forward FLOPs (torch FlopCounterMode, multiply-add = 2 FLOPs),
fp32 weight size, CPU latency at batch 1 (median of N runs after warm-up) and
throughput at batch 32. Weights do not affect cost, so random initialisation is used.

    python -m rmbff.efficiency --threads 2 --out outputs/efficiency.csv
"""
from __future__ import annotations

import argparse
import itertools
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from .backbones import BRANCHES, create_backbone
from .model import RMBFF


def flops(model, x):
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()


def latency(model, x, n=30, warm=5):
    with torch.inference_mode():
        for _ in range(warm):
            model(x)
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); model(x); t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000), float(np.percentile(t, 90) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="outputs/efficiency.csv")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    bbs = {k: create_backbone(k, pretrained=False).eval() for k in BRANCHES}
    rows = []
    subsets = [c for r in (1, 2, 3, 4) for c in itertools.combinations(sorted(BRANCHES), r)]
    for sub in subsets:
        model = RMBFF({k: bbs[k] for k in sub}, n_classes=7).eval()
        x1, x32 = torch.randn(1, 3, 224, 224), torch.randn(32, 3, 224, 224)
        with torch.inference_mode():
            f = flops(model, x1)
        med, p90 = latency(model, x1, a.n)
        t0 = time.perf_counter()
        with torch.inference_mode():
            model(x32)
        thr = 32 / (time.perf_counter() - t0)
        npar = sum(p.numel() for p in model.parameters())
        rows.append(dict(branches="".join(sub), params_M=npar / 1e6,
                         head_params_M=sum(p.numel() for p in model.head.parameters()) / 1e6,
                         gflops=f / 1e9, size_MB=npar * 4 / 2 ** 20, lat_ms_b1=med, lat_p90_ms_b1=p90,
                         thr_img_s_b32=thr, threads=a.threads, torch=torch.__version__,
                         cpu=platform.machine()))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
