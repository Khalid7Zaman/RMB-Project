"""E9: protocol sensitivity on FER2013 (frozen features).

Compares, for the same models and hyper-parameters:
  official      : Training / PublicTest / PrivateTest (as in the main study)
  random        : stratified random 70/30 image-level split of all 35,887 images
  aug_after     : random 70/30 split of the ORIGINAL images, augmented copies added to training only (correct)
  aug_before    : originals + 2 augmented copies pooled, THEN random 70/30 split (copies of one image can
                  fall on both sides; the protocol used in many FER papers)
In every protocol 10% of the training part is held out for early stopping.

    python -m rmbff.protocol_study --models v mv --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .losses import ImbalanceLoss
from .metrics import compute_metrics
from .model import RMBFFHead
from .train_head import FEAT, resolve, set_seed

OUT = "outputs/runs/protocol_study"


def feats(variant, branches):
    d = os.path.join(FEAT, "fer2013", variant)
    return {b: np.load(os.path.join(d, f"{b}.npy")).astype(np.float32) for b in branches}


def train_eval(X, y, idx_tr, idx_va, idx_te, cfg, classes, seed):
    set_seed(seed)
    br = sorted(X)
    stats = {b: (X[b][idx_tr].mean(0), X[b][idx_tr].std(0) + 1e-6) for b in br}
    T = lambda ii: {b: torch.from_numpy((X[b][ii] - stats[b][0]) / stats[b][1]) for b in br}
    counts = np.bincount(y[idx_tr], minlength=len(classes))
    head = RMBFFHead({b: X[b].shape[1] for b in br}, len(classes), d=cfg["d"], fusion="concat",
                     p_drop=cfg["dropout"], branch_drop=cfg["branch_drop"] if len(br) > 1 else 0.0)
    crit = ImbalanceLoss(cfg["loss"], counts)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=cfg["wd"])
    E, bs = 30, 256
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=E * int(np.ceil(len(idx_tr) / bs)),
                                                pct_start=0.1)
    Xt, yt = T(idx_tr), torch.from_numpy(y[idx_tr]).long()
    Xv, Xe = T(idx_va), T(idx_te)
    g = torch.Generator().manual_seed(seed)
    best, state, bad = -1, None, 0

    def pred(Z):
        head.eval()
        with torch.no_grad():
            return torch.softmax(head(Z), 1).numpy()
    for ep in range(E):
        head.train()
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            loss = crit(head({b: Xt[b][j] for b in br}), yt[j])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        f1 = compute_metrics(y[idx_va], pred(Xv), classes)["macro_f1"]
        if f1 > best + 1e-5:
            best, state, bad = f1, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= 6:
                break
    head.load_state_dict(state)
    m = compute_metrics(y[idx_te], pred(Xe), classes)
    return {k: m[k] for k in ("acc", "macro_f1", "bal_acc")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v", "mv"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    fer = pd.read_csv("outputs/manifests/fer2013.csv")
    classes = [c for c in ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]]
    y0 = fer.label.values
    N = len(fer)
    rows = []
    for mk in a.models:
        br = list(mk)
        cfg = resolve(dict(branches=br, tuned="self", loss="ce", d=256, dropout=0.3, wd=0.05, branch_drop=0.0))
        Xc, Xf, Xr = feats("clean", br), feats("aug_flip_s0", br), feats("aug_rot_s0", br)
        Xall = {b: np.concatenate([Xc[b], Xf[b], Xr[b]]) for b in br}      # rows: orig, flip, rot
        yall = np.concatenate([y0, y0, y0])
        src = np.concatenate([np.arange(N)] * 3)                            # original image id
        for seed in a.seeds:
            f = os.path.join(OUT, f"{mk}_s{seed}.json")
            if os.path.exists(f):
                rows += json.load(open(f)); continue
            res = []
            # official
            tr = np.where(fer.split == "train")[0]; va = np.where(fer.split == "val")[0]; te = np.where(fer.split == "test")[0]
            res.append(dict(protocol="official", **train_eval(Xc, y0, tr, va, te, cfg, classes, seed)))
            # random image-level 70/30
            tr_all, te = train_test_split(np.arange(N), test_size=0.3, stratify=y0, random_state=seed)
            tr, va = train_test_split(tr_all, test_size=0.1, stratify=y0[tr_all], random_state=seed)
            res.append(dict(protocol="random", **train_eval(Xc, y0, tr, va, te, cfg, classes, seed)))
            # augmentation AFTER split (correct): copies of training originals only
            tr_aug = np.concatenate([tr, tr + N, tr + 2 * N])
            res.append(dict(protocol="aug_after", **train_eval(Xall, yall, tr_aug, va, te, cfg, classes, seed)))
            # augmentation BEFORE split (leaky): split the pooled copies
            idx = np.arange(3 * N)
            tr_all, te = train_test_split(idx, test_size=0.3, stratify=yall, random_state=seed)
            tr, va = train_test_split(tr_all, test_size=0.1, stratify=yall[tr_all], random_state=seed)
            leak = np.isin(src[te], src[tr]).mean()
            r = train_eval(Xall, yall, tr, va, te, cfg, classes, seed)
            res.append(dict(protocol="aug_before", leak_frac=float(leak), **r))
            for r in res:
                r.update(model=mk, seed=seed)
            json.dump(res, open(f, "w"), indent=1)
            rows += res
            print(mk, seed, [(r["protocol"], round(r["acc"], 4)) for r in res], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("tables/E9_protocol.csv", index=False)
    print(df.groupby(["model", "protocol"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
