"""Train / evaluate RMB-FF fusion heads on cached frozen-backbone features.

    python -m rmbff.train_head --exp fer_single fer_combos ...   (see configs/experiments.yaml)

Each run writes outputs/runs/<exp>/<run_id>/ with
    config.json, metrics.json (val/test + extra test sets), curves.csv,
    val_probs.npy, test_probs.npy, test_y.npy, head.pt
Model selection uses the validation split only (macro-F1, early stopping);
test data are never used for any choice.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml

from .data import CANON, SHARED6
from .losses import ImbalanceLoss
from .metrics import compute_metrics
from .model import RMBFFHead

MAN = os.environ.get("RMBFF_MAN", "outputs/manifests")
FEAT = os.environ.get("RMBFF_FEAT", "outputs/features")
_CACHE: dict = {}


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_feats(dataset: str, variant: str, branches) -> tuple[dict, np.ndarray]:
    """Per-branch cache (float32) shared by all branch subsets, to bound memory."""
    d = os.path.join(FEAT, dataset, variant)
    ik = (dataset, variant, "index")
    if ik not in _CACHE:
        _CACHE[ik] = np.load(os.path.join(d, "index.npy"))
    out = {}
    for b in branches:
        k = (dataset, variant, b)
        if k not in _CACHE:
            if variant != "clean":  # corrupted sets are used once per run; do not keep them
                out[b] = np.load(os.path.join(d, f"{b}.npy")).astype(np.float32)
                continue
            _CACHE[k] = np.load(os.path.join(d, f"{b}.npy")).astype(np.float32)
        out[b] = _CACHE[k]
    return out, _CACHE[ik]


def manifest(dataset: str) -> pd.DataFrame:
    if dataset == "fer2013":
        return pd.read_csv(f"{MAN}/fer2013.csv")
    if dataset == "cmd":
        return pd.read_csv(f"{MAN}/cmd_groupfolds.csv")
    if dataset == "cmd_random":
        return pd.read_csv(f"{MAN}/cmd_randomfolds.csv")
    raise ValueError(dataset)


def build_split(cfg) -> dict:
    """Return dict split -> (dataset, variant, row indices into manifest, labels, classes)."""
    classes = cfg.get("classes") or None
    proto = cfg["protocol"]
    out = {}
    if proto == "fer_official":
        df = manifest("fer2013")
        classes = classes or [c for c in CANON if c in set(df.label_name)]
        drop = set()
        if cfg.get("dedup"):  # drop val/test images with an exact duplicate in train
            dup = pd.read_csv(f"{MAN}/fer2013_dup_rows.csv")
            drop = set(dup.row)
        for sp in ("train", "val", "test"):
            m = (df.split == sp) & df.label_name.isin(classes)
            if sp != "train":
                m &= ~df.index.isin(drop)
            out[sp] = ("fer2013", np.where(m)[0])
        dfs = {"fer2013": df}
    elif proto in ("cmd_group", "cmd_random"):
        df = manifest("cmd" if proto == "cmd_group" else "cmd_random")
        classes = classes or [c for c in CANON if c in set(df.label_name)]
        k, K = cfg["fold"], 5
        lab_ok = df.label_name.isin(classes)
        out["test"] = ("cmd", np.where((df.fold == k) & lab_ok)[0])
        out["val"] = ("cmd", np.where((df.fold == (k + 1) % K) & lab_ok)[0])
        out["train"] = ("cmd", np.where(~df.fold.isin([k, (k + 1) % K]) & lab_ok)[0])
        dfs = {"cmd": df}
    elif proto == "fer_to_cmd":
        fer, cmd = manifest("fer2013"), manifest("cmd")
        classes = classes or SHARED6
        for sp in ("train", "val"):
            out[sp] = ("fer2013", np.where((fer.split == sp) & fer.label_name.isin(classes))[0])
        out["test"] = ("cmd", np.where(cmd.label_name.isin(classes))[0])
        dfs = {"fer2013": fer, "cmd": cmd}
    else:
        raise ValueError(proto)
    cmap = {c: i for i, c in enumerate(classes)}
    res = {}
    for sp, (ds, rows) in out.items():
        res[sp] = dict(dataset=ds, rows=rows, y=dfs[ds].label_name.values[rows])
        res[sp]["y"] = np.array([cmap[c] for c in res[sp]["y"]])
    res["classes"] = classes
    return res


def gather(dataset, variant, branches, rows):
    F, idx = load_feats(dataset, variant, branches)
    pos = {r: i for i, r in enumerate(idx)} if len(idx) != idx.max() + 1 or not (idx == np.arange(len(idx))).all() else None
    sel = rows if pos is None else np.array([pos[r] for r in rows])
    return {b: F[b][sel] for b in branches}


def predict(head, X, stats, mask=None, bs=4096):
    head.eval()
    outs = []
    n = len(next(iter(X.values())))
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = {b: torch.from_numpy((X[b][i:i + bs] - stats[b][0]) / stats[b][1]) for b in X}
            outs.append(torch.softmax(head(xb, mask=mask), 1).numpy())
    return np.concatenate(outs)


HP_KEYS = ("loss", "d", "dropout", "wd", "branch_drop")


def resolve(cfg: dict) -> dict:
    """Fill validation-selected settings from outputs/selection.json.

    * loss == "SELECTED_LOSS": objective chosen in stage 1 (E1) for this branch set.
    * tuned == "self": all of HP_KEYS chosen in stage 2 (E1b) for this branch set.
    * tuned == "fused": HP_KEYS of the 4-branch model (used for subset/fusion ablations).
    Single-branch sets fall back to their own entry; unknown multi-branch sets use the 4-branch entry.
    """
    sel = json.load(open("outputs/selection.json")) if os.path.exists("outputs/selection.json") else {}
    key = "".join(sorted(cfg["branches"]))
    if cfg.get("loss") == "SELECTED_LOSS":
        cfg = dict(cfg, loss=sel["loss"].get(key, sel["loss"]["mrsv"]))
    if cfg.get("tuned"):
        hp = sel["hp"]
        src = hp.get(key, hp["mrsv"]) if cfg["tuned"] == "self" else hp["mrsv"]
        fixed = set(cfg.get("_grid", []))  # grid-varied keys are never overridden
        cfg = dict(cfg, **{k: src[k] for k in HP_KEYS if k in src and k not in fixed})
        if len(cfg["branches"]) == 1:
            cfg["branch_drop"] = 0.0
    return cfg


def run(cfg: dict, out_dir: str) -> dict:
    cfg = resolve(cfg)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        return json.load(open(os.path.join(out_dir, "metrics.json")))
    set_seed(cfg["seed"])
    sp = build_split(cfg)
    classes, br = sp["classes"], sorted(cfg["branches"])
    X = {s: gather(sp[s]["dataset"], "clean", br, sp[s]["rows"]) for s in ("train", "val", "test")}
    # standardisation statistics fitted on the training split only
    stats = {b: (X["train"][b].mean(0), X["train"][b].std(0) + 1e-6) for b in br}
    ytr = sp["train"]["y"]
    counts = np.bincount(ytr, minlength=len(classes))
    head = RMBFFHead({b: X["train"][b].shape[1] for b in br}, len(classes), d=cfg.get("d", 256),
                     fusion=cfg.get("fusion", "concat"), p_drop=cfg.get("dropout", 0.3),
                     branch_drop=cfg.get("branch_drop", 0.0), proj=cfg.get("proj", True))
    crit = ImbalanceLoss(cfg.get("loss", "ce"), counts, beta=cfg.get("beta", 0.999),
                         gamma=cfg.get("gamma", 1.0), tau=cfg.get("tau", 1.0))
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 0.05))
    E, bs, pat = cfg.get("epochs", 30), cfg.get("bs", 256), cfg.get("patience", 6)
    steps = E * int(np.ceil(len(ytr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 1e-3), total_steps=steps, pct_start=0.1)
    Xt = {b: torch.from_numpy((X["train"][b] - stats[b][0]) / stats[b][1]) for b in br}
    yt = torch.from_numpy(ytr).long()
    yv = sp["val"]["y"]
    g = torch.Generator().manual_seed(cfg["seed"])
    best, best_state, bad, curves, t0 = -1, None, 0, [], time.time()
    for ep in range(E):
        head.train()
        perm = torch.randperm(len(yt), generator=g)
        tl = 0.0
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            loss = crit(head({b: Xt[b][j] for b in br}), yt[j])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(j)
        pv = predict(head, X["val"], stats)
        mv = compute_metrics(yv, pv, classes)
        curves.append(dict(epoch=ep + 1, train_loss=tl / len(yt), val_nll=mv["nll"], val_acc=mv["acc"],
                           val_macro_f1=mv["macro_f1"]))
        if mv["macro_f1"] > best + 1e-5:
            best, best_state, bad = mv["macro_f1"], copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= pat:
                break
    train_time = time.time() - t0
    head.load_state_dict(best_state)
    pv = predict(head, X["val"], stats)
    pt = predict(head, X["test"], stats)
    res = dict(val=compute_metrics(yv, pv, classes), test=compute_metrics(sp["test"]["y"], pt, classes),
               epochs_run=len(curves), best_epoch=int(np.argmax([c["val_macro_f1"] for c in curves]) + 1),
               train_seconds=train_time, n_params_head=sum(p.numel() for p in head.parameters()),
               classes=classes, n_train=int(len(ytr)), n_val=int(len(yv)), n_test=int(len(pt)))
    # extra evaluations: corrupted test sets and test-time branch removal
    for v in cfg.get("eval_variants", []):
        try:
            Xc = gather(sp["test"]["dataset"], v, br, sp["test"]["rows"])
            pc = predict(head, Xc, stats)
            np.save(os.path.join(out_dir, f"test_probs@{v}.npy"), pc.astype(np.float32))
            res[f"test@{v}"] = {k: val for k, val in compute_metrics(sp["test"]["y"], pc, classes).items()
                                if k not in ("per_class",)}
        except (FileNotFoundError, KeyError) as e:
            res[f"test@{v}"] = dict(error=str(e))
    if cfg.get("eval_missing") and len(br) > 1:
        for b in br:
            pm = predict(head, X["test"], stats, mask={b: 0.0})
            m = compute_metrics(sp["test"]["y"], pm, classes)
            res[f"test@missing_{b}"] = dict(acc=m["acc"], macro_f1=m["macro_f1"])
    if cfg.get("fusion") == "gated":
        head.eval()
        with torch.no_grad():
            head({b: torch.from_numpy((X["test"][b] - stats[b][0]) / stats[b][1]) for b in br})
        res["gate_mean"] = dict(zip(head.keys, head.last_gate.mean(0).tolist()))
    pd.DataFrame(curves).to_csv(os.path.join(out_dir, "curves.csv"), index=False)
    np.save(os.path.join(out_dir, "val_probs.npy"), pv.astype(np.float32))
    np.save(os.path.join(out_dir, "test_probs.npy"), pt.astype(np.float32))
    np.save(os.path.join(out_dir, "test_y.npy"), sp["test"]["y"])
    np.save(os.path.join(out_dir, "test_rows.npy"), sp["test"]["rows"])
    torch.save(head.state_dict(), os.path.join(out_dir, "head.pt"))
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    return res


def expand(exp: dict) -> list[dict]:
    base = exp.get("base", {})
    grid = exp.get("grid", {})
    keys = list(grid)
    cfgs = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        c = dict(base); c.update(dict(zip(keys, vals))); c["_grid"] = keys
        cfgs.append(c)
    return cfgs


def run_id(cfg: dict, grid_keys) -> str:
    parts = []
    for k in grid_keys:
        v = cfg[k]
        v = "".join(sorted(v)) if isinstance(v, list) else v
        parts.append(f"{k}-{v}")
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    allexp = yaml.safe_load(open(a.config))
    for name in a.exp:
        exp = allexp[name]
        cfgs = expand(exp)
        print(f"[{name}] {len(cfgs)} runs", flush=True)
        for c in cfgs:
            rid = run_id(c, list(exp.get("grid", {})))
            t = time.time()
            r = run(c, os.path.join(os.environ.get("RMBFF_RUNS", "outputs/runs"), name, rid))
            print(f"  {rid}: val F1 {r['val']['macro_f1']:.4f} test acc {r['test']['acc']:.4f} "
                  f"F1 {r['test']['macro_f1']:.4f} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
