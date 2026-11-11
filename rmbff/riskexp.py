"""Risk-control experiments R1-R3 on saved posteriors (any regime: frozen heads or fine-tuned runs).

    python -m rmbff.riskexp --runs outputs/runs --reps 100
Outputs: tables/R1_cascade.csv, tables/R2_failure.csv, tables/R3_adapt.csv, outputs/risk_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch

from .analyze import load_exp
from .riskcontrol import (cascade_outcome, evaluate, grid_for, hb_pvalue, ltt_fst, online_adapt, to_actions)

GF = {"m": 0.43, "mv": 8.91, "mrsv": 20.32, "late": 20.32}   # cumulative GFLOPs (edge reuses the robot's embedding)
TIERS = ["m", "mv", "mrsv"]
G1 = [(x,) for x in np.round(np.arange(0.30, 1.0, 0.01), 2)] + [(1.01,)]
G2 = grid_for(2, list(np.round(np.arange(0.5, 1.0, 0.02), 2)) + [1.01])
G3 = grid_for(3, list(np.round(np.arange(0.5, 1.0, 0.05), 2)) + [0.99, 1.01])


def masked_probs(run_dir, mask):
    """Re-predict a saved fused head with branches removed at test time (branch failure).
    Fine-tuned runs store these posteriors directly (rmbff.ft_eval --drop), so they are loaded."""
    tag = "".join(sorted(b for b, w in mask.items() if w == 0.0)).upper()
    if os.path.exists(f"{run_dir}/test_probs_no{tag}.npy"):
        return np.load(f"{run_dir}/test_probs_no{tag}.npy")
    from .model import RMBFFHead
    from .train_head import build_split, gather, predict
    cfg = json.load(open(f"{run_dir}/config.json"))
    sp = build_split(cfg); br = sorted(cfg["branches"])
    Xtr = gather(sp["train"]["dataset"], "clean", br, sp["train"]["rows"])
    stats = {b: (Xtr[b].mean(0), Xtr[b].std(0) + 1e-6) for b in br}
    head = RMBFFHead({b: Xtr[b].shape[1] for b in br}, len(sp["classes"]), d=cfg["d"], fusion=cfg.get("fusion", "concat"),
                     p_drop=cfg["dropout"], branch_drop=cfg.get("branch_drop", 0.0), proj=cfg.get("proj", True))
    head.load_state_dict(torch.load(f"{run_dir}/head.pt"))
    Xte = gather(sp["test"]["dataset"], "clean", br, sp["test"]["rows"])
    return predict(head, Xte, stats, mask=mask)


def r1_r2(reps, eps_list, delta, failure=True):
    em = load_exp("fer_main")
    rb = load_exp("fer_robust_mv") if failure else None
    rows, frows = [], []
    for seed in sorted(em.seed.unique()):
        d = {b: em[(em.branches == b) & (em.seed == seed)].dir.iloc[0] for b in TIERS + ["v", "r", "s"]}
        y = np.load(f"{d['m']}/test_y.npy"); classes = json.load(open(f"{d['m']}/metrics.json"))["classes"]
        P = {b: np.load(f"{d[b]}/test_probs.npy") for b in d}
        P["late"] = np.mean([P[b] for b in "mvrs"], 0)
        for q in ((0.0, 0.25) if failure else ()):   # edge model trained without / with branch dropout, ViT removed
            rd = rb[(rb.seed == seed) & (rb.branch_drop == q)].dir.iloc[0]
            P[f"mv_q{q}"] = np.load(f"{rd}/test_probs.npy")
            P[f"mv_q{q}_noV"] = masked_probs(rd, {"v": 0.0})
        A = {}
        for b in P:
            A[b], ya, names = to_actions(P[b], y, classes)
        n = len(y)
        for rep in range(reps):
            rng = np.random.default_rng(1000 * int(seed) + rep)
            perm = rng.permutation(n); cal, ev = perm[: n // 2], perm[n // 2:]
            for eps in eps_list:
                def run(name, models, grid, cw=0.0, method="joint", store=rows, ev_models=None):
                    cost = [GF[b.split("_")[0]] for b in models]
                    PAc = [A[b][cal] for b in models]
                    if method == "joint":
                        t = ltt_fst(PAc, ya[cal], grid, cost, eps, delta, cw, rng)
                    elif method == "naive":
                        t = tuple(ltt_fst([A[b][cal]], ya[cal], G1, [c], eps, delta, 0.0, rng)[0] for b, c in zip(models, cost))
                    elif method == "always":
                        t = tuple([0.0] * len(models))
                    else:
                        t = tuple([0.9] * len(models))
                    evm = ev_models or models
                    r = evaluate([A[b][ev] for b in evm], ya[ev], t, cost)
                    rec = dict(name=name, seed=int(seed), rep=rep, eps=eps, cw=cw, t=list(map(float, t)), risk=r["risk"],
                               decisive=r["decisive"], cost=r["cost"], cond_err=r["cond_err"],
                               tier_share=r["tier_share"], ok=r["risk"] <= eps)
                    store.append(rec)
                    return t
                run("Robot only, always act (top-1)", ["m"], G1, method="always")
                run("Cloud only, always act (top-1)", ["mrsv"], G1, method="always")
                run("Robot only (MobileNetV3-L)", ["m"], G1)
                run("Edge only (RMB-FF-MV)", ["mv"], G1)
                run("Cloud only (RMB-FF, 4 branches)", ["mrsv"], G1)
                run("Cloud only (late fusion, 4 models)", ["late"], G1)
                run("Robot→edge, fixed softmax threshold 0.9", ["m", "mv"], G2, method="fixed")
                run("Robot→edge, per-tier calibration", ["m", "mv"], G2, method="naive")
                run("Robot→edge, joint calibration (decisiveness-first)", ["m", "mv"], G2, cw=0.0)
                run("Robot→edge, joint calibration (cost-aware)", ["m", "mv"], G2, cw=0.25)
                run("Robot→edge→cloud, joint calibration (decisiveness-first)", TIERS, G3, cw=0.0)
                run("Robot→edge→cloud, joint calibration (cost-aware)", TIERS, G3, cw=0.25)
                # R2: ViT-branch failure at the edge (models with branch-dropout rate q in training)
                for q in ((0.0, 0.25) if failure else ()):
                    t = run(f"q={q}: healthy", ["m", f"mv_q{q}"], G2, cw=0.25, store=frows)
                    r = evaluate([A["m"][ev], A[f"mv_q{q}_noV"][ev]], ya[ev], t, [GF["m"], GF["mv"]])
                    frows.append(dict(name=f"q={q}: ViT branch down, thresholds kept", seed=int(seed), rep=rep, eps=eps,
                                      cw=0.25, risk=r["risk"], decisive=r["decisive"], cost=r["cost"],
                                      cond_err=r["cond_err"], ok=r["risk"] <= eps))
                    run(f"q={q}: ViT branch down, recalibrated", ["m", f"mv_q{q}_noV"], G2, cw=0.25, store=frows)
                if failure:
                    run("Edge unreachable: robot only, recalibrated", ["m"], G1, store=frows)
    return pd.DataFrame(rows), pd.DataFrame(frows)


def cmd_streams(rows_df, rng):
    """Order CMD test frames into per-group streams: emotion episodes in random order, frames in file order."""
    k = rows_df.path.map(lambda p: int(re.search(r"\((\d+)\)", os.path.basename(p)).group(1))
                         if re.search(r"\((\d+)\)", os.path.basename(p)) else
                         int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    rows_df = rows_df.assign(k=k.values)
    streams = {}
    for g, gdf in rows_df.groupby("group"):
        eps_order = list(gdf.label_name.unique()); rng.shuffle(eps_order)
        order = np.concatenate([gdf[gdf.label_name == e].sort_values("k").pos.values for e in eps_order])
        streams[g] = order
    return streams


def r3(eps_list, delta, gammas=(0.005, 0.02)):
    cmd = pd.read_csv("outputs/manifests/cmd_groupfolds.csv")
    out = []
    settings = [("FER2013 -> DFERCD (cross-dataset)", "cross_fer2cmd", "seed"),
                ("DFERCD group-disjoint folds", "cmd_group", "fold")]
    for sname, exp, unit in settings:
        e = load_exp(exp)
        for b in ["mrsv", "mv"]:
            for _, r in e[e.branches == b].iterrows():
                d = r.dir
                classes = json.load(open(f"{d}/metrics.json"))["classes"]
                pv, pt = np.load(f"{d}/val_probs.npy"), np.load(f"{d}/test_probs.npy")
                yt, rows = np.load(f"{d}/test_y.npy"), np.load(f"{d}/test_rows.npy")
                cfg = json.load(open(f"{d}/config.json"))
                from .train_head import build_split
                yv = build_split(cfg)["val"]["y"]
                Av, yav, _ = to_actions(pv, yv, classes)
                At, yat, _ = to_actions(pt, yt, classes)
                sub = cmd.iloc[rows].reset_index(drop=True).assign(pos=np.arange(len(rows)))
                streams = cmd_streams(sub, np.random.default_rng(int(r[unit]) if r[unit] == r[unit] else 0))
                for eps in eps_list:
                    t0 = ltt_fst([Av], yav, G1, [GF[b]], eps, delta, 0.0, np.random.default_rng(7))[0]
                    for g, order in streams.items():
                        conf, pred, ya = At[order].max(1), At[order].argmax(1), yat[order]
                        acted = conf >= t0; wrong = acted & (pred != ya)
                        base = dict(setting=sname, model=b, unit=int(r[unit]), eps=eps, group=g, n=len(order),
                                    t0=float(t0))
                        out.append(dict(base, method="static (calibrated on source)", risk=wrong.mean(),
                                        decisive=acted.mean()))
                        for gm in gammas:
                            a2, w2 = online_adapt(conf, pred, ya, t0, eps, gm)
                            out.append(dict(base, method=f"person-adaptive (gamma={gm})", risk=w2.mean(),
                                            decisive=a2.mean()))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--eps", nargs="+", type=float, default=[0.05, 0.10])
    ap.add_argument("--tag", default="", help="suffix for output tables, e.g. _ft for fine-tuned runs")
    ap.add_argument("--no-failure", action="store_true", help="skip R2 (needs frozen-head checkpoints)")
    a = ap.parse_args()
    os.makedirs("tables", exist_ok=True)
    r1, r2 = r1_r2(a.reps, a.eps, a.delta, failure=not a.no_failure)
    r1.to_csv(f"tables/R1_cascade_raw{a.tag}.csv", index=False); r2.to_csv(f"tables/R2_failure_raw{a.tag}.csv", index=False)
    agg = lambda df: df.groupby(["eps", "name"]).agg(risk=("risk", "mean"), risk_q90=("risk", lambda x: x.quantile(.9)),
                                                    ok=("ok", "mean"), decisive=("decisive", "mean"),
                                                    decisive_sd=("decisive", "std"), cost=("cost", "mean"),
                                                    cond_err=("cond_err", "mean"), n=("risk", "size")).reset_index()
    agg(r1).to_csv(f"tables/R1_cascade{a.tag}.csv", index=False)
    if len(r2):
        agg(r2).to_csv(f"tables/R2_failure{a.tag}.csv", index=False)
    shares = r1[r1.name.str.contains("proposed")].groupby("eps").tier_share.apply(
        lambda s: np.mean(np.vstack(s.values), 0).tolist()).to_dict()
    d3 = r3(a.eps, a.delta)
    d3.to_csv(f"tables/R3_adapt_raw{a.tag}.csv", index=False)
    g3 = d3.groupby(["setting", "model", "eps", "method"]).agg(
        risk_mean=("risk", "mean"), risk_max=("risk", "max"),
        frac_groups_ok=("risk", lambda x: None), decisive=("decisive", "mean"), groups=("group", "nunique"),
        streams=("risk", "size")).reset_index()
    ok = d3.assign(ok=lambda x: x.risk <= x.eps * 1.0 + 0.01).groupby(["setting", "model", "eps", "method"]).ok.mean().values
    g3["frac_streams_within_eps_plus_1pt"] = ok; g3 = g3.drop(columns="frac_groups_ok")
    g3.to_csv(f"tables/R3_adapt{a.tag}.csv", index=False)
    json.dump(dict(tier_share=shares, delta=a.delta, reps=a.reps), open(f"outputs/risk_summary{a.tag}.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(agg(r1).round(4).to_string()); print(agg(r2).round(4).to_string() if len(r2) else ""); print(g3.round(4).to_string())


if __name__ == "__main__":
    main()
