"""Aggregate saved run outputs into tables (CSV/JSON) and statistical tests.

    python -m rmbff.analyze

Every number reported in the manuscript is produced here from outputs/runs/*
and written to tables/*.csv and outputs/summary.json.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

from .metrics import cohens_dz, compute_metrics, holm, mcnemar, paired_bootstrap_diff

R = os.environ.get("RMBFF_RUNS", "outputs/runs")
T = "tables"
os.makedirs(T, exist_ok=True)
BR = {"m": "MobileNetV3-L", "v": "ViT-S/16", "r": "RegNetY-1.6GF", "s": "SE-ResNet-50",
      "mrsv": "RMB-FF (4 branches)", "mv": "RMB-FF-MV (val.-selected subset)", "late": "Late fusion (4 posteriors)",
      "late_mv": "Late fusion (MobileNetV3-L + ViT-S)"}
METRICS = ["acc", "macro_f1", "bal_acc", "weighted_f1", "auroc_macro", "auprc_macro", "ece", "nll"]


def mf1(y, p):
    return f1_score(y, p, average="macro")


def load_exp(exp: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(f"{R}/{exp}/*/metrics.json")):
        d = os.path.dirname(f)
        c, m = json.load(open(f"{d}/config.json")), json.load(open(f))
        row = dict(dir=d, branches="".join(sorted(c["branches"])), loss=c.get("loss"), seed=c.get("seed"),
                   fold=c.get("fold"), fusion=c.get("fusion", "concat"), proj=c.get("proj", True),
                   branch_drop=c.get("branch_drop", 0.0), d=c.get("d"), dropout=c.get("dropout"), wd=c.get("wd"),
                   epochs_run=m["epochs_run"], train_s=m["train_seconds"], n_test=m["n_test"])
        for k in METRICS:
            row[f"val_{k}"] = m["val"][k]; row[f"test_{k}"] = m["test"][k]
        for k, v in m.items():
            if k.startswith("test@") and isinstance(v, dict) and "macro_f1" in v:
                row[f"{k}_acc"] = v["acc"]; row[f"{k}_macro_f1"] = v["macro_f1"]
        if "gate_mean" in m:
            for b, g in m["gate_mean"].items():
                row[f"gate_{b}"] = g
        rows.append(row)
    return pd.DataFrame(rows)


def ms(x):
    x = np.asarray(x, float)
    return f"{100 * x.mean():.2f} ± {100 * x.std(ddof=1):.2f}" if len(x) > 1 else f"{100 * x.mean():.2f}"


def ci_t(x):
    x = np.asarray(x, float); n = len(x)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return 100 * (x.mean() - h), 100 * (x.mean() + h)


def probs(d):
    return np.load(f"{d}/test_probs.npy"), np.load(f"{d}/test_y.npy")


def late(exp: str, members: str, unit: str = "seed", fname: str = "test_probs.npy") -> pd.DataFrame:
    """Late fusion: average posteriors of single-backbone runs of `exp` sharing the same seed/fold."""
    e = load_exp(exp)
    rows = []
    for u, g in e[e.branches.isin(list(members))].groupby(unit):
        if set(g.branches) != set(members):
            continue
        ps = [np.load(f"{d}/{fname}") for d in g.dir]
        y = np.load(f"{g.dir.iloc[0]}/test_y.npy")
        cls = json.load(open(f"{g.dir.iloc[0]}/metrics.json"))["classes"]
        m = compute_metrics(y, np.mean(ps, 0), cls)
        rows.append({unit: u, **{f"test_{k}": m[k] for k in METRICS}})
    return pd.DataFrame(rows)


def main():
    S = {}
    sel = json.load(open("outputs/selection.json"))
    S["selected_loss"] = sel["loss"]
    # ---- E1: loss selection table (validation) + test for completeness
    e1 = load_exp("fer_loss")
    t = e1.groupby(["branches", "loss"]).agg(val_f1=("val_macro_f1", ms), test_f1=("test_macro_f1", ms),
                                             test_acc=("test_acc", ms)).reset_index()
    t.to_csv(f"{T}/E1_loss_selection.csv", index=False)

    # ---- Main comparison on FER2013 official test: each model with its own validation-selected loss
    hp = json.load(open("outputs/selection.json"))["hp"]
    tune = pd.concat([load_exp("fer_tune"), load_exp("fer_tune_bd")])
    tt_ = tune.groupby(["branches", "loss", "d", "dropout", "wd", "branch_drop"]).val_macro_f1.agg(["mean", "std"]).reset_index()
    tt_.to_csv(f"{T}/E1b_tuning.csv", index=False)
    S["hp"] = hp
    em = load_exp("fer_main")
    main_rows, P = [], {}
    for b in ["m", "v", "r", "s", "mrsv", "mv"]:
        sub = em[em.branches == b].sort_values("seed")
        P[b] = {int(r.seed): probs(r.dir) for r in sub.itertuples()}
        h = hp.get(b, hp["mrsv"])
        rec = dict(model=BR[b], branches=b, loss=h["loss"], n_seeds=len(sub),
                   config=f"d={h['d']}, p={h['dropout']}, wd={h['wd']}" + (f", q={h['branch_drop']}" if len(b) > 1 else ""),
                   epochs=sub.epochs_run.mean(), head_params=None)
        for k in METRICS:
            rec[k] = ms(sub[f"test_{k}"]); rec[f"{k}_mean"] = sub[f"test_{k}"].mean(); rec[f"{k}_sd"] = sub[f"test_{k}"].std(ddof=1)
        rec["macro_f1_ci95"] = "[%.2f, %.2f]" % ci_t(sub.test_macro_f1)
        rec["acc_ci95"] = "[%.2f, %.2f]" % ci_t(sub.test_acc)
        main_rows.append(rec)
    # late-fusion ensemble baseline: average posteriors of the four single-backbone models (same seed)
    classes_ = json.load(open(glob.glob(f"{R}/fer_main/*/metrics.json")[0]))["classes"]
    ens = []
    for s in sorted(P["m"]):
        pr = np.mean([P[b][s][0] for b in "mvrs"], 0); y = P["m"][s][1]
        ens.append(compute_metrics(y, pr, json.load(open(glob.glob(f"{R}/fer_main/*/metrics.json")[0]))["classes"]))
        P.setdefault("late", {})[s] = (pr, y)
    rec = dict(model=BR["late"], branches="late", loss="per-branch", n_seeds=len(ens))
    for k in METRICS:
        v = [e[k] for e in ens]
        rec[k] = ms(v); rec[f"{k}_mean"] = np.mean(v); rec[f"{k}_sd"] = np.std(v, ddof=1)
    rec["macro_f1_ci95"] = "[%.2f, %.2f]" % ci_t([e["macro_f1"] for e in ens])
    rec["acc_ci95"] = "[%.2f, %.2f]" % ci_t([e["acc"] for e in ens])
    main_rows.insert(4, rec)
    ens2 = []
    for s_ in sorted(P["m"]):
        pr = np.mean([P[b][s_][0] for b in "mv"], 0); y = P["m"][s_][1]
        ens2.append(compute_metrics(y, pr, classes_)); P.setdefault("late_mv", {})[s_] = (pr, y)
    rec = dict(model=BR["late_mv"], branches="late_mv", loss="per-branch", n_seeds=len(ens2))
    for k in METRICS:
        v = [e[k] for e in ens2]
        rec[k] = ms(v); rec[f"{k}_mean"] = np.mean(v); rec[f"{k}_sd"] = np.std(v, ddof=1)
    rec["macro_f1_ci95"] = "[%.2f, %.2f]" % ci_t([e["macro_f1"] for e in ens2])
    rec["acc_ci95"] = "[%.2f, %.2f]" % ci_t([e["acc"] for e in ens2])
    main_rows.insert(5, rec)
    main = pd.DataFrame(main_rows)
    main.to_csv(f"{T}/T_main_fer2013.csv", index=False)
    S["main"] = main.to_dict(orient="records")

    # ---- Paired statistical comparison (seed-matched), Holm within each proposed-model family
    tests = []
    for A, bases in (("mrsv", ["m", "v", "r", "s", "late"]), ("mv", ["m", "v", "r", "s", "late", "late_mv", "mrsv"])):
        fam = []
        for b in bases:
            seeds = sorted(set(P[A]) & set(P[b]))
            f_a = [mf1(P[A][s][1], P[A][s][0].argmax(1)) for s in seeds]
            f_b = [mf1(P[b][s][1], P[b][s][0].argmax(1)) for s in seeds]
            a_a = [accuracy_score(P[A][s][1], P[A][s][0].argmax(1)) for s in seeds]
            a_b = [accuracy_score(P[b][s][1], P[b][s][0].argmax(1)) for s in seeds]
            tt = stats.ttest_rel(f_a, f_b)
            mc = [mcnemar(P[A][s][1], P[A][s][0], P[b][s][0]) for s in seeds]
            pb = paired_bootstrap_diff(P[A][0][1], P[A][0][0], P[b][0][0], mf1, 2000, 0)
            fam.append(dict(model=BR[A], baseline=BR[b], dF1_mean=100 * np.mean(np.subtract(f_a, f_b)),
                            dF1_sd=100 * np.std(np.subtract(f_a, f_b), ddof=1),
                            dAcc_mean=100 * np.mean(np.subtract(a_a, a_b)),
                            wins=int(np.sum(np.subtract(f_a, f_b) > 0)), n=len(f_a), t=tt.statistic, p_t=tt.pvalue,
                            dz=cohens_dz(f_a, f_b), mcnemar_sig_seeds=int(sum(m["p"] < 0.05 for m in mc)),
                            mcnemar_p_median=float(np.median([m["p"] for m in mc])),
                            boot_dF1_seed0=100 * pb["diff"], boot_lo=100 * pb["lo"], boot_hi=100 * pb["hi"]))
        ph = holm([f["p_t"] for f in fam])
        for f, p_ in zip(fam, ph):
            f["p_t_holm"] = p_
        tests += fam
    tdf = pd.DataFrame(tests)
    tdf.to_csv(f"{T}/T_stats_fer2013.csv", index=False)
    S["stats"] = tdf.to_dict(orient="records")

    # ---- per-class F1 (mean over seeds)
    classes = json.load(open(glob.glob(f"{R}/fer_main/*/metrics.json")[0]))["classes"]
    pc = []
    for b in ["m", "v", "r", "s", "late", "late_mv", "mrsv", "mv"]:
        for s in sorted(P[b]):
            m = compute_metrics(P[b][s][1], P[b][s][0], classes)
            for c in classes:
                pc.append(dict(model=b, seed=s, cls=c, **m["per_class"][c]))
    pcd = pd.DataFrame(pc)
    pcd.groupby(["model", "cls"])[["precision", "recall", "f1"]].agg(["mean", "std"]).to_csv(f"{T}/T_perclass_fer2013.csv")

    # ---- E2 subsets
    sub_rows = []
    allsub = pd.concat([load_exp("fer_subsets"), em]).drop_duplicates(["branches", "seed"])  # fer_main mv == fer_subsets mv
    eff = pd.read_csv("outputs/efficiency.csv", dtype={"branches": str})
    for b, g in allsub.groupby("branches"):
        e = eff[eff.branches == b].iloc[0]
        sub_rows.append(dict(branches=b, k=len(b), val_f1=g.val_macro_f1.mean(), val_f1_sd=g.val_macro_f1.std(ddof=1),
                             test_f1=g.test_macro_f1.mean(), test_f1_sd=g.test_macro_f1.std(ddof=1),
                             test_acc=g.test_acc.mean(), gflops=e.gflops, params_M=e.params_M,
                             lat_ms=e.lat_ms_b1, n=len(g)))
    sd = pd.DataFrame(sub_rows).sort_values(["k", "val_f1"])
    sd.to_csv(f"{T}/E2_subsets.csv", index=False)
    S["subsets"] = sd.to_dict(orient="records")
    # leave-one-branch-out contribution (validation and test)
    full = sd[sd.branches == "mrsv"].iloc[0]
    S["loo"] = {b: dict(val_drop=100 * (full.val_f1 - sd[sd.branches == "mrsv".replace(b, "")].iloc[0].val_f1),
                        test_drop=100 * (full.test_f1 - sd[sd.branches == "mrsv".replace(b, "")].iloc[0].test_f1))
                for b in "mrsv"}

    # ---- E3 fusion operators / projection
    fu = load_exp("fer_fusion"); npj = load_exp("fer_noproj"); npj["fusion"] = "concat (no projection)"
    fu = pd.concat([fu, npj])
    ft = fu.groupby("fusion").agg(val_f1=("val_macro_f1", ms), test_f1=("test_macro_f1", ms), test_acc=("test_acc", ms),
                                  test_ece=("test_ece", ms), n=("seed", "count")).reset_index()
    if "gate_m" in fu:
        g = fu[fu.fusion == "gated"][[c for c in fu if c.startswith("gate_")]].mean()
        S["gate_mean"] = g.to_dict()
    ft.to_csv(f"{T}/E3_fusion.csv", index=False)
    S["fusion"] = ft.to_dict(orient="records")

    # ---- E4 robustness
    rb = pd.concat([load_exp("fer_robust"), load_exp("fer_robust_mv")]); rs = load_exp("fer_robust_single")
    cols = [c for c in rb if c.startswith("test@") and c.endswith("_macro_f1")]
    rob = []
    for (b, bd), g in rb.groupby(["branches", "branch_drop"]):
        r = dict(model=("RMB-FF" if b == "mrsv" else "RMB-FF-MV") + f" (q={bd})", branches=b, q=bd,
                 clean=g.test_macro_f1.mean(), clean_sd=g.test_macro_f1.std(ddof=1), val=g.val_macro_f1.mean(), n=len(g))
        for c in cols:
            if c in g and g[c].notna().all():
                r[c.replace("test@", "").replace("_macro_f1", "")] = g[c].mean()
                r[c.replace("test@", "").replace("_macro_f1", "") + "_sd"] = g[c].std(ddof=1)
        rob.append(r)
    for b, g in rs.groupby("branches"):
        r = dict(model=BR[b], branches=b, q=0.0, clean=g.test_macro_f1.mean(), clean_sd=g.test_macro_f1.std(ddof=1),
                 val=g.val_macro_f1.mean(), n=len(g))
        for c in cols:
            if c in g and not c.startswith("test@missing"):
                r[c.replace("test@", "").replace("_macro_f1", "")] = g[c].mean()
                r[c.replace("test@", "").replace("_macro_f1", "") + "_sd"] = g[c].std(ddof=1)
        rob.append(r)
    variants = [c.replace("test@", "").replace("_macro_f1", "") for c in cols if "missing" not in c]
    for name, mem in (("Late fusion (4 posteriors)", "mrsv"), ("Late fusion (MobileNetV3-L + ViT-S)", "mv")):
        base = late("fer_robust_single", mem)
        r = dict(model=name, branches="late_" + mem, q=0.0, clean=base.test_macro_f1.mean(),
                 clean_sd=base.test_macro_f1.std(ddof=1), n=len(base))
        for v in variants:
            lv = late("fer_robust_single", mem, fname=f"test_probs@{v}.npy")
            if len(lv):
                r[v] = lv.test_macro_f1.mean(); r[v + "_sd"] = lv.test_macro_f1.std(ddof=1)
        if mem == "mrsv":  # missing branch: average the remaining three posteriors
            for b in mem:
                lm = late("fer_robust_single", mem.replace(b, ""))
                r[f"missing_{b}"] = lm.test_macro_f1.mean(); r[f"missing_{b}_sd"] = lm.test_macro_f1.std(ddof=1)
        rob.append(r)
    robd = pd.DataFrame(rob)
    robd.to_csv(f"{T}/E4_robustness.csv", index=False)
    S["robust"] = robd.to_dict(orient="records")

    # ---- E5 sensitivity (validation only)
    se = load_exp("fer_sensitivity")
    se.groupby(["d", "dropout"]).val_macro_f1.agg(["mean", "std"]).reset_index().to_csv(f"{T}/E5_sensitivity.csv", index=False)

    # ---- E6 dedup
    dd = load_exp("fer_dedup")
    for mem in ("mrsv", "mv"):
        lf = late("fer_dedup", mem).assign(branches="late_" + mem, n_test=dd.n_test.iloc[0])
        lf["val_macro_f1"] = np.nan
        dd = pd.concat([dd, lf])
    dd.groupby("branches").agg(test_f1=("test_macro_f1", ms), test_acc=("test_acc", ms), n_test=("n_test", "first")).reset_index() \
      .to_csv(f"{T}/E6_dedup.csv", index=False)
    S["dedup"] = dd.groupby("branches")[["test_macro_f1", "test_acc", "val_macro_f1"]].agg(["mean", "std"]).round(5) \
        .reset_index().to_dict(orient="records")
    S["dedup_n_test"] = int(dd.n_test.iloc[0]) if len(dd) else None

    # ---- E7 cross-dataset
    cx = load_exp("cross_fer2cmd")
    for mem in ("mrsv", "mv"):
        lf = late("cross_fer2cmd", mem).assign(branches="late_" + mem)
        lf["val_macro_f1"] = np.nan
        cx = pd.concat([cx, lf])
    cx.groupby("branches").agg(val_f1_fer=("val_macro_f1", ms), test_f1_cmd=("test_macro_f1", ms),
                               test_acc_cmd=("test_acc", ms), test_balacc=("test_bal_acc", ms)).reset_index() \
      .to_csv(f"{T}/E7_cross.csv", index=False)
    S["cross"] = cx.groupby("branches")[["test_macro_f1", "test_acc", "test_bal_acc"]].agg(["mean", "std"]).round(4).to_dict()

    # ---- E8 CMD group vs random CV
    cg, cr = load_exp("cmd_group"), load_exp("cmd_random")
    for mem in ("mrsv", "mv"):
        cg = pd.concat([cg, late("cmd_group", mem, unit="fold").assign(branches="late_" + mem)])
        cr = pd.concat([cr, late("cmd_random", mem, unit="fold").assign(branches="late_" + mem)])
    cg["protocol"], cr["protocol"] = "group-disjoint", "frame-level (leaky)"
    cc = pd.concat([cg, cr])
    cc.groupby(["protocol", "branches"]).agg(test_f1=("test_macro_f1", ms), test_acc=("test_acc", ms), folds=("fold", "count")) \
      .reset_index().to_csv(f"{T}/E8_cmd_cv.csv", index=False)
    S["cmd_cv"] = cc.groupby(["protocol", "branches"])[["test_macro_f1", "test_acc"]].agg(["mean", "std"]).round(4) \
        .reset_index().to_dict(orient="records")
    # paired across folds: fused vs best single (group protocol)
    S["cmd_group_pairs"] = {}
    for A in ("mrsv", "mv"):
        for b in ["m", "v", "r", "s", "late_" + A]:
            a = cg[cg.branches == A].sort_values("fold").test_macro_f1.values
            c = cg[cg.branches == b].sort_values("fold").test_macro_f1.values
            S["cmd_group_pairs"][f"{A}_vs_{b}"] = dict(diff=100 * float(np.mean(a - c)), wins=int((a > c).sum()),
                                                      p_wilcoxon=float(stats.wilcoxon(a, c).pvalue) if np.any(a != c) else 1.0)
    # cross-dataset paired over seeds
    S["cross_pairs"] = {}
    for A in ("mrsv", "mv"):
        for b in ["m", "v", "r", "s", "late_" + A]:
            a = cx[cx.branches == A].sort_values("seed").test_macro_f1.values
            c = cx[cx.branches == b].sort_values("seed").test_macro_f1.values
            tt = stats.ttest_rel(a, c)
            S["cross_pairs"][f"{A}_vs_{b}"] = dict(diff=100 * float(np.mean(a - c)), wins=int((a > c).sum()), p_t=float(tt.pvalue))
    # leakage inflation within CMD
    S["cmd_inflation"] = {b: 100 * float(cr[cr.branches == b].test_macro_f1.mean() - cg[cg.branches == b].test_macro_f1.mean())
                          for b in sorted(set(cg.branches))}

    # ---- efficiency
    S["efficiency"] = eff.to_dict(orient="records")
    def clean(o):
        if isinstance(o, dict):
            return {("|".join(map(str, k)) if isinstance(k, tuple) else k): clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    json.dump(clean(S), open("outputs/summary.json", "w"), indent=1, default=float)
    print(main[["model", "loss", "acc", "macro_f1", "bal_acc", "auroc_macro", "ece"]].to_string(index=False))
    print(tdf[["model", "baseline", "dF1_mean", "wins", "p_t", "p_t_holm", "dz", "mcnemar_sig_seeds"]].to_string(index=False))


if __name__ == "__main__":
    main()
