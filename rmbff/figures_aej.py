"""Figures of the paper, drawn from saved tables and run outputs.

    RMBFF_TAG=_ft RMBFF_RUNS=outputs/runs_ft python -m rmbff.figures_aej   # Figs. 2-7 (fine-tuned)
    python -m rmbff.figures_aej                                           # frozen versions of Figs. 4-7
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style

style.apply()
P = style.PAL
F = "figures"
TAG = os.environ.get("RMBFF_TAG", "")   # "_ft": Figs. 4-7 from the fine-tuned runs (paper); "": frozen versions
FR = F if TAG else f"{F}/frozen"


def fig_datasets():
    fer = pd.read_csv("outputs/manifests/fer2013.csv"); cmd = pd.read_csv("outputs/manifests/cmd_groupfolds.csv")
    CL = style.CLASS_LABEL
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2), gridspec_kw=dict(width_ratios=[1.25, 1, 1.1]))
    cls = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    x = np.arange(len(cls)); w = 0.27
    for i, (sp, c, lab) in enumerate([("train", "#4C72B0", "Training"), ("val", "#DD8452", "Public Test (val.)"),
                                      ("test", "#55A868", "Private Test (test)")]):
        n = fer[fer.split == sp].label_name.value_counts().reindex(cls).values
        axs[0].bar(x + (i - 1) * w, n, w, color=c, label=lab)
    axs[0].set_xticks(x, [CL[c] for c in cls], rotation=35, ha="right"); axs[0].set_yscale("log"); axs[0].set_ylim(30, 2e5)
    axs[0].set_ylabel("Images (log scale)"); axs[0].set_title("(a) FER2013 (n = 35,887)")
    axs[0].legend(loc="upper center", ncol=3, fontsize=5.5, columnspacing=0.6, handlelength=1)
    ccls = ["anger", "contempt", "disgust", "fear", "happy", "sad", "surprise"]
    axs[1].bar(np.arange(7), cmd.label_name.value_counts().reindex(ccls).values, color="#8172B2")
    axs[1].set_xticks(np.arange(7), [CL[c] for c in ccls], rotation=35, ha="right")
    axs[1].set_ylabel("Frames"); axs[1].set_title(f"(b) DFERCD by class (n = {len(cmd):,})")
    g = cmd.groupby("group").size().sort_values(ascending=False)
    axs[2].bar(np.arange(len(g)), g.values, color="#8172B2")
    axs[2].set_xticks(np.arange(len(g)), [s.replace("cmd_", "") for s in g.index], rotation=90, fontsize=5.5)
    axs[2].set_ylabel("Frames"); axs[2].set_xlabel("Recording group (file-name prefix)")
    axs[2].set_title(f"(c) DFERCD by recording group (G = {len(g)})")
    fig.tight_layout(); style.save(fig, f"{F}/aej_fig2_datasets"); plt.close(fig)


def fig_protocol():
    e9 = pd.read_csv("tables/E9_protocol.csv")
    cv = pd.read_csv("tables/E8_cmd_cv.csv", dtype={"branches": str})
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw=dict(width_ratios=[1.3, 1]))
    order = ["official", "aug_after", "random", "aug_before"]
    lab = {"official": "Official split\n(this study)", "aug_after": "Random 70/30,\naug. after split",
           "random": "Random 70/30\n(image level)", "aug_before": "Random 70/30,\naug. before split"}
    models = [("v", "ViT-S/16", P["v"]), ("mv", "RMB-FF-MV", P["accent"])]
    x = np.arange(len(order)); w = 0.36
    for i, (mk, nm, c) in enumerate(models):
        g = e9[e9.model == mk].groupby("protocol").acc
        mu = [100 * g.mean()[o] for o in order]; sd = [100 * g.std()[o] for o in order]
        axs[0].bar(x + (i - 0.5) * w, mu, w, yerr=sd, color=c, label=nm, error_kw=dict(lw=0.6, capsize=1.5))
        for xi, m in zip(x + (i - 0.5) * w, mu):
            axs[0].text(xi, m + 2.6, f"{m:.1f}", ha="center", fontsize=5.8)
    axs[0].set_xticks(x, [lab[o] for o in order], fontsize=6.3); axs[0].set_ylabel("Test accuracy (%)")
    axs[0].set_title("(a) FER2013: the split protocol changes accuracy"); axs[0].legend(loc="upper left", fontsize=6)
    axs[0].set_ylim(0, 105)
    mm = ["m", "v", "r", "s", "mrsv", "mv"]
    nmap = {"m": "MobileNetV3-L", "v": "ViT-S/16", "r": "RegNetY", "s": "SE-ResNet-50", "mrsv": "RMB-FF", "mv": "RMB-FF-MV"}
    f = lambda p, b: float(cv[(cv.protocol == p) & (cv.branches == b)].test_acc.iloc[0].split("±")[0])
    x2 = np.arange(len(mm))
    axs[1].bar(x2 - 0.2, [f("frame-level (leaky)", b) for b in mm], 0.4, color="#C44E52", label="Frame-level split")
    axs[1].bar(x2 + 0.2, [f("group-disjoint", b) for b in mm], 0.4, color="#4C72B0", label="Group-disjoint split")
    axs[1].set_xticks(x2, [nmap[b] for b in mm], rotation=30, ha="right"); axs[1].set_ylabel("Test accuracy (%)")
    axs[1].set_title("(b) DFERCD, 5-fold CV"); axs[1].legend(fontsize=6, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    axs[1].set_ylim(0, 118)
    fig.tight_layout(); style.save(fig, f"{F}/aej_fig3_protocol"); plt.close(fig)


SHORT = {"Robot only, always act (top-1)": "Robot, always act", "Cloud only, always act (top-1)": "Cloud, always act",
         "Robot only (MobileNetV3-L)": "Robot only", "Edge only (RMB-FF-MV)": "Edge only (MV)",
         "Cloud only (RMB-FF, 4 branches)": "Cloud only (RMB-FF)", "Cloud only (late fusion, 4 models)": "Cloud only (late fusion)",
         "Robot→edge, fixed softmax threshold 0.9": "R→E, fixed 0.9",
         "Robot→edge, per-tier calibration": "R→E, per-tier calib.",
         "Robot→edge, joint calibration (decisiveness-first)": "R→E, joint (decisive)",
         "Robot→edge, joint calibration (cost-aware)": "R→E, joint (cost-aware)",
         "Robot→edge→cloud, joint calibration (decisiveness-first)": "R→E→C, joint (decisive)",
         "Robot→edge→cloud, joint calibration (cost-aware)": "R→E→C, joint (cost-aware)"}


def fig_risk_validity():
    raw = pd.read_csv(f"tables/R1_cascade_raw{TAG}.csv")
    names = [n for n in SHORT if "always" not in n]
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=False)
    for ax, eps in zip(axs, [0.05, 0.10]):
        data = [100 * raw[(raw.eps == eps) & (raw.name == n)].risk.values for n in names]
        bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", lw=0.8), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
                        capprops=dict(lw=0.6))
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor("#F2B8B5" if ("per-tier" in n or "fixed" in n) else ("#FDE3B4" if "joint" in n else "#D9E4F0"))
        ax.axvline(100 * eps, color="#B22222", ls="--", lw=0.8)
        ax.set_yticks(range(1, len(names) + 1), [SHORT[n] for n in names] if eps == 0.05 else [""] * len(names))
        ax.set_xlabel("Wrong-action rate on held-out half (%)")
        ax.set_title(f"({'a' if eps == 0.05 else 'b'}) target ε = {int(100 * eps)}%  (red line)")
    fig.tight_layout(); style.save(fig, f"{FR}/aej_fig4_risk_validity"); plt.close(fig)


def fig_frontier():
    t = pd.read_csv(f"tables/R1_cascade{TAG}.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.7))
    mk = {"Robot only (MobileNetV3-L)": ("o", P["m"]), "Edge only (RMB-FF-MV)": ("o", P["accent"]),
          "Cloud only (RMB-FF, 4 branches)": ("o", "#000000"), "Cloud only (late fusion, 4 models)": ("o", P["grey"]),
          "Robot→edge, joint calibration (decisiveness-first)": ("D", P["v"]),
          "Robot→edge, joint calibration (cost-aware)": ("s", P["v"]),
          "Robot→edge→cloud, joint calibration (decisiveness-first)": ("D", P["r"]),
          "Robot→edge→cloud, joint calibration (cost-aware)": ("s", P["r"])}
    for ax, eps in zip(axs, [0.05, 0.10]):
        s = t[t.eps == eps].set_index("name")
        for n, (m, c) in mk.items():
            r = s.loc[n]
            ax.errorbar(r.cost, 100 * r.decisive, yerr=100 * r.decisive_sd, fmt=m, color=c, ms=4.5, lw=0.6, capsize=1.5,
                        label=SHORT[n])
        ax.set_xscale("symlog", linthresh=1); ax.set_xticks([0.43, 1, 5, 10, 20], ["0.43", "1", "5", "10", "20"])
        ax.set_xlabel("Mean compute per face (GFLOPs)"); ax.set_ylabel("Decisive actions (%)")
        ax.set_title(f"({'a' if eps == 0.05 else 'b'}) ε = {int(100 * eps)}%, all points certified")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1)); style.save(fig, f"{FR}/aej_fig5_frontier"); plt.close(fig)


def fig_failure():
    t = pd.read_csv(f"tables/R2_failure{TAG}.csv")
    order = ["healthy", "ViT branch down, thresholds kept", "ViT branch down, recalibrated"]
    lab = ["Healthy", "ViT branch down,\nthresholds kept", "ViT branch down,\nrecalibrated"]
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.4))
    for ax, eps in zip(axs, [0.05, 0.10]):
        s = t[t.eps == eps].set_index("name")
        x = np.arange(3)
        for i, (q, c) in enumerate([(0.0, "#999999"), (0.25, P["s"])]):
            v = [100 * s.loc[f"q={q}: {o}"].risk for o in order]
            ok = [100 * s.loc[f"q={q}: {o}"].ok for o in order]
            ax.bar(x + (i - 0.5) * 0.36, v, 0.36, color=c, label=f"edge model trained with q = {q}")
            for xi, vi, oi in zip(x + (i - 0.5) * 0.36, v, ok):
                ax.text(xi, vi + 0.3, f"{oi:.0f}%", ha="center", fontsize=5.6)
        ax.axhline(100 * eps, color="#B22222", ls="--", lw=0.8); ax.set_ylim(0, 100 * eps * 1.8)
        ax.set_xticks(x, lab, fontsize=6.5); ax.set_ylabel("Wrong-action rate (%)")
        ax.set_title(f"({'a' if eps == 0.05 else 'b'}) ε = {int(100 * eps)}%; labels: splits meeting ε")
    axs[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout(); style.save(fig, f"{FR}/aej_fig6_failure"); plt.close(fig)


def fig_failure_ft(eps=0.10):
    """Fine-tuned edge models: (a) wrong-action rate, (b) decisiveness at eps, (c) accuracy with/without ViT."""
    import glob
    t = pd.read_csv(f"tables/R2_failure{TAG}.csv"); s = t[t.eps == eps].set_index("name")
    order = ["healthy", "ViT branch down, thresholds kept", "ViT branch down, recalibrated"]
    lab = ["Healthy", "ViT down,\nthresholds kept", "ViT down,\nrecalibrated"]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.4), gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.8]))
    x = np.arange(3); cols = [(0.0, "#999999"), (0.25, P["s"])]
    for i, (q, c) in enumerate(cols):
        rk = [100 * s.loc[f"q={q}: {o}"].risk for o in order]; ok = [100 * s.loc[f"q={q}: {o}"].ok for o in order]
        dc = [100 * s.loc[f"q={q}: {o}"].decisive for o in order]
        xx = x + (i - 0.5) * 0.36
        axs[0].bar(xx, rk, 0.36, color=c, label=f"edge model trained with q = {q}")
        for xi, vi, oi in zip(xx, rk, ok):
            axs[0].text(xi, vi + 0.2, f"{oi:.0f}%", ha="center", fontsize=5.4)
        axs[1].bar(xx, dc, 0.36, color=c)
    axs[0].axhline(100 * eps, color="#B22222", ls="--", lw=0.8); axs[0].set_ylim(0, 100 * eps * 1.5)
    axs[0].set_ylabel("Wrong-action rate (%)"); axs[0].set_title(f"(a) Risk, ε = {int(100 * eps)}%")
    axs[1].set_ylabel("Decisive actions (%)"); axs[1].set_title(f"(b) Decisiveness, ε = {int(100 * eps)}%")
    for ax in axs[:2]:
        ax.set_xticks(x, lab, fontsize=6.2)
    axs[0].legend(fontsize=5.6, loc="upper left")
    R = os.environ.get("RMBFF_RUNS", "outputs/runs")
    for i, (q, c) in enumerate(cols):
        h = [json.load(open(f))["test"]["acc"] for f in sorted(glob.glob(f"{R}/fer_robust_mv/branches-mv_bd-{q}_seed-*/metrics.json"))]
        n = [json.load(open(f))["test"]["acc"] for f in sorted(glob.glob(f"{R}/fer_robust_mv/branches-mv_bd-{q}_seed-*/metrics_noV.json"))]
        axs[2].bar(np.array([0, 1]) + (i - 0.5) * 0.36, [100 * np.mean(h), 100 * np.mean(n)], 0.36,
                   yerr=[100 * np.std(h, ddof=1), 100 * np.std(n, ddof=1)], color=c, capsize=1.5, error_kw=dict(lw=0.6))
    axs[2].set_xticks([0, 1], ["Healthy", "ViT\nremoved"], fontsize=6.2); axs[2].set_ylabel("FER2013 test accuracy (%)")
    axs[2].set_title("(c) Edge-model accuracy")
    fig.tight_layout(); style.save(fig, f"{FR}/aej_fig6_failure"); plt.close(fig)


def fig_adapt():
    d = pd.read_csv(f"tables/R3_adapt_raw{TAG}.csv")
    d = d[(d.model == "mrsv") & (d.eps == 0.10)]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.5), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for ax, st, tt in zip(axs[:2], ["FER2013 -> DFERCD (cross-dataset)", "DFERCD group-disjoint folds"],
                          ["(a) FER2013 → DFERCD", "(b) DFERCD, unseen groups"]):
        s = d[d.setting == st]
        for meth, c, mk in [("static (calibrated on source)", "#C44E52", "o"), ("person-adaptive (gamma=0.02)", P["v"], "s")]:
            q = s[s.method == meth]
            ax.scatter(q.n, 100 * q.risk, s=8, color=c, marker=mk, alpha=0.75,
                       label="Static calibration" if "static" in meth else "Person-adaptive (γ = 0.02)")
        ax.axhline(10, color="#B22222", ls="--", lw=0.8); ax.set_xscale("log")
        ax.set_xlabel("Frames in the person's stream"); ax.set_ylabel("Wrong-action rate (%)"); ax.set_title(tt)
    axs[0].legend(fontsize=5.8, loc="upper right")
    # threshold trajectory for one long stream
    from .riskcontrol import online_adapt, to_actions
    from .analyze import load_exp
    from .riskexp import cmd_streams
    e = load_exp("cross_fer2cmd"); r = e[(e.branches == "mrsv") & (e.seed == 0)].iloc[0]
    cl = json.load(open(f"{r.dir}/metrics.json"))["classes"]
    At, yat, _ = to_actions(np.load(f"{r.dir}/test_probs.npy"), np.load(f"{r.dir}/test_y.npy"), cl)
    rows = np.load(f"{r.dir}/test_rows.npy"); cmd = pd.read_csv("outputs/manifests/cmd_groupfolds.csv")
    sub = cmd.iloc[rows].reset_index(drop=True).assign(pos=np.arange(len(rows)))
    streams = cmd_streams(sub, np.random.default_rng(0))
    g = max(streams, key=lambda k: len(streams[k])); order = streams[g]
    t0 = float(pd.read_csv(f"tables/R3_adapt_raw{TAG}.csv").query("setting.str.startswith('FER') and model=='mrsv' and eps==0.1 and unit==0").t0.iloc[0])
    conf, pred, ya = At[order].max(1), At[order].argmax(1), yat[order]
    t = t0; ts = []; L = []
    for i in range(len(ya)):
        ts.append(t); a_ = conf[i] >= t; l_ = float(a_ and pred[i] != ya[i]); L.append(l_)
        t = t + 0.02 * (l_ - 0.10)
    L = np.array(L); stat = (conf >= t0) & (pred != ya)
    k = np.arange(1, len(L) + 1)
    axs[2].plot(k, 100 * np.cumsum(stat) / k, color="#C44E52", lw=1, label="Static: running wrong-action rate")
    axs[2].plot(k, 100 * np.cumsum(L) / k, color=P["v"], lw=1, label="Adaptive: running wrong-action rate")
    axs[2].axhline(10, color="#B22222", ls="--", lw=0.8)
    ax2 = axs[2].twinx(); ax2.plot(k, ts, color="#777777", lw=0.7, ls=":", label="Adaptive threshold t")
    ax2.set_ylabel("Threshold t", fontsize=7); ax2.spines["right"].set_visible(True)
    axs[2].set_xlabel(f"Encounter (group {g.replace('cmd_', '')}, {len(L):,} frames)"); axs[2].set_ylabel("Wrong-action rate (%)")
    axs[2].set_title("(c) One stream over time"); axs[2].legend(fontsize=5.4, loc="lower right")
    fig.tight_layout(); style.save(fig, f"{FR}/aej_fig7_adapt"); plt.close(fig)


def main():
    os.makedirs(FR, exist_ok=True)
    for f in (fig_datasets, fig_protocol, fig_risk_validity, fig_frontier, fig_failure_ft if TAG else fig_failure,
              fig_adapt):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, repr(e))


if __name__ == "__main__":
    main()
