"""Fig. 1 of the paper: risk-controlled robot-edge-cloud RMB-FF (vector, programmatic)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import style

style.apply()
P = style.PAL


def box(ax, x, y, w, h, lines, fc="white", ec="#333333", lw=0.8, bold_first=True, fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", fc=fc, ec=ec, lw=lw))
    n = len(lines)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 + (n - 1) * 0.021 - i * 0.042, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if (i == 0 and bold_first) else "normal")


def arrow(ax, x0, y0, x1, y1, c="#333333", ls="-", lw=0.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=lw, color=c,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def main(out="figures/aej_fig1_framework"):
    fig = plt.figure(figsize=(7.16, 3.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    acc = "#FDF1DC"
    # calibration band
    box(ax, 0.17, 0.82, 0.80, 0.12, ["Joint risk calibration (Learn-then-Test, fixed-sequence)",
                                     "thresholds t = (t1, t2, t3) certified so that P(wrong-action rate ≤ ε) ≥ 1 − δ"],
        fc="#E8EEF4", ec=P["m"], lw=1.1)
    # input
    box(ax, 0.01, 0.47, 0.12, 0.17, ["Face frame", "IoT / robot", "camera"])
    xs = [0.17, 0.445, 0.72]; w = 0.25
    tiers = [["Tier 1: robot", "MobileNetV3-L head", "0.43 GFLOPs"],
             ["Tier 2: edge gateway", "RMB-FF-MV fusion (M + V)", "+8.48 GFLOPs"],
             ["Tier 3: cloud", "RMB-FF 4-branch fusion", "+11.41 GFLOPs"]]
    for x, t in zip(xs, tiers):
        box(ax, x, 0.47, w, 0.17, t)
        arrow(ax, x + w / 2, 0.82, x + w / 2, 0.64, c=P["m"], ls=(0, (3, 2)))
    arrow(ax, 0.13, 0.555, 0.17, 0.555)
    arrow(ax, 0.42, 0.555, 0.445, 0.555); arrow(ax, 0.695, 0.555, 0.72, 0.555)
    ax.text(0.4325, 0.665, "escalate if max p(a|x) < t1", fontsize=5.8, ha="center", color="#555555")
    ax.text(0.7075, 0.665, "escalate if < t2", fontsize=5.8, ha="center", color="#555555")
    # action layer
    box(ax, 0.17, 0.22, 0.80, 0.13, ["Action-level decision (emotions → robot actions)",
                                     "engage: happy, neutral · comfort: sad, fear · de-escalate: anger, disgust, contempt · check-in: surprise"],
        fc=acc, ec="#B8860B", lw=1.0)
    for x, lab in zip(xs, ["act if max p(a|x) ≥ t1", "act if ≥ t2", "act if ≥ t3, else ask / defer"]):
        arrow(ax, x + w / 2, 0.47, x + w / 2, 0.35)
        ax.text(x + w / 2 + 0.01, 0.405, lab, fontsize=5.8, color="#555555")
    # adaptation + failure notes
    box(ax, 0.17, 0.03, 0.39, 0.13, ["Person-adaptive update (online)", "t ← t + γ (L − ε) after each encounter"],
        fc="white", ec=P["v"], lw=1.1)
    box(ax, 0.58, 0.03, 0.39, 0.13, ["Fault tolerance", "branch dropout (q) + recalibration per tier configuration"],
        fc="white", ec=P["s"], lw=1.1)
    arrow(ax, 0.365, 0.22, 0.365, 0.16, c=P["v"])
    ax.text(0.375, 0.185, "feedback L = 1{acted and wrong}", fontsize=5.8, color="#555555")
    style.save(fig, out)


if __name__ == "__main__":
    main()
