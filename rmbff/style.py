"""Shared figure style: Times-compatible serif, restrained colour-blind-safe palette."""
import matplotlib as mpl

# Times New Roman is used when installed; otherwise metric-compatible clones.
SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIX Two Text", "DejaVu Serif"]

# Okabe-Ito derived palette (colour-blind safe)
PAL = dict(m="#0072B2", v="#009E73", r="#CC79A7", s="#D55E00", fused="#000000", grey="#7F7F7F",
           light="#E8EEF4", accent="#E69F00")
BRANCH_LABEL = {"m": "MobileNetV3-L", "v": "ViT-S/16", "r": "RegNetY-1.6GF", "s": "SE-ResNet-50"}
CLASS_LABEL = {"anger": "Anger", "contempt": "Contempt", "disgust": "Disgust", "fear": "Fear", "happy": "Happy",
               "neutral": "Neutral", "sad": "Sad", "surprise": "Surprise"}


def apply():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
    })


def save(fig, path_stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{path_stem}.{ext}")
