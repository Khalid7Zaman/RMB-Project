"""Evaluation metrics and statistical procedures."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)


def ece_score(prob: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    conf, pred = prob.max(1), prob.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def compute_metrics(y: np.ndarray, prob: np.ndarray, classes: list[str]) -> dict:
    C = len(classes)
    pred = prob.argmax(1)
    yoh = np.eye(C)[y]
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(C), zero_division=0)
    present = yoh.sum(0) > 0
    out = dict(
        acc=accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro", labels=range(C), zero_division=0),
        weighted_f1=f1_score(y, pred, average="weighted", labels=range(C), zero_division=0),
        bal_acc=balanced_accuracy_score(y, pred),
        auroc_macro=float(np.mean([roc_auc_score(yoh[:, c], prob[:, c]) for c in range(C) if present[c]])),
        auprc_macro=float(np.mean([average_precision_score(yoh[:, c], prob[:, c]) for c in range(C) if present[c]])),
        nll=float(-np.mean(np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)))),
        brier=float(np.mean(np.sum((prob - yoh) ** 2, 1))),
        ece=ece_score(prob, y),
        n=int(len(y)),
    )
    out["per_class"] = {c: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i]))
                        for i, c in enumerate(classes)}
    out["confusion"] = confusion_matrix(y, pred, labels=range(C)).tolist()
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in out.items()}


def bootstrap_ci(y, prob, fn, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI over test samples for a scalar metric fn(y, pred)."""
    rng = np.random.default_rng(seed)
    pred = prob.argmax(1)
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = fn(y[i], pred[i])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(y, pa, pb, fn, n_boot: int = 2000, seed: int = 0):
    """Bootstrap distribution of fn(A) - fn(B) on the same resampled test items."""
    rng = np.random.default_rng(seed)
    a, b = pa.argmax(1), pb.argmax(1)
    n = len(y)
    d = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, n, n)
        d[k] = fn(y[i], a[i]) - fn(y[i], b[i])
    obs = fn(y, a) - fn(y, b)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(diff=float(obs), lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)),
                p_boot=float(min(1.0, p)))


def mcnemar(y, pa, pb) -> dict:
    """Exact (binomial) McNemar test on per-sample correctness (Dietterich, 1998)."""
    ca, cb = pa.argmax(1) == y, pb.argmax(1) == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return dict(a_only=n01, b_only=n10, p=float(p))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * pvals[i]))
        adj[i] = run
    return adj.tolist()


def cohens_dz(a, b) -> float:
    """Paired effect size d_z = mean(diff) / sd(diff) across seeds/folds."""
    d = np.asarray(a) - np.asarray(b)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("inf") * np.sign(d.mean())
