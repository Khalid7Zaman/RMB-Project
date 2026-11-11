r"""Risk-controlled robot-edge-cloud cascade (RMB-FF v2), evaluated on saved posteriors.

Notation
    K tiers k = 1..K with posteriors p_k(c|x) over emotion classes c.
    A : map from emotion classes to robot actions a = A(c); p_k(a|x) = sum_{c: A(c)=a} p_k(c|x).
    Thresholds t = (t_1..t_K). Tier k ACTS with argmax_a p_k(a|x) if max_a p_k(a|x) >= t_k; otherwise the
    frame escalates to tier k+1; after tier K the robot ASKS / DEFERS (no emotion-driven action).
    Loss  L(x, y; t) = 1{robot acts and the chosen action != A(y)}       (wrong emotion-driven action)
    Risk  R(t) = E[L].  Cost C(t) = E[cumulative GFLOPs].  Decisiveness D(t) = P(robot acts).

Calibration: Learn-then-Test (Angelopoulos et al.) with Hoeffding-Bentkus p-values and Bonferroni
correction over a finite grid. Every selected t satisfies P(R(t) <= eps) >= 1 - delta for exchangeable data.
Online person adaptation: per-person threshold updated after each encounter with feedback,
t <- t + gamma * (L - eps)  (adaptive conformal inference applied to the action threshold), which bounds
the long-run wrong-action rate of each person's stream at eps without exchangeability.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

ACTIONS = {"engage": ["happy", "neutral"], "comfort": ["sad", "fear"],
           "de-escalate": ["anger", "disgust", "contempt"], "check-in": ["surprise"]}
ACTION_NAMES = list(ACTIONS)


def action_matrix(classes):
    M = np.zeros((len(classes), len(ACTION_NAMES)))
    for i, c in enumerate(classes):
        for j, a in enumerate(ACTION_NAMES):
            if c in ACTIONS[a]:
                M[i, j] = 1
    keep = M.sum(0) > 0
    return M[:, keep], [a for a, k in zip(ACTION_NAMES, keep) if k]


def to_actions(P, y, classes):
    M, names = action_matrix(classes)
    return P @ M, (M.argmax(1))[y], names


def hb_pvalue(rhat, n, eps):
    """Hoeffding-Bentkus p-value for H0: R > eps given empirical risk rhat over n samples."""
    if rhat >= eps:
        return 1.0
    a = rhat
    h = (a * np.log(a / eps) if a > 0 else 0) + (1 - a) * np.log((1 - a) / (1 - eps))
    p_h = np.exp(-n * h)
    p_b = np.e * stats.binom.cdf(np.ceil(n * rhat), n, eps)
    return float(min(p_h, p_b, 1.0))


def cascade_outcome(PA, ya, t):
    """PA: list of K arrays (n, A) action posteriors; t: K thresholds. Returns acted, wrong, tier_reached."""
    n = len(ya)
    acted = np.zeros(n, bool); wrong = np.zeros(n, bool); tier = np.full(n, len(PA) - 1)
    open_ = np.ones(n, bool)
    for k, (P, tk) in enumerate(zip(PA, t)):
        conf = P.max(1); pred = P.argmax(1)
        go = open_ & (conf >= tk)
        acted[go] = True; wrong[go] = pred[go] != ya[go]; tier[go] = k
        open_ &= ~go
        if k < len(PA) - 1:
            tier[open_] = k + 1
    return acted, wrong, tier


def evaluate(PA, ya, t, cost_cum):
    acted, wrong, tier = cascade_outcome(PA, ya, t)
    return dict(risk=wrong.mean(), decisive=acted.mean(), cost=np.asarray(cost_cum)[tier].mean(),
                cond_err=(wrong[acted].mean() if acted.any() else 0.0),
                tier_share=[float((tier == k).mean()) for k in range(len(PA))])


def ltt_calibrate(PA, ya, grid, cost_cum, eps, delta, cost_weight=0.05):
    """Return the valid threshold vector maximising decisiveness - cost_weight * cost / max cost."""
    n = len(ya); m = len(grid)
    best, best_obj = None, -np.inf
    for t in grid:
        acted, wrong, tier = cascade_outcome(PA, ya, t)
        if hb_pvalue(wrong.mean(), n, eps) <= delta / m:
            obj = acted.mean() - cost_weight * np.asarray(cost_cum)[tier].mean() / cost_cum[-1]
            if obj > best_obj:
                best, best_obj = t, obj
    if best is None:   # nothing certifiable -> never act (risk 0)
        best = tuple([1.01] * len(PA))
    return best


def grid_for(K, values):
    return list(itertools.product(*[values] * K))


def online_adapt(conf, pred, ya, t0, eps, gamma):
    """Per-stream adaptive threshold; returns acted, wrong arrays in stream order."""
    t = t0; acted = np.zeros(len(ya), bool); wrong = np.zeros(len(ya), bool)
    for i in range(len(ya)):
        if conf[i] >= t:
            acted[i] = True; wrong[i] = pred[i] != ya[i]
        t = t + gamma * (float(wrong[i]) - eps)
    return acted, wrong


def outcomes_grid(PA, ya, grid, cost_cum):
    """Vectorised risk / decisiveness / cost for every threshold vector in grid."""
    res = np.empty((len(grid), 3))
    for i, t in enumerate(grid):
        acted, wrong, tier = cascade_outcome(PA, ya, t)
        res[i] = (wrong.mean(), acted.mean(), np.asarray(cost_cum)[tier].mean())
    return res


def ltt_fst(PA, ya, grid, cost_cum, eps, delta, cost_weight, rng):
    """Learn-then-Test with a data split and fixed-sequence testing (no Bonferroni penalty).

    Half A of the calibration data orders the candidate thresholds from safest (lowest empirical risk)
    to riskiest, keeping those with empirical risk <= eps; half B tests them in that order with
    Hoeffding-Bentkus p-values at level delta and stops at the first non-rejection. Among the validated
    prefix the candidate with the best objective (decisiveness - cost_weight * cost / max cost) on half B
    is returned. FWER <= delta, hence P(R(t) <= eps) >= 1 - delta.
    """
    n = len(ya); perm = rng.permutation(n); ia, ib = perm[: n // 2], perm[n // 2:]
    ra = outcomes_grid([P[ia] for P in PA], ya[ia], grid, cost_cum)
    cand = np.where(ra[:, 0] <= eps)[0]
    cand = cand[np.argsort(ra[cand, 0], kind="stable")]
    best, best_obj = tuple([1.01] * len(PA)), -np.inf
    for i in cand:
        acted, wrong, tier = cascade_outcome([P[ib] for P in PA], ya[ib], grid[i])
        if hb_pvalue(wrong.mean(), len(ib), eps) > delta:
            break
        obj = acted.mean() - cost_weight * np.asarray(cost_cum)[tier].mean() / cost_cum[-1]
        if obj > best_obj:
            best, best_obj = grid[i], obj
    return best
