r"""Class-imbalance-aware objectives.

Let n_c be the number of training samples of class c, N = sum_c n_c, and
pi_c = n_c / N the empirical prior.

ce      : standard cross-entropy.
wce     : inverse-frequency weighted CE, w_c = N / (C n_c).
cb_focal: class-balanced focal loss; w_c = (1-beta)/(1-beta^{n_c}) (effective
          number of samples), normalised to sum C, focal exponent gamma.
la      : logit-adjusted CE: CE(s + tau log pi, y). At inference the raw
          logits s are used, which corresponds to the balanced-error Bayes rule.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImbalanceLoss(nn.Module):
    def __init__(self, kind: str, counts, beta: float = 0.999, gamma: float = 1.0, tau: float = 1.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        counts = np.asarray(counts, dtype=np.float64)
        C, N = len(counts), counts.sum()
        self.kind, self.gamma, self.ls = kind, gamma, label_smoothing
        if kind == "wce":
            w = N / (C * counts)
        elif kind == "cb_focal":
            eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
            w = 1.0 / eff
            w = w / w.sum() * C
        else:
            w = np.ones(C)
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32))
        self.register_buffer("log_prior", torch.tensor(np.log(counts / N), dtype=torch.float32))
        self.tau = tau

    def forward(self, logits, y):
        if self.kind == "la":
            return F.cross_entropy(logits + self.tau * self.log_prior, y, label_smoothing=self.ls)
        if self.kind == "cb_focal":
            logp = F.log_softmax(logits, 1)
            lp_y = logp.gather(1, y[:, None]).squeeze(1)
            focal = (1 - lp_y.exp()).pow(self.gamma)
            return (-(self.w[y]) * focal * lp_y).mean()
        weight = self.w if self.kind == "wce" else None
        return F.cross_entropy(logits, y, weight=weight, label_smoothing=self.ls)
