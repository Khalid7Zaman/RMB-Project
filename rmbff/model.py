r"""RMB-FF fusion model.

Notation (Section III of the manuscript)
    x            input face image
    f_b(x)       pooled embedding of backbone b in B = {m, v, r, s}, dim d_b
    z_b          projected branch embedding  z_b = GELU(W_b LN(f_b(x)) + c_b), dim d
    z            fused representation
    p(y|x)       class posterior from the class-imbalance-aware classifier

Fusion operators
    concat  : z = P([z_m ; z_v ; z_r ; z_s])             (proposed, Fig. 1)
    gated   : z = sum_b a_b(x) z_b,  a = softmax(g([z_b]))  (ablation)
    mean    : z = mean_b z_b                              (ablation)
Branch dropout (training only) zeroes each z_b with probability q and
rescales the rest, to make the fused head robust to a degraded/missing branch.

The head can operate on cached frozen features (``RMBFFHead``) or on images
with trainable backbones (``RMBFF``, used for end-to-end fine-tuning).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchProjector(nn.Module):
    def __init__(self, d_in: int, d: int, p_drop: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(p_drop))

    def forward(self, x):
        return self.net(x)


class RMBFFHead(nn.Module):
    def __init__(self, dims: dict[str, int], n_classes: int, d: int = 256, fusion: str = "concat",
                 p_drop: float = 0.3, branch_drop: float = 0.0, proj: bool = True):
        super().__init__()
        self.keys = sorted(dims)
        self.fusion, self.branch_drop, self.proj = fusion, branch_drop, proj
        if proj:
            self.projectors = nn.ModuleDict({k: BranchProjector(dims[k], d, p_drop) for k in self.keys})
            widths = {k: d for k in self.keys}
        else:  # ablation: no per-branch projection (raw features, LayerNorm only)
            self.projectors = nn.ModuleDict({k: nn.LayerNorm(dims[k]) for k in self.keys})
            widths = dict(dims)
        if fusion == "concat":
            self.fuse = nn.Sequential(nn.Linear(sum(widths.values()), d), nn.GELU(), nn.Dropout(p_drop))
        elif fusion == "gated":
            assert proj, "gated fusion requires equal widths"
            self.gate = nn.Linear(d * len(self.keys), len(self.keys))
            self.fuse = nn.Identity()
        elif fusion == "mean":
            assert proj
            self.fuse = nn.Identity()
        else:
            raise ValueError(fusion)
        self.classifier = nn.Linear(d, n_classes)
        self.last_gate = None

    def branch_embeddings(self, feats: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        zs = [self.projectors[k](feats[k]) for k in self.keys]
        if self.training and self.branch_drop > 0 and len(zs) > 1:
            B = zs[0].shape[0]
            keep = (torch.rand(B, len(zs), device=zs[0].device) > self.branch_drop).float()
            keep[keep.sum(1) == 0, 0] = 1.0  # never drop all branches
            scale = len(zs) / keep.sum(1, keepdim=True)
            zs = [z * (keep[:, i:i + 1] * scale) for i, z in enumerate(zs)]
        return zs

    def forward(self, feats: dict[str, torch.Tensor], mask: dict[str, float] | None = None):
        zs = self.branch_embeddings(feats)
        if mask is not None:  # test-time branch removal (robustness analysis)
            m = torch.tensor([mask.get(k, 1.0) for k in self.keys], device=zs[0].device)
            scale = len(zs) / m.sum().clamp(min=1)
            zs = [z * m[i] * scale for i, z in enumerate(zs)]
        if self.fusion == "concat":
            z = self.fuse(torch.cat(zs, 1))
        elif self.fusion == "gated":
            a = torch.softmax(self.gate(torch.cat(zs, 1)), 1)
            self.last_gate = a.detach()
            z = sum(a[:, i:i + 1] * zi for i, zi in enumerate(zs))
        else:
            z = torch.stack(zs, 0).mean(0)
        return self.classifier(z)


class RMBFF(nn.Module):
    """End-to-end model: backbones + RMBFFHead (for cluster fine-tuning)."""

    def __init__(self, backbones: dict[str, nn.Module], n_classes: int, **head_kw):
        super().__init__()
        self.backbones = nn.ModuleDict(backbones)
        dims = {}
        with torch.no_grad():  # infer pooled-embedding width (differs from num_features for MobileNetV3)
            for k, m in backbones.items():
                was = m.training; m.eval()
                dims[k] = m(torch.zeros(1, 3, 224, 224)).shape[1]
                m.train(was)
        self.head = RMBFFHead(dims, n_classes, **head_kw)

    def forward(self, x: dict[str, torch.Tensor] | torch.Tensor):
        # x may be a dict of per-backbone normalised tensors or a single tensor
        feats = {k: m(x[k] if isinstance(x, dict) else x) for k, m in self.backbones.items()}
        return self.head(feats)
