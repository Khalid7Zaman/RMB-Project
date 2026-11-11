"""Backbone branches of RMB-FF (timm implementations, ImageNet-pretrained).

Branch keys follow the notation of the paper:
    m : MobileNetV3-Large        -> z_m
    v : Vision Transformer (ViT) -> z_v
    r : RegNetY                  -> z_r
    s : SE-ResNet (SE-enhanced residual) -> z_s

Weights are loaded through timm / Hugging Face Hub. For offline clusters set
``RMBFF_WEIGHTS=/path/to/dir`` containing <timm_name>.safetensors files
(see scripts/download_weights.py).
"""
from __future__ import annotations

import os

import timm
import torch

BRANCHES = {
    "m": "mobilenetv3_large_100.ra_in1k",
    "v": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "r": "regnety_016.tv2_in1k",
    "s": "seresnet50.a1_in1k",
}
BRANCH_NAMES = {"m": "MobileNetV3-L", "v": "ViT-S/16", "r": "RegNetY-1.6GF", "s": "SE-ResNet-50"}


def create_backbone(key: str, pretrained: bool | None = None) -> torch.nn.Module:
    """``RMBFF_PRETRAINED=0`` gives random weights (smoke tests only)."""
    if pretrained is None:
        pretrained = os.environ.get("RMBFF_PRETRAINED", "1") != "0"
    name = BRANCHES[key]
    wdir = os.environ.get("RMBFF_WEIGHTS")
    f = os.path.join(wdir, f"{name}.safetensors") if wdir else None
    if pretrained and f and os.path.exists(f):
        # offline: build the architecture, then load the checkpoint strictly (classifier dropped)
        from safetensors.torch import load_file
        m = timm.create_model(name, pretrained=False)
        m.load_state_dict(load_file(f), strict=True)
        m.reset_classifier(0)  # pooled embedding output
        return m
    return timm.create_model(name, pretrained=pretrained, num_classes=0)


def data_config(model: torch.nn.Module) -> dict:
    """Per-backbone preprocessing (input size, interpolation, mean, std)."""
    return timm.data.resolve_data_config({}, model=model)
