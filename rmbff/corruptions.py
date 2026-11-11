"""Deterministic image corruptions emulating IoT camera / channel degradations.

Each corruption is applied to the PIL image before backbone-specific resizing.
Severity levels follow a single mid setting by default; the RNG is seeded per
image index so every backbone sees exactly the same corrupted pixels.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

CORRUPTIONS = ["gauss_noise", "blur", "low_light", "jpeg", "low_res", "occlusion"]


def apply(img: Image.Image, kind: str, idx: int, severity: int = 3) -> Image.Image:
    rng = np.random.default_rng(10_000 + idx)
    if kind == "clean":
        return img
    if kind == "aug_flip":               # augmentation copy 1: horizontal flip
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if kind == "aug_rot":                # augmentation copy 2: small rotation + brightness change
        ang = float(rng.uniform(-10, 10)); gain = float(rng.uniform(0.8, 1.2))
        r = img.rotate(ang, resample=Image.BILINEAR)
        a2 = np.asarray(r).astype(np.float32) * gain
        return Image.fromarray(np.clip(a2, 0, 255).astype(np.uint8))
    a = np.asarray(img).astype(np.float32)
    if kind == "gauss_noise":            # sensor noise, sigma in 8-bit units
        sigma = [0, 8, 16, 24, 32, 48][severity]
        a = a + rng.normal(0, sigma, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if kind == "blur":                   # defocus / motion surrogate, radius relative to width
        r = [0, .01, .02, .03, .045, .06][severity] * img.width
        return img.filter(ImageFilter.GaussianBlur(radius=r))
    if kind == "low_light":              # under-exposure: gamma + gain
        g, k = [(1, 1), (1.5, .8), (2, .6), (2.5, .45), (3, .35), (3.5, .25)][severity]
        a = 255.0 * k * (a / 255.0) ** g
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if kind == "jpeg":                   # bandwidth-constrained transmission
        q = [95, 40, 25, 15, 10, 6][severity]
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        return Image.open(io.BytesIO(buf.getvalue())).convert(img.mode)
    if kind == "low_res":                # distant subject / low-resolution camera
        s = [1, .75, .5, .375, .25, .2][severity]
        w, h = img.size
        small = img.resize((max(4, int(w * s)), max(4, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    if kind == "occlusion":              # square occluder covering a fraction of area
        frac = [0, .05, .1, .15, .2, .3][severity]
        w, h = img.size
        side = int(np.sqrt(frac) * min(w, h))
        x0 = int(rng.integers(0, w - side + 1)); y0 = int(rng.integers(0, h - side + 1))
        a[y0:y0 + side, x0:x0 + side] = 0
        return Image.fromarray(a.astype(np.uint8))
    raise ValueError(kind)
