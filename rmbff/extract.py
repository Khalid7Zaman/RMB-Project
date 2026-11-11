"""Frozen-backbone feature extraction (Regime A).

Usage
    python -m rmbff.extract --manifest outputs/manifests/fer2013.csv --dataset fer2013 \
        --branches m v r s [--variant clean|gauss_noise|... --subset test]

Writes outputs/features/<dataset>/<variant>/<branch>.npy (float16, row order =
manifest order restricted to --subset) and a JSON log with timing and
software versions. Extraction is resumable (chunked checkpoints).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .backbones import BRANCHES, create_backbone, data_config
from .corruptions import apply as corrupt


class ImgDS(Dataset):
    def __init__(self, paths, tf, variant, severity, offsets):
        self.paths, self.tf, self.variant, self.sev, self.off = paths, tf, variant, severity, offsets

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = corrupt(img, self.variant, int(self.off[i]), self.sev)
        return self.tf(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--branches", nargs="+", default=list(BRANCHES))
    ap.add_argument("--variant", default="clean")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--subset", default="all", help="'all' or a split name, e.g. test")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", default="outputs/features")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(a.manifest)
    idx = np.arange(len(df)) if a.subset == "all" else np.where(df.split == a.subset)[0]
    paths = df.path.values[idx]
    tag = a.variant if a.variant == "clean" else f"{a.variant}_s{a.severity}"
    odir = os.path.join(a.out, a.dataset, tag)
    os.makedirs(odir, exist_ok=True)
    np.save(os.path.join(odir, "index.npy"), idx)
    for b in a.branches:
        fout = os.path.join(odir, f"{b}.npy")
        if os.path.exists(fout):
            print("exists", fout); continue
        model = create_backbone(b).eval().to(dev)
        cfg = data_config(model)
        cfg["crop_pct"] = 1.0  # faces are pre-cropped: resize whole image, no centre crop
        tf = timm.data.create_transform(**cfg, is_training=False)
        feats, t0 = [], time.time()
        for c0 in range(0, len(paths), a.chunk):
            cf = os.path.join(odir, f".{b}_{c0}.npy")
            if os.path.exists(cf):
                feats.append(np.load(cf)); continue
            ds = ImgDS(paths[c0:c0 + a.chunk], tf, a.variant, a.severity, idx[c0:c0 + a.chunk])
            out = []
            with torch.inference_mode():
                for x in DataLoader(ds, batch_size=a.bs, num_workers=a.workers):
                    out.append(model(x.to(dev)).float().cpu().numpy().astype(np.float16))
            out = np.concatenate(out)
            np.save(cf, out); feats.append(out)
            print(f"{b} {c0 + len(out)}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
        F = np.concatenate(feats)
        np.save(fout, F)
        for c0 in range(0, len(paths), a.chunk):
            p = os.path.join(odir, f".{b}_{c0}.npy")
            if os.path.exists(p):
                os.remove(p)
        log = dict(branch=b, timm_name=BRANCHES[b], n=len(F), dim=F.shape[1], seconds=time.time() - t0,
                   device=dev, data_cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                   torch=torch.__version__, timm=timm.__version__, python=platform.python_version(),
                   cpu=platform.processor() or platform.machine(), threads=torch.get_num_threads(),
                   variant=a.variant, severity=a.severity, subset=a.subset)
        json.dump(log, open(os.path.join(odir, f"{b}.json"), "w"), indent=1)
        print("saved", fout, F.shape, f"{log['seconds']:.0f}s")


if __name__ == "__main__":
    main()
