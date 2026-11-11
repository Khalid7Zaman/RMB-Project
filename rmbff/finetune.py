"""End-to-end fine-tuning of RMB-FF (Regime B, GPU cluster).

    python -m rmbff.finetune --protocol fer_official --branches m v r s --loss la --seed 0 \
        --epochs 30 --out outputs/finetune/fer_mrsv_la_s0

Same split builder, metrics and output format as rmbff.train_head, so the
analysis/figure scripts work unchanged. Each backbone receives its own input
normalisation (ViT-augreg uses mean=std=0.5, CNNs use ImageNet statistics).
Checkpoints are written every epoch (resume with --resume).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .backbones import create_backbone, data_config
from .losses import ImbalanceLoss
from .metrics import compute_metrics
from .model import RMBFFHead
from .train_head import build_split, manifest, set_seed


class PerBranchNorm(nn.Module):
    def __init__(self, backbone, mean, std):
        super().__init__()
        self.b = backbone
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.b((x - self.mean) / self.std)


class E2E(nn.Module):
    def __init__(self, keys, n_classes, **kw):
        super().__init__()
        bbs = {}
        for k in keys:
            m = create_backbone(k)
            c = data_config(m)
            bbs[k] = PerBranchNorm(m, c["mean"], c["std"])
        self.backbones = nn.ModuleDict(bbs)
        with torch.no_grad():
            dims = {k: m.eval()(torch.zeros(1, 3, 224, 224)).shape[1] for k, m in bbs.items()}
        self.head = RMBFFHead(dims, n_classes, **kw)

    def forward(self, x):
        return self.head({k: m(x) for k, m in self.backbones.items()})


class DS(Dataset):
    def __init__(self, paths, y, tf):
        self.p, self.y, self.tf = paths, y, tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), int(self.y[i])


def loaders(cfg, bs, workers):
    sp = build_split(cfg)
    dfs = {"fer2013": manifest("fer2013"), "cmd": manifest("cmd")}
    tr_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                       T.RandomHorizontalFlip(), T.RandomRotation(10), T.ColorJitter(0.3, 0.3, 0.2),
                       T.ToTensor(), T.RandomErasing(p=0.25, scale=(0.02, 0.15))])
    te_tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])
    out = {}
    for s in ("train", "val", "test"):
        paths = dfs[sp[s]["dataset"]].path.values[sp[s]["rows"]]
        out[s] = DataLoader(DS(paths, sp[s]["y"], tr_tf if s == "train" else te_tf), batch_size=bs,
                            shuffle=(s == "train"), num_workers=workers, pin_memory=True,
                            drop_last=(s == "train"), persistent_workers=workers > 0)
    return sp, out


@torch.no_grad()
def infer(model, dl, dev):
    model.eval(); P = []
    for x, _ in dl:
        with torch.autocast(dev, enabled=dev == "cuda"):
            P.append(torch.softmax(model(x.to(dev, non_blocking=True)).float(), 1).cpu().numpy())
    return np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="fer_official")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--branches", nargs="+", default=["m", "v", "r", "s"])
    ap.add_argument("--loss", default="la")
    ap.add_argument("--fusion", default="concat")
    ap.add_argument("--branch_drop", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr_backbone", type=float, default=5e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(vars(a), protocol=a.protocol, fold=a.fold, dedup=a.dedup)
    sp, dl = loaders(cfg, a.bs, a.workers)
    classes = sp["classes"]
    counts = np.bincount(sp["train"]["y"], minlength=len(classes))
    model = E2E(sorted(a.branches), len(classes), fusion=a.fusion, branch_drop=a.branch_drop).to(dev)
    crit = ImbalanceLoss(a.loss, counts, label_smoothing=a.label_smoothing).to(dev)
    params = [dict(params=model.backbones.parameters(), lr=a.lr_backbone),
              dict(params=model.head.parameters(), lr=a.lr_head)]
    opt = torch.optim.AdamW(params, weight_decay=a.wd)
    steps = a.epochs * len(dl["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_backbone, a.lr_head], total_steps=steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    start, best, bad, curves = 0, -1.0, 0, []
    ck = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start, best, bad, curves = s["epoch"], s["best"], s["bad"], s["curves"]
    t0 = time.time()
    for ep in range(start, a.epochs):
        model.train(); tl, n = 0.0, 0
        for x, y in dl["train"]:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); n += len(y)
        pv = infer(model, dl["val"], dev)
        mv = compute_metrics(sp["val"]["y"], pv, classes)
        curves.append(dict(epoch=ep + 1, train_loss=tl / n, val_nll=mv["nll"], val_acc=mv["acc"],
                           val_macro_f1=mv["macro_f1"], seconds=time.time() - t0))
        print(curves[-1], flush=True)
        if mv["macro_f1"] > best + 1e-5:
            best, bad = mv["macro_f1"], 0
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1,
                        best=best, bad=bad, curves=curves), ck)
        if bad >= a.patience:
            break
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt"), map_location=dev))
    pv, pt = infer(model, dl["val"], dev), infer(model, dl["test"], dev)
    res = dict(val=compute_metrics(sp["val"]["y"], pv, classes), test=compute_metrics(sp["test"]["y"], pt, classes),
               classes=classes, epochs_run=len(curves), train_seconds=time.time() - t0,
               n_train=int(len(sp["train"]["y"])), n_val=int(len(sp["val"]["y"])), n_test=int(len(sp["test"]["y"])),
               gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", torch=torch.__version__)
    pd.DataFrame(curves).to_csv(os.path.join(a.out, "curves.csv"), index=False)
    np.save(os.path.join(a.out, "val_probs.npy"), pv); np.save(os.path.join(a.out, "test_probs.npy"), pt)
    np.save(os.path.join(a.out, "test_y.npy"), sp["test"]["y"]); np.save(os.path.join(a.out, "test_rows.npy"), sp["test"]["rows"])
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print("TEST", {k: res["test"][k] for k in ("acc", "macro_f1", "bal_acc")})


if __name__ == "__main__":
    main()
