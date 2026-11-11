"""Dataset indexing and leakage-aware splitting for RMB-FF.

Every dataset is converted into a manifest (pandas DataFrame) with columns
    path, label, label_name, split, group
where ``group`` identifies the recording unit (subject / video / session)
used for group-wise splitting. Manifests are written to CSV so that every
experiment uses exactly the same partition.

Supported datasets
------------------
* FER2013   folder layout  <root>/{train,test}/<class>/<Usage>_<id>.jpg
            Official protocol: Training -> train, PublicTest -> val,
            PrivateTest -> test. Groups are unknown (web images), so each
            image is its own group.
* CMD       authors' custom dataset, folder layout <root>/<Class>/<prefix> (k).jpg
            The filename prefix encodes the recording group (video/subject);
            splitting is done on groups (StratifiedGroupKFold) to avoid
            identity / near-duplicate-frame leakage.
* folder    any ImageFolder-style dataset with predefined train/val/test
            sub-folders (e.g. licensed AffectNet or RAF-DB on the cluster).
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Canonical label space (Ekman basic emotions + neutral + contempt)
CANON = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
ALIASES = {"angry": "anger", "anger": "anger", "contempt": "contempt", "disgust": "disgust",
           "fear": "fear", "happy": "happy", "happiness": "happy", "neutral": "neutral",
           "sad": "sad", "sadness": "sad", "surprise": "surprise"}
SHARED6 = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _canon(name: str) -> str:
    key = name.strip().lower()
    if key not in ALIASES:
        raise ValueError(f"Unknown emotion folder name: {name}")
    return ALIASES[key]


def file_md5(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def index_fer2013(root: str) -> pd.DataFrame:
    rows = []
    usage_to_split = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
    for sub in ("train", "test"):
        for cdir in sorted((Path(root) / sub).iterdir()):
            if not cdir.is_dir():
                continue
            lab = _canon(cdir.name)
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() not in IMG_EXT:
                    continue
                usage = f.name.split("_")[0]
                rows.append(dict(path=str(f), label_name=lab, split=usage_to_split[usage],
                                 group=f"fer_{f.stem}"))
    df = pd.DataFrame(rows)
    return _finalise(df)


def _cmd_group(fname: str) -> str:
    m = re.fullmatch(r"(.+?) \((\d+)\)\.\w+", fname)
    if m:
        return m.group(1)
    return re.split(r"[-_]", fname)[0]  # e.g. 'img-756.jpg' -> 'img'


def index_cmd(root: str) -> pd.DataFrame:
    rows = []
    for cdir in sorted(Path(root).iterdir()):
        if not cdir.is_dir():
            continue
        lab = _canon(cdir.name)
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() not in IMG_EXT:
                continue
            rows.append(dict(path=str(f), label_name=lab, split="all",
                             group=f"cmd_{_cmd_group(f.name)}"))
    return _finalise(pd.DataFrame(rows))


def index_folder(root: str) -> pd.DataFrame:
    """ImageFolder with root/{train,val,test}/<class>/*.jpg (cluster use)."""
    rows = []
    for split in ("train", "val", "test"):
        sdir = Path(root) / split
        if not sdir.exists():
            continue
        for cdir in sorted(sdir.iterdir()):
            if cdir.is_dir():
                lab = _canon(cdir.name)
                for f in sorted(cdir.iterdir()):
                    if f.suffix.lower() in IMG_EXT:
                        rows.append(dict(path=str(f), label_name=lab, split=split,
                                         group=f"{split}_{f.stem}"))
    return _finalise(pd.DataFrame(rows))


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    classes = [c for c in CANON if c in set(df.label_name)]
    df["label"] = df.label_name.map({c: i for i, c in enumerate(classes)}).astype(int)
    df.attrs["classes"] = classes
    return df.reset_index(drop=True)


def classes_of(df: pd.DataFrame) -> list[str]:
    return [c for c in CANON if c in set(df.label_name)]


def relabel(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """Restrict to a label subset and re-index labels in canonical order."""
    out = df[df.label_name.isin(keep)].copy()
    classes = [c for c in CANON if c in keep]
    out["label"] = out.label_name.map({c: i for i, c in enumerate(classes)}).astype(int)
    return out


def cmd_group_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Assign outer test fold + inner validation fold, both group-disjoint.

    For outer fold k the test set is fold k; validation is the next fold
    (k+1 mod n); the remaining folds are training. Returns df with column
    ``fold`` in [0, n_splits).
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, te) in enumerate(sgkf.split(df.path, df.label, df.group)):
        fold[te] = k
    out = df.copy()
    out["fold"] = fold
    return out


def cmd_random_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Frame-level (leaky) stratified split, used ONLY to quantify leakage."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, te) in enumerate(skf.split(df.path, df.label)):
        fold[te] = k
    out = df.copy()
    out["fold"] = fold
    return out


def duplicate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Exact-duplicate (MD5) audit, reports duplicates within / across splits."""
    md5 = df.path.map(file_md5)
    d = df.assign(md5=md5)
    g = d.groupby("md5")
    rows = []
    for h, grp in g:
        if len(grp) > 1:
            rows.append(dict(md5=h, n=len(grp), splits="|".join(sorted(set(grp.split))),
                             labels="|".join(sorted(set(grp.label_name)))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer2013", default=os.environ.get("FER2013_ROOT", "data/fer2013"))
    ap.add_argument("--cmd", default=os.environ.get("CMD_ROOT", "data/cmd"))
    ap.add_argument("--out", default="outputs/manifests")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fer = index_fer2013(a.fer2013)
    fer.to_csv(f"{a.out}/fer2013.csv", index=False)
    cmd = index_cmd(a.cmd)
    cmd_group_folds(cmd, 5, a.seed).to_csv(f"{a.out}/cmd_groupfolds.csv", index=False)
    cmd_random_folds(cmd, 5, a.seed).to_csv(f"{a.out}/cmd_randomfolds.csv", index=False)
    # rows of FER2013 val/test whose exact pixel duplicate occurs in train
    md5 = fer.path.map(file_md5)
    train_h = set(md5[fer.split == "train"])
    leak = fer.index[(fer.split != "train") & md5.isin(train_h)]
    pd.DataFrame(dict(row=leak, split=fer.split[leak])).to_csv(f"{a.out}/fer2013_dup_rows.csv", index=False)
    print("FER2013 val/test images with exact duplicate in train:", fer.split[leak].value_counts().to_dict())
    for name, d in (("fer2013", fer), ("cmd", cmd)):
        dup = duplicate_report(d)
        dup.to_csv(f"{a.out}/{name}_duplicates.csv", index=False)
        print(name, len(d), d.groupby(["split", "label_name"]).size().unstack(0).to_dict(),
              "dup-groups:", len(dup))
