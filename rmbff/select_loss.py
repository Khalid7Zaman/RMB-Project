"""Select the training objective per branch set by mean validation macro-F1 (E1)."""
import glob
import json
import os

import pandas as pd

rows = []
for f in glob.glob("outputs/runs/fer_loss/*/metrics.json"):
    c = json.load(open(os.path.join(os.path.dirname(f), "config.json")))
    m = json.load(open(f))
    rows.append(dict(branches="".join(sorted(c["branches"])), loss=c["loss"], seed=c["seed"],
                     val_f1=m["val"]["macro_f1"]))
df = pd.DataFrame(rows)
agg = df.groupby(["branches", "loss"]).val_f1.agg(["mean", "std", "count"]).reset_index()
best = agg.loc[agg.groupby("branches")["mean"].idxmax()]
sel = {"loss": dict(zip(best.branches, best.loss)), "criterion": "mean validation macro-F1 over seeds",
       "table": agg.to_dict(orient="records")}
json.dump(sel, open("outputs/selection.json", "w"), indent=1)
print(agg.to_string(index=False)); print(sel["loss"])


# ---- stage 2 (E1b): regularisation / width / branch-dropout per branch set, objective fixed from stage 1
rows = []
for f in glob.glob("outputs/runs/fer_tune*/*/metrics.json"):
    c = json.load(open(os.path.join(os.path.dirname(f), "config.json")))
    m = json.load(open(f))
    rows.append(dict(branches="".join(sorted(c["branches"])), d=c["d"], dropout=c["dropout"], wd=c["wd"],
                     branch_drop=c.get("branch_drop", 0.0), loss=c["loss"], seed=c["seed"],
                     val_f1=m["val"]["macro_f1"]))
if rows:
    df = pd.DataFrame(rows)
    keys = ["branches", "d", "dropout", "wd", "branch_drop", "loss"]
    agg = df.groupby(keys).val_f1.agg(["mean", "std", "count"]).reset_index()
    best = agg.loc[agg.groupby("branches")["mean"].idxmax()]
    sel["hp"] = {r.branches: dict(loss=r.loss, d=int(r.d), dropout=float(r.dropout), wd=float(r.wd),
                                  branch_drop=float(r.branch_drop), val_f1_mean=float(r["mean"]))
                 for _, r in best.iterrows()}
    sel["hp_criterion"] = "mean validation macro-F1 over seeds 0-1 (stage 2 grid)"
    sel["hp_table"] = agg.to_dict(orient="records")
    json.dump(sel, open("outputs/selection.json", "w"), indent=1)
    print(best.to_string(index=False))
