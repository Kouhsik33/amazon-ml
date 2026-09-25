import sys
sys.path.insert(0, r"D:\Dataset_ML_C\submission\code\business_entity_resolution\src")
import lightgbm  # noqa: keep first on this machine
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from common import WORK_DIR
import s03_train as t

truth, n_true = t.load_truth()
m = pd.read_parquet(WORK_DIR / "cache" / "meta2_oof.parquet")
cidx = {c: i for i, c in enumerate(sorted(m.country.unique()))}
ck = (m.b_num.to_numpy().astype(np.int64) * 2 + (m.src.to_numpy() == 3)) * 4 + m.country.map(cidx).to_numpy()
ents = {c: g.s1_num.unique() for c, g in m.groupby("country")}
all_ids = np.concatenate(list(ents.values()))
s1, y, cty, src = m.s1_num.to_numpy(), m.y.to_numpy(), m.country.to_numpy(), m.src.to_numpy()


def owners(col):
    p = m[col].to_numpy()
    idx = np.flatnonzero(p >= 0.05)
    order = idx[np.lexsort((-p[idx], ck[idx]))]
    first = np.r_[True, ck[order][1:] != ck[order][:-1]]
    return order[first]


own = {c: owners(c) for c in ("p", "p2")}


def per_entity(col, thr, ids):
    rows = own[col]
    rows = rows[m[col].to_numpy()[rows] >= thr]
    rows = rows[np.isin(s1[rows], ids)]
    npred = pd.Series(1, index=s1[rows]).groupby(level=0).sum().reindex(ids).fillna(0).to_numpy()
    tp = pd.Series(y[rows].astype(int), index=s1[rows]).groupby(level=0).sum().reindex(ids).fillna(0).to_numpy()
    nt = n_true.reindex(ids).to_numpy()
    prec = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
    rec = np.where(nt > 0, tp / np.maximum(nt, 1), 0.0)
    f = np.where(prec + rec > 0, 1.25 * prec * rec / (0.25 * prec + rec + 1e-12), 0.0)
    f = np.where(nt == 0, (npred == 0).astype(float), f)
    return pd.DataFrame({"s1": ids, "npred": npred, "tp": tp, "nt": nt, "prec": prec, "rec": rec, "f": f})


def summ(e):
    ht, hp = e.nt > 0, e.npred > 0
    tp, npred, nt = e.tp.sum(), e.npred.sum(), e.nt.sum()
    mp, mr = tp / npred, tp / nt
    return {"macro F0.5": e.f.mean(), "macro precision": e.prec[hp].mean(), "macro recall": e.rec[ht].mean(),
            "micro precision": mp, "micro recall": mr, "micro F1": 2 * mp * mr / (mp + mr),
            "singletons predicted empty": (e.npred[~ht] == 0).mean(), "matched entities: perfect set": (e.f[ht] > 0.999).mean(),
            "matched entities: empty prediction": (e.npred[ht] == 0).mean(), "entities with any wrong pair": ((e.npred - e.tp) > 0).mean(),
            "TP": int(tp), "FP": int(npred - tp), "FN": int(nt - tp)}


THR = 0.70
rows = {}
E = {}
for col, nm in (("p", "stage 1"), ("p2", "stage 2")):
    E[nm] = per_entity(col, THR, all_ids)
    rows[nm] = summ(E[nm])
main = pd.DataFrame(rows)
main["change"] = main["stage 2"] - main["stage 1"]

d = E["stage 2"].f.to_numpy() - E["stage 1"].f.to_numpy()
se = d.std(ddof=1) / np.sqrt(len(d))
print(f"paired difference in macro F0.5 at threshold {THR}: {d.mean():+.5f}  (95% CI {d.mean() - 1.96 * se:+.5f} .. {d.mean() + 1.96 * se:+.5f}); entities better/worse/same: "
      f"{(d > 1e-9).sum()}/{(d < -1e-9).sum()}/{(np.abs(d) <= 1e-9).sum()}")

by_c = {}
for c, ids in ents.items():
    for col, nm in (("p", "stage 1"), ("p2", "stage 2")):
        s = summ(per_entity(col, THR, ids))
        by_c[(c, nm)] = {k: s[k] for k in ("macro F0.5", "micro precision", "micro recall", "singletons predicted empty", "matched entities: perfect set")}
by_c = pd.DataFrame(by_c).T

nt = n_true.reindex(all_ids).to_numpy()
bucket = np.minimum(nt, 6)
nb = pd.DataFrame({"n_true": bucket, "stage 1": E["stage 1"].f.to_numpy(), "stage 2": E["stage 2"].f.to_numpy()}).groupby("n_true").agg(entities=("stage 1", "size"), stage1=("stage 1", "mean"), stage2=("stage 2", "mean"))
nb["change"] = nb.stage2 - nb.stage1

sweep = []
for th in (0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9):
    r = {"threshold": th}
    for col, nm in (("p", "stage 1"), ("p2", "stage 2")):
        s = summ(per_entity(col, th, all_ids))
        r[f"{nm} F0.5"], r[f"{nm} precision"], r[f"{nm} recall"] = s["macro F0.5"], s["micro precision"], s["micro recall"]
    sweep.append(r)
sweep = pd.DataFrame(sweep).set_index("threshold")

pair = {}
for col, nm in (("p", "stage 1"), ("p2", "stage 2")):
    p = m[col].to_numpy()
    acc = p >= THR
    pair[nm] = {"rows": len(m), "ROC-AUC": roc_auc_score(y, p), "average precision": average_precision_score(y, p),
                "log-loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "pair accuracy @0.7": (acc == y).mean(),
                "pair precision @0.7": (acc & y).sum() / acc.sum(), "pair recall @0.7": (acc & y).sum() / y.sum()}
pair = pd.DataFrame(pair)
pair["change"] = pair["stage 2"] - pair["stage 1"]

pd.set_option("display.width", 250)
for title, tb in (("Entity-level metrics (100,124 held-out entities, threshold 0.7, one-owner rule)", main), ("By country", by_c), ("By number of true matches (macro F0.5)", nb),
                  ("Threshold sweep", sweep), ("Pair-level metrics on the 2.99M candidate rows of the held-out entities", pair)):
    print("\n##", title)
    print(tb.round(5).to_string())
with open(r"D:\Dataset_ML_C\submission\metrics\stage_comparison.md", "w", encoding="utf-8") as f:
    for title, tb in (("Entity-level", main), ("By country", by_c), ("By number of true matches", nb), ("Threshold sweep", sweep), ("Pair-level", pair)):
        f.write(f"## {title}\n\n```\n{tb.round(5).to_string()}\n```\n\n")
