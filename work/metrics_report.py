import sys
sys.path.insert(0, r"D:\Dataset_ML_C\submission\code\business_entity_resolution\src")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

from common import WORK_DIR, pack_pair
import s03_train as t

OUT = Path(r"D:\Dataset_ML_C\submission\metrics")
OUT.mkdir(exist_ok=True)
truth, n_true = t.load_truth()
h = pd.read_parquet(WORK_DIR / "hold_pred.parquet")
h["bk"] = h.b_num.astype(np.int64) * 2 + (h.src == 3)
h["ck"] = h.country + "|" + h.bk.astype(str)
hs1 = h[h.hold_s1]
ents = hs1.groupby("country").s1_num.unique()
allE = np.concatenate(list(ents.values))
ent_country = pd.concat([pd.Series(v, index=v).map(lambda _: k) for k, v in ents.items()]) if False else None
country_of = {int(e): c for c, v in ents.items() for e in v}

tk = truth[np.isin(truth >> 33, allE)]
true_df = pd.DataFrame({"s1_num": tk >> 33, "src": np.where((tk & 1) == 1, 3, 2)})
true_df["country"] = true_df.s1_num.map(country_of)


def owned_at(p_min=0.05):
    return h.loc[h[h.p >= p_min].groupby("ck").p.idxmax()]


owned = owned_at()


def entity_frame(ids, thr, src=None):
    sel = owned[(owned.p >= thr) & owned.hold_s1 & np.isin(owned.s1_num, ids)]
    if src:
        sel = sel[sel.src == src]
    npred = sel.groupby("s1_num").size().reindex(ids).fillna(0).to_numpy()
    tp = sel.groupby("s1_num").y.sum().reindex(ids).fillna(0).to_numpy()
    tdf = true_df[np.isin(true_df.s1_num, ids)]
    if src:
        tdf = tdf[tdf.src == src]
    nt = tdf.groupby("s1_num").size().reindex(ids).fillna(0).to_numpy()
    prec = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
    rec = np.where(nt > 0, tp / np.maximum(nt, 1), 0.0)
    f = np.where(prec + rec > 0, 1.25 * prec * rec / (0.25 * prec + rec + 1e-12), 0.0)
    f = np.where(nt == 0, (npred == 0).astype(float), f)
    return pd.DataFrame({"s1": ids, "npred": npred, "tp": tp, "nt": nt, "prec": prec, "rec": rec, "f": f})


def summarize(e):
    has_t, has_p = e.nt > 0, e.npred > 0
    tp, npred, nt = e.tp.sum(), e.npred.sum(), e.nt.sum()
    mp, mr = tp / max(npred, 1), tp / max(nt, 1)
    return {
        "entities": len(e),
        "macro F0.5": e.f.mean(),
        "macro precision (entities with a prediction)": e.prec[has_p].mean(),
        "macro recall (entities with true matches)": e.rec[has_t].mean(),
        "micro precision": mp,
        "micro recall": mr,
        "micro F1": 2 * mp * mr / (mp + mr),
        "singleton share": (~has_t).mean(),
        "singletons predicted empty": (e.npred[~has_t] == 0).mean() if (~has_t).any() else np.nan,
        "entities with matches: perfect set": (e.f[has_t] > 0.999).mean(),
        "entities with matches: empty prediction": (e.npred[has_t] == 0).mean(),
        "entities with any wrong pair": ((e.npred - e.tp) > 0).mean(),
        "pairs: TP": int(tp), "pairs: FP": int(npred - tp), "pairs: FN": int(nt - tp),
    }


THR = 0.7
rows = {"overall": summarize(entity_frame(allE, THR))}
for c, v in ents.items():
    rows[c] = summarize(entity_frame(v, THR))
tab_main = pd.DataFrame(rows)

# by source (pair level)
src_rows = {}
for s, nm in ((2, "S2 matches"), (3, "S3 matches")):
    e = entity_frame(allE, THR, src=s)
    d = summarize(e)
    src_rows[nm] = {k: d[k] for k in ("micro precision", "micro recall", "micro F1", "pairs: TP", "pairs: FP", "pairs: FN")}
tab_src = pd.DataFrame(src_rows)

# by number of true matches
e = entity_frame(allE, THR)
e["n_true"] = np.minimum(e.nt, 8).astype(int)
tab_n = e.groupby("n_true").agg(entities=("f", "size"), macro_F05=("f", "mean"), mean_precision=("prec", "mean"), mean_recall=("rec", "mean"),
                                pct_perfect=("f", lambda s: (s > 0.999).mean())).round(4)

# threshold sweep
sweep = []
for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    d = summarize(entity_frame(allE, th))
    sweep.append({"threshold": th, "macro F0.5": d["macro F0.5"], "micro precision": d["micro precision"], "micro recall": d["micro recall"],
                  "singletons predicted empty": d["singletons predicted empty"], "entities with matches, empty prediction": d["entities with matches: empty prediction"]})
tab_sweep = pd.DataFrame(sweep).set_index("threshold")
no_owner = h[(h.p >= THR) & h.hold_s1]
e0 = t.macro_f05(allE, n_true, no_owner.s1_num.to_numpy(), no_owner.y.to_numpy())
oracle = t.macro_f05(allE, n_true, h[h.hold_s1 & h.y].s1_num.to_numpy(), np.ones(int((h.hold_s1 & h.y).sum()), bool))

# model quality on all hold rows
yy, pp = h.y.to_numpy(), h.p.to_numpy()
model = {"hold rows": len(h), "positive rate": yy.mean(), "ROC-AUC": roc_auc_score(yy, pp), "average precision": average_precision_score(yy, pp),
         "log-loss": log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6))}
tab_model = pd.Series(model).to_frame("value")

# blocking quality on full train candidates
blk = []
for c in ("India", "US"):
    cand = pd.read_parquet(WORK_DIR / "train" / c / "cand.parquet", columns=["s1_num", "b_num", "src"])
    rec = pd.read_parquet(WORK_DIR / "train" / c / "records.parquet", columns=["src"])
    n1, n2, n3 = [(rec.src == s).sum() for s in (1, 2, 3)]
    ck = pack_pair(cand.s1_num.values, cand.b_num.values, cand.src.values)
    hit = np.isin(ck, truth)
    truth_c = truth[np.isin(truth >> 33, np.unique(cand.s1_num.values))]
    for s in (2, 3):
        ts = truth_c[(truth_c & 1) == (1 if s == 3 else 0)]
        hs = hit & (cand.src.values == s)
        blk.append({"country": c, "source": f"S{s}", "S1 entities": n1, "pool size": n2 if s == 2 else n3, "true pairs": len(ts),
                    "candidates": int((cand.src.values == s).sum()), "candidates per S1": (cand.src.values == s).sum() / n1,
                    "pair recall": hs.sum() / len(ts), "reduction ratio": 1 - (cand.src.values == s).sum() / (n1 * (n2 if s == 2 else n3))})
    # entity coverage
    df = pd.DataFrame({"s1": cand.s1_num.values, "hit": hit})
    tr = pd.DataFrame({"s1": truth_c >> 33}).groupby("s1").size()
    found = df[df.hit].groupby("s1").size().reindex(tr.index).fillna(0)
    blk.append({"country": c, "source": "S2+S3 (entity level)", "S1 entities": n1, "pool size": n2 + n3, "true pairs": int(tr.sum()),
                "candidates": len(cand), "candidates per S1": len(cand) / n1, "pair recall": hit.sum() / tr.sum(),
                "reduction ratio": 1 - len(cand) / (n1 * (n2 + n3)),
                "entities with all true matches in candidates": (found == tr).mean(), "entities with >=1 true match in candidates": (found > 0).mean()})
tab_blk = pd.DataFrame(blk)

# test-set output statistics
s1t = pd.read_csv(r"D:\Dataset_ML_C\student_resource\dataset\test\test_source1.tsv", sep="\t", dtype=str, usecols=["entity_id", "country"])
m = pd.read_csv(r"D:\Dataset_ML_C\submission\output\matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
d = s1t.merge(m, left_on="entity_id", right_on="source1_entity_id")
d["n"] = np.where(d.matched_entity_ids == "", 0, d.matched_entity_ids.str.count(",") + 1)
d["is_empty"] = d.n == 0
ctest = {c: pd.read_parquet(WORK_DIR / "test" / c / "cand.parquet", columns=["src"]).shape[0] for c in d.country.unique()}
tab_test = d.groupby("country").agg(S1_entities=("n", "size"), mean_matches=("n", "mean"), empty_rate=("is_empty", "mean"), max_matches=("n", "max")).round(4)
tab_test["candidate pairs"] = pd.Series(ctest)
tab_test["candidates per S1"] = (tab_test["candidate pairs"] / tab_test.S1_entities).round(1)
tab_test.loc["ALL"] = [len(d), d.n.mean(), d.is_empty.mean(), d.n.max(), sum(ctest.values()), sum(ctest.values()) / len(d)]

extra = pd.Series({"macro F0.5 with threshold only (no one-owner rule)": e0.mean(), "macro F0.5 upper bound (every true pair in the candidate set accepted)": oracle.mean(),
                   "selected threshold": THR}).to_frame("value")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
def fmt(df):
    df = df.round(4)
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, r in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [("" if pd.isna(v) else (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v))) for v in r.tolist()]) + " |")
    return chr(10).join(lines)

sections = [
    ("1. Held-out matching quality (train entities never seen by the model, threshold 0.7 + one-owner-per-B rule)", tab_main),
    ("2. Pair-level quality by source", tab_src),
    ("3. Quality by number of true matches", tab_n),
    ("4. Threshold sweep", tab_sweep),
    ("5. Decision-rule reference points", extra),
    ("6. Model quality on all held-out candidate pairs", tab_model),
    ("7. Blocking quality on the full train set (full density)", tab_blk),
    ("8. Test-set submission statistics", tab_test),
]
with open(OUT / "metrics.md", "w", encoding="utf-8") as f:
    for title, tb in sections:
        f.write(f"## {title}\n\n{fmt(tb)}\n\n")
for i, (title, tb) in enumerate(sections, 1):
    tb.to_csv(OUT / f"metrics_{i}.csv")
    print("\n##", title)
    print(tb.round(4).to_string())
