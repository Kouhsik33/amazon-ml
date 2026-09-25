"""Step 3: build training data from train candidates, train the LightGBM matcher, tune the decision threshold.

Row selection (per country): a random ~HFRAC share of S1 entities H is split into H_train / H_hold.
  hold rows  = every candidate pair whose B record is a candidate of some H_hold entity (so the one-owner-per-B rule
               can be evaluated exactly on the held-out entities)
  train rows = candidate pairs of H_train entities whose B record is not among the hold rows' B records
"""
import gc
import json
import sys
import time

import lightgbm as lgb  # keep before pandas/numpy on this Windows setup
import numpy as np
import pandas as pd

from common import DATA_DIR, WORK_DIR, country_dir, id_num, pack_pair
from features import ALL_COLS, BASE_COLS, COMP_COLS, PairFeaturizer, competition_features

HFRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def u01(s1_num):
    return ((s1_num.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32)).astype(np.float64) / 2 ** 32


def load_truth():
    gt = pd.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    n_true = pd.Series(np.where(gt.matched_entity_ids == "", 0, gt.matched_entity_ids.str.count(",") + 1), index=id_num(gt.source1_entity_id))
    gt = gt[gt.matched_entity_ids != ""]
    p = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    keys = pack_pair(id_num(p.source1_entity_id), id_num(p.m), p.m.str.get(1).astype(int).to_numpy())
    return np.sort(keys), n_true


def macro_f05(s1_nums, n_true, sel_s1, sel_true):
    """s1_nums: evaluated entities; sel_*: accepted pairs (s1 and whether correct)."""
    tot = pd.Series(1, index=sel_s1).groupby(level=0).sum().reindex(s1_nums).fillna(0).to_numpy()
    tp = pd.Series(sel_true.astype(int), index=sel_s1).groupby(level=0).sum().reindex(s1_nums).fillna(0).to_numpy()
    nt = n_true.reindex(s1_nums).to_numpy()
    prec = np.where(tot > 0, tp / np.maximum(tot, 1), 0.0)
    rec = np.where(nt > 0, tp / np.maximum(nt, 1), 0.0)
    f = np.where(prec + rec > 0, 1.25 * prec * rec / (0.25 * prec + rec + 1e-12), 0.0)
    f = np.where(nt == 0, (tot == 0).astype(float), f)
    return f


def main():
    truth, n_true = load_truth()
    log(f"truth pairs {len(truth):,}")
    X_l, y_l, meta_l = [], [], []
    ents = {}
    for cdir in sorted((WORK_DIR / "train").iterdir()):
        country = cdir.name
        cand = pd.read_parquet(cdir / "cand.parquet")
        recs_all = pd.read_parquet(cdir / "records.parquet")
        recs = {s: recs_all[recs_all.src == s].reset_index(drop=True) for s in (1, 2, 3)}
        del recs_all
        u1 = u01(recs[1].num.to_numpy())
        ents[country] = recs[1].num.to_numpy()[(u1 >= HFRAC * 0.7) & (u1 < HFRAC)]
        dup = recs[1].groupby("name_n").num.transform("size")
        s1_dup = pd.Series(dup.to_numpy(), index=recs[1].num.to_numpy())
        comp = competition_features(cand, s1_dup)
        u = u01(cand.s1_num.to_numpy())
        bk = cand.b_num.to_numpy().astype(np.int64) * 2 + (cand.src.to_numpy() == 3)
        in_hold_s1 = (u >= HFRAC * 0.7) & (u < HFRAC)
        in_train_s1 = u < HFRAC * 0.7
        hold_b = np.unique(bk[in_hold_s1])
        hold_rows = np.isin(bk, hold_b)
        train_rows = in_train_s1 & ~hold_rows
        sel = np.flatnonzero(hold_rows | train_rows)
        c = cand.iloc[sel].reset_index(drop=True)
        log(f"{country}: {len(cand):,} candidate pairs; selected {len(c):,} (hold {int(hold_rows.sum()):,}, train {int(train_rows.sum()):,})")
        pf = PairFeaturizer(str(cdir / "idf.pkl"), recs)
        base = pf.compute(c)
        pf.close()
        X = np.concatenate([base, comp.iloc[sel][COMP_COLS].to_numpy(np.float32)], axis=1)
        y = np.isin(pack_pair(c.s1_num.to_numpy(), c.b_num.to_numpy(), c.src.to_numpy()), truth)
        meta = pd.DataFrame({"country": country, "s1_num": c.s1_num.to_numpy(), "b_num": c.b_num.to_numpy(), "src": c.src.to_numpy(),
                             "hold": hold_rows[sel], "hold_s1": in_hold_s1[sel]})
        X_l.append(X)
        y_l.append(y)
        meta_l.append(meta)
        log(f"{country}: features done, positives {int(y.sum()):,}")
        del cand, recs, comp, c, base, X
        gc.collect()

    X = np.concatenate(X_l)
    y = np.concatenate(y_l).astype(np.float32)
    meta = pd.concat(meta_l, ignore_index=True)
    del X_l, y_l, meta_l
    tr = ~meta.hold.to_numpy()
    ho = meta.hold.to_numpy()
    log(f"train rows {int(tr.sum()):,}  hold rows {int(ho.sum()):,}  features {X.shape[1]}")

    params = dict(objective="binary", learning_rate=0.06, num_leaves=127, min_child_samples=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=12)
    dtr = lgb.Dataset(X[tr], y[tr], feature_name=ALL_COLS)
    dho = lgb.Dataset(X[ho], y[ho], reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dho], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
    booster.save_model(str(WORK_DIR / "model.txt"))
    log(f"trained, best iteration {booster.best_iteration}")

    h = meta[ho].reset_index(drop=True)
    h["p"] = booster.predict(X[ho], num_iteration=booster.best_iteration)
    h["y"] = y[ho] > 0
    h.to_parquet(WORK_DIR / "hold_pred.parquet", index=False)
    imp = pd.Series(booster.feature_importance("gain"), index=ALL_COLS)
    print((imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_string())

    h["bk"] = h.b_num.astype(np.int64) * 2 + (h.src == 3)
    h["ck"] = h.country + "|" + h.bk.astype(str)
    owned = h.loc[h[h.p >= 0.05].groupby("ck").p.idxmax()]
    best = (-1, None)
    rows = []
    for t in np.arange(0.3, 0.96, 0.05):
        sel = owned[owned.p >= t]
        sel = sel[sel.hold_s1]
        tot_f, tot_n = 0.0, 0
        per_c = {}
        for country, ids in ents.items():
            s = sel[sel.country == country]
            f = macro_f05(ids, n_true, s.s1_num.to_numpy(), s.y.to_numpy())
            per_c[country] = f.mean()
            tot_f += f.sum()
            tot_n += len(f)
        rows.append((round(t, 2), tot_f / tot_n, per_c))
        if tot_f / tot_n > best[0]:
            best = (tot_f / tot_n, round(float(t), 2))
    for t, f, pc in rows:
        print(f"t={t}: macro F0.5={f:.4f}  " + "  ".join(f"{k}={v:.4f}" for k, v in pc.items()))
    log(f"best threshold {best[1]} with macro F0.5 {best[0]:.4f} on {sum(len(v) for v in ents.values()):,} held-out entities")
    with open(WORK_DIR / "decision.json", "w") as f:
        json.dump({"threshold": best[1], "holdout_macro_f05": best[0], "features": ALL_COLS}, f)


if __name__ == "__main__":
    main()
