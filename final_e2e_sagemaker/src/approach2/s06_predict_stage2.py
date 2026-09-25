"""Step 6: second-stage scoring of the test candidates.

Needs: step 4 already run (WORK_DIR/test/<country>/pred.parquet holds stage-1 probabilities), model2.txt, decision2.json.
Writes matching_results.tsv (stage-2 decisions) into ER_OUT_DIR2 (default: <OUT_DIR>_stage2); candidate_pairs.tsv is unchanged
(the same candidate set is scored), so it is copied.
"""
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb  # keep before pandas/numpy on this Windows setup
import numpy as np
import pandas as pd

from common import DATA_DIR, OUT_DIR, WORK_DIR
from features import BASE_COLS, COMP_COLS, PairFeaturizer, competition_features
from s04_predict import join_ids, write_tsv
from stage2 import ANCHOR_COLS, context_features

OUT2 = Path(os.environ.get("ER_OUT_DIR2", str(OUT_DIR) + "_stage2"))
THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 0.70
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def score_country(cdir, booster):
    cand = pd.read_parquet(cdir / "cand.parquet")
    pred = pd.read_parquet(cdir / "pred.parquet")
    assert (cand.s1_num.to_numpy() == pred.s1_num.to_numpy()).all() and (cand.b_num.to_numpy() == pred.b_num.to_numpy()).all()
    p1 = pred.p.to_numpy()
    recs_all = pd.read_parquet(cdir / "records.parquet")
    recs = {s: recs_all[recs_all.src == s].reset_index(drop=True) for s in (1, 2, 3)}
    del recs_all
    dup = recs[1].groupby("name_n").num.transform("size")
    comp = competition_features(cand, pd.Series(dup.to_numpy(), index=recs[1].num.to_numpy()))[COMP_COLS].to_numpy(np.float32)
    df = cand[["s1_num", "b_num", "src"]].copy()
    df["p"] = p1
    ctx, anchor = context_features(df)
    ctx = ctx.to_numpy(np.float32)
    log(f"  {cdir.name}: context features done ({len(df):,} rows)")

    a = np.where(anchor >= 0, anchor, np.arange(len(df)))
    tab = pd.DataFrame({"l_src": df.src.to_numpy()[a], "l_num": df.b_num.to_numpy()[a], "b_num": df.b_num.to_numpy(), "src": df.src.to_numpy(),
                        "sn": np.float32(0), "sa": np.float32(0), "sj": np.float32(0)})
    keep_cols = [BASE_COLS.index(c) for c in ANCHOR_COLS]
    del pred, df, a
    gc.collect()
    pf = PairFeaturizer(str(cdir / "idf.pkl"), recs, workers=4)
    n_rows = len(cand)
    anc = np.empty((n_rows, len(keep_cols)), np.float32)
    for lo, hi, F in pf.iter_chunks(tab):
        anc[lo:hi] = F[:, keep_cols]
    anc[anchor < 0] = np.nan
    del tab
    log(f"  {cdir.name}: anchor features done")

    p2 = np.empty(n_rows, np.float32)
    for lo, hi, base in pf.iter_chunks(cand):
        X = np.concatenate([base, comp[lo:hi], ctx[lo:hi], anc[lo:hi]], axis=1)
        p2[lo:hi] = booster.predict(X)
        if (lo // pf.chunk) % 10 == 0:
            log(f"  {cdir.name}: stage-2 scored {hi:,}/{n_rows:,}")
    pf.close()
    out = cand[["s1_num", "b_num", "src"]].copy()
    out["p"] = p1
    out["p2"] = p2
    out.to_parquet(cdir / "pred2.parquet", index=False)
    return out


def main(rescore=True):
    OUT2.mkdir(parents=True, exist_ok=True)
    booster = lgb.Booster(model_file=str(WORK_DIR / "model2.txt"))
    assert booster.feature_name() == json.load(open(WORK_DIR / "decision2.json"))["features"]
    log(f"stage-2 threshold {THRESHOLD}")
    s1 = pd.read_csv(DATA_DIR / "test" / "test_source1.tsv", sep="\t", dtype=str, usecols=["entity_id"])
    s1_nums = s1.entity_id.str.slice(3).astype(np.int64).to_numpy()
    matches = []
    for cdir in sorted((WORK_DIR / "test").iterdir()):
        if rescore or not (cdir / "pred2.parquet").exists():
            pred = score_country(cdir, booster)
        else:
            pred = pd.read_parquet(cdir / "pred2.parquet")
        keep = pred[pred.p2 >= 0.05]
        bk = keep.b_num.to_numpy().astype(np.int64) * 2 + (keep.src.to_numpy() == 3)
        win = keep.iloc[pd.Series(keep.p2.to_numpy()).groupby(bk).idxmax().to_numpy()]
        acc = win[win.p2 >= THRESHOLD]
        matches.append(join_ids(acc))
        log(f"{cdir.name}: stage-2 accepted {len(acc):,} pairs")
        del pred, keep, win, acc
        gc.collect()
    match_map = pd.concat(matches)
    write_tsv(OUT2 / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids", s1.entity_id.to_numpy(), s1_nums, match_map)
    shutil.copyfile(OUT_DIR / "candidate_pairs.tsv", OUT2 / "candidate_pairs.tsv")
    log(f"written to {OUT2}; entities with >=1 match {len(match_map):,} ({len(match_map) / len(s1):.1%})")


if __name__ == "__main__":
    main(rescore="--reuse" not in sys.argv)
