"""E04 -- first end-to-end LightGBM pair matcher.

Pipeline: cached candidates -> context features -> pairwise similarity
features -> LightGBM binary classifier -> F0.5-optimised decision layer.

The objective is binary log-loss, but *selection* is never done on log-loss or
accuracy: a classifier that calls everything negative scores 91% accuracy here
and 0.0 on the matched entities. Everything reported is the competition's
macro-averaged F_0.5 per S1 entity.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.io import CACHE, load_ground_truth  # noqa: E402
from data.splits import load_split  # noqa: E402
from features.pair_features import attach_sides, compute_features  # noqa: E402
from features.prepare import load_norm  # noqa: E402
from models.dataset import add_context, context_matrix, label, sample_negatives  # noqa: E402

MODEL_DIR = CACHE / "models"
FEAT_CHUNK = 1_500_000


def load_sides(split: str):
    """Load the normalised query/pool frames once.

    Re-loading these inside every ``featurize`` call was what made dev scoring
    take 10,090s instead of ~230s: two copies of the 10.3M-row pool coexisted
    and the process went to swap, running 44x slower than its measured
    11,000 pairs/s. They are hoisted and passed in explicitly.
    """
    s1 = load_norm(split, "source1")
    pool = pl.concat([load_norm(split, "source2"), load_norm(split, "source3")])
    return s1, pool


def featurize(pairs: pl.DataFrame, split: str, verbose: bool = True, sides=None):
    """Attach both sides, compute similarity + context features in chunks."""
    s1, pool = sides if sides is not None else load_sides(split)
    meta_cols = ["source1_entity_id", "cand_id"] + (["y"] if "y" in pairs.columns else [])
    xs, metas, names = [], [], None
    for lo in range(0, pairs.height, FEAT_CHUNK):
        sl = pairs[lo:lo + FEAT_CHUNK]
        j = attach_sides(sl.select("source1_entity_id", "cand_id"), s1, pool)
        # attach_sides is an inner join, so re-attach context/labels by key
        j = j.join(sl, on=["source1_entity_id", "cand_id"], how="left")
        X_sim, names_sim = compute_features(j)
        X_ctx, names_ctx = context_matrix(j)
        xs.append(np.hstack([X_sim, X_ctx]))
        # keep only the id/label columns -- retaining the full 18-string-column
        # joined frame per chunk is what exhausts RAM at this scale
        metas.append(j.select(meta_cols))
        del j, X_sim, X_ctx
        if names is None:
            names = names_sim + names_ctx
        if verbose:
            print(f"    features {min(lo + FEAT_CHUNK, pairs.height):,}/{pairs.height:,}",
                  end="\r", flush=True)
    X = np.vstack(xs)
    del xs
    meta = pl.concat(metas)
    if verbose:
        print()
    return X, names, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cand", required=True)
    ap.add_argument("--dev-cand", required=True)
    ap.add_argument("--n-hard", type=int, default=8)
    ap.add_argument("--n-rand", type=int, default=4)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--learning-rate", type=float, default=0.06)
    ap.add_argument("--out", default=str(MODEL_DIR / "lgbm_e04.txt"))
    a = ap.parse_args()

    t0 = time.time()
    gt = load_ground_truth()

    print("building training pairs ...")
    tr = add_context(pl.read_parquet(a.train_cand))
    tr = sample_negatives(label(tr, gt), a.n_hard, a.n_rand)
    sides = load_sides("train")
    Xtr, names, mtr = featurize(tr, "train", sides=sides)
    ytr = mtr["y"].to_numpy()
    print(f"  X_train={Xtr.shape} pos={int(ytr.sum()):,} "
          f"({Xtr.nbytes / 1e9:.2f} GB) [{time.time() - t0:.0f}s]")

    print("training LightGBM ...")
    t1 = time.time()
    model = lgb.LGBMClassifier(
        objective="binary", num_leaves=a.num_leaves, n_estimators=a.n_estimators,
        learning_rate=a.learning_rate, min_child_samples=60,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        reg_lambda=1.0, n_jobs=-1, verbose=-1, random_state=7)
    model.fit(Xtr, ytr, feature_name=names)
    t_train = time.time() - t1
    print(f"  trained in {t_train:.0f}s")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(a.out)
    json.dump(names, open(Path(a.out).with_suffix(".features.json"), "w"))

    imp = sorted(zip(names, model.booster_.feature_importance("gain")),
                 key=lambda x: -x[1])
    print("  top features by gain:")
    for n, g in imp[:18]:
        print(f"    {n:24s} {g:,.0f}")

    del Xtr, ytr, tr, mtr

    print("scoring dev candidates ...")
    t2 = time.time()
    dev = add_context(pl.read_parquet(a.dev_cand))
    dev = label(dev, gt)
    Xd, _, md = featurize(dev, "train", sides=sides)
    proba = model.predict_proba(Xd)[:, 1].astype(np.float32)
    t_infer = time.time() - t2
    scored = md.with_columns(pl.Series("p", proba))
    out = CACHE / "candidates" / f"dev_scored_{Path(a.out).stem}.parquet"
    scored.write_parquet(out, compression="zstd")
    print(f"  scored {scored.height:,} dev pairs in {t_infer:.0f}s -> {out.name}")
    json.dump({"train_seconds": t_train, "infer_seconds": t_infer,
               "n_train_pairs": int(len(proba))},
              open(Path(a.out).with_suffix(".timing.json"), "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
