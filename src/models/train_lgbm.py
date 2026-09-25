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

import gc
import resource
import subprocess

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
FEAT_CHUNK = 1_000_000


def peak_rss_gb() -> float:
    """macOS reports ru_maxrss in bytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def swap_used_gb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout
    try:
        return float(out.split("used =")[1].split("M")[0].strip()) / 1024.0
    except Exception:
        return float("nan")


def free_ram_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "Pages free" in line:
            return int(line.split(":")[1].strip().rstrip(".")) * 4096 / 1e9
    return float("nan")


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


def featurize(pairs: pl.DataFrame, split: str, verbose: bool = True, sides=None,
              mmap_path: str | None = None):
    """Attach both sides, compute similarity + context features in chunks."""
    s1, pool = sides if sides is not None else load_sides(split)
    meta_cols = ["source1_entity_id", "cand_id"] + (["y"] if "y" in pairs.columns else [])
    # Preallocate and fill in place. np.vstack over chunk list transiently holds
    # two full copies of the matrix, which at 500k entities (~2 GB) is the
    # difference between fitting in RAM and swapping.
    X = None
    off = 0
    metas, names = [], None
    for lo in range(0, pairs.height, FEAT_CHUNK):
        sl = pairs[lo:lo + FEAT_CHUNK]
        j = attach_sides(sl.select("source1_entity_id", "cand_id"), s1, pool)
        # attach_sides is an inner join, so re-attach context/labels by key
        j = j.join(sl, on=["source1_entity_id", "cand_id"], how="left")
        X_sim, names_sim = compute_features(j)
        X_ctx, names_ctx = context_matrix(j)
        if names is None:
            names = names_sim + names_ctx
            shape = (pairs.height, len(names))
            if mmap_path:
                # Back the matrix with a file so its pages are evictable while
                # the ~3 GB of Polars side frames are still resident. At 500k
                # entities an in-RAM matrix (~2.2 GB) plus sides exceeds free
                # RAM and forces swap, which the runtime investigation showed
                # costs far more than the I/O does.
                X = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=shape)
            else:
                X = np.empty(shape, dtype=np.float32)
        n_chunk = X_sim.shape[0]
        X[off:off + n_chunk, :X_sim.shape[1]] = X_sim
        X[off:off + n_chunk, X_sim.shape[1]:] = X_ctx
        off += n_chunk
        # keep only the id/label columns -- retaining the full 18-string-column
        # joined frame per chunk is what exhausts RAM at this scale
        metas.append(j.select(meta_cols))
        del j, X_sim, X_ctx
        if verbose:
            print(f"    features {min(lo + FEAT_CHUNK, pairs.height):,}/{pairs.height:,}",
                  end="\r", flush=True)
    if mmap_path:
        X.flush()
    X = X[:off]                      # inner join may drop rows; trim to actual
    meta = pl.concat(metas)
    del metas
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
    ap.add_argument("--mmap", default=None,
                    help="path for a memory-mapped feature matrix (keeps RAM free "
                         "at large entity counts)")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="LightGBM threads. Default -1 reproduces E05 exactly; "
                         "6 (physical cores) is faster on this 12-logical box.")
    ap.add_argument("--out", default=str(MODEL_DIR / "lgbm_e04.txt"))
    a = ap.parse_args()

    t0 = time.time()
    mem = {"swap_used_gb_start": round(swap_used_gb(), 2),
           "free_ram_gb_start": round(free_ram_gb(), 2),
           "n_jobs": a.n_jobs}
    gt = load_ground_truth()

    print("building training pairs ...")
    tr = add_context(pl.read_parquet(a.train_cand))
    tr = sample_negatives(label(tr, gt), a.n_hard, a.n_rand)
    sides = load_sides("train")
    t_feat0 = time.time()
    Xtr, names, mtr = featurize(tr, "train", sides=sides, mmap_path=a.mmap)
    t_feat = time.time() - t_feat0
    ytr = mtr["y"].to_numpy()
    n_train_pairs, n_train_pos = int(Xtr.shape[0]), int(ytr.sum())
    print(f"  X_train={Xtr.shape} pos={int(ytr.sum()):,} "
          f"({Xtr.nbytes / 1e9:.2f} GB) [{time.time() - t0:.0f}s]")

    # Release everything the boosting loop does not read. Holding the
    # normalised S1 frame and the 10.3M-row S2/S3 pool through fit() measured
    # 1.5x slower in the runtime investigation (experiments/bench/README.md).
    del tr, sides
    gc.collect()
    mem["free_ram_gb_before_fit"] = round(free_ram_gb(), 2)
    mem["swap_used_gb_before_fit"] = round(swap_used_gb(), 2)
    mem["peak_rss_gb_before_fit"] = round(peak_rss_gb(), 2)
    print(f"  freed side frames -> free RAM {mem['free_ram_gb_before_fit']} GB, "
          f"swap {mem['swap_used_gb_before_fit']} GB, peak RSS "
          f"{mem['peak_rss_gb_before_fit']} GB")

    print("training LightGBM ...")
    t1 = time.time()
    model = lgb.LGBMClassifier(
        objective="binary", num_leaves=a.num_leaves, n_estimators=a.n_estimators,
        learning_rate=a.learning_rate, min_child_samples=60,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        reg_lambda=1.0, n_jobs=a.n_jobs, verbose=-1, random_state=7)
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

    del Xtr, ytr, mtr
    gc.collect()

    print("scoring dev candidates ...")
    t2 = time.time()
    sides = load_sides("train")          # reloaded only now that fit is done
    dev = add_context(pl.read_parquet(a.dev_cand))
    dev = label(dev, gt)
    Xd, _, md = featurize(dev, "train", sides=sides)
    proba = model.predict_proba(Xd)[:, 1].astype(np.float32)
    t_infer = time.time() - t2
    scored = md.with_columns(pl.Series("p", proba))
    out = CACHE / "candidates" / f"dev_scored_{Path(a.out).stem}.parquet"
    scored.write_parquet(out, compression="zstd")
    print(f"  scored {scored.height:,} dev pairs in {t_infer:.0f}s -> {out.name}")
    mem["peak_rss_gb"] = round(peak_rss_gb(), 2)
    mem["swap_used_gb_end"] = round(swap_used_gb(), 2)
    mem["swap_delta_gb"] = round(mem["swap_used_gb_end"] - mem["swap_used_gb_start"], 2)
    mem["free_ram_gb_end"] = round(free_ram_gb(), 2)
    json.dump({"train_seconds": t_train, "infer_seconds": t_infer,
               "feature_seconds": t_feat, "n_dev_pairs": int(len(proba)),
               "n_train_pairs": int(n_train_pairs),
               "n_train_pos": int(n_train_pos), **mem},
              open(Path(a.out).with_suffix(".timing.json"), "w"))
    print(f"  memory: peak RSS {mem['peak_rss_gb']} GB, swap delta "
          f"{mem['swap_delta_gb']:+.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
