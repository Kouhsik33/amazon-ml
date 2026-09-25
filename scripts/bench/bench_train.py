#!/usr/bin/env python3
"""Benchmark LightGBM training for one feature set, in an isolated process.

Run one variant per process so peak RSS is attributable and no allocator state
carries over between variants. Phases timed separately:

    load        -- read the cached matrix from disk
    select      -- column selection + contiguity (the "numpy conversion" cost)
    dataset     -- lgb.Dataset construction, i.e. histogram bin-finding
    train       -- boosting itself
    score       -- prediction over the same rows

LightGBM's own log is captured at verbose=1, because its auto-selection of
row-wise vs col-wise multi-threading and any memory warnings are printed there
and are prime suspects for a super-linear slowdown.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

BENCH = Path("experiments/bench")

# Exactly the E05 training configuration -- unchanged, this is the control.
PARAMS = dict(
    objective="binary", num_leaves=127, n_estimators=600, learning_rate=0.06,
    min_child_samples=60, subsample=0.85, subsample_freq=1,
    colsample_bytree=0.85, reg_lambda=1.0, random_state=7,
)


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def swap_used_gb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    # parse "total = 9216.00M  used = 8478.25M  free = 737.75M"
    try:
        used = out.split("used =")[1].split("M")[0].strip()
        return float(used) / 1024.0
    except Exception:
        return float("nan")


def free_ram_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "Pages free" in line:
            return int(line.split(":")[1].strip().rstrip(".")) * 4096 / 1e9
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["e04", "e05"], required=True)
    ap.add_argument("--n-threads", type=int, default=-1)
    ap.add_argument("--force", choices=["auto", "col", "row"], default="auto")
    ap.add_argument("--n-rows", type=int, default=0, help="subsample rows (0 = all)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--hold-sides", action="store_true",
                    help="keep the Polars normalisation frames resident during "
                         "training, reproducing what train_lgbm.py actually does")
    a = ap.parse_args()

    names = json.loads((BENCH / "feature_names.json").read_text())
    e04_names = json.loads(Path("data_cache/models/lgbm_e04.features.json").read_text())

    rec = {"variant": a.features, "n_threads_requested": a.n_threads,
           "force_mode": a.force, "tag": a.tag,
           "swap_used_gb_before": round(swap_used_gb(), 2),
           "free_ram_gb_before": round(free_ram_gb(), 2)}

    t = time.time()
    X_all = np.load(BENCH / "X_e05.npy", mmap_mode=None)
    y = np.load(BENCH / "y.npy")
    rec["load_seconds"] = round(time.time() - t, 1)

    t = time.time()
    if a.features == "e04":
        idx = [names.index(n) for n in e04_names]
        X = np.ascontiguousarray(X_all[:, idx])
        del X_all
        feat = e04_names
    else:
        X = X_all
        feat = names
    if a.n_rows:
        X = np.ascontiguousarray(X[:a.n_rows])
        y = y[:a.n_rows]
    rec["select_seconds"] = round(time.time() - t, 1)
    rec["n_pairs"] = int(X.shape[0])
    rec["n_features"] = int(X.shape[1])
    rec["matrix_gb"] = round(float(X.nbytes / 1e9), 3)

    held = None
    if a.hold_sides:
        # train_lgbm.py builds the feature matrix from `sides` (the normalised
        # S1 + 10.3M-row S2/S3 pool) and those frames stay referenced for the
        # whole run, so LightGBM boosts with several GB of Polars data still
        # resident. Reproduce that residency exactly.
        sys.path.insert(0, str(Path("src").resolve()))
        import polars as pl
        from models.train_lgbm import load_sides
        from models.dataset import add_context
        t = time.time()
        held = (load_sides("train"), add_context(pl.read_parquet(
            "data_cache/candidates/cand_train_200k_k80.parquet")))
        rec["hold_sides_load_seconds"] = round(time.time() - t, 1)
        rec["rss_after_holding_gb"] = round(peak_rss_gb(), 2)
        rec["free_ram_gb_after_holding"] = round(free_ram_gb(), 2)

    params = dict(PARAMS)
    params["n_jobs"] = a.n_threads
    if a.force == "col":
        params["force_col_wise"] = True
    elif a.force == "row":
        params["force_row_wise"] = True
    rec["params"] = {k: v for k, v in params.items()}

    # --- explicit Dataset construction: isolates histogram bin-finding ---
    t = time.time()
    ds = lgb.Dataset(X, label=y, free_raw_data=False,
                     params={"max_bin": 255, "num_threads": a.n_threads})
    ds.construct()
    rec["dataset_construct_seconds"] = round(time.time() - t, 1)
    rec["dataset_num_features"] = int(ds.num_feature())
    del ds

    # --- training, exactly as E05 ran it ---
    log_path = BENCH / f"lgbm_log_{a.features}{a.tag}.txt"
    t = time.time()
    model = lgb.LGBMClassifier(**params, verbose=1)
    model.fit(X, y, feature_name=feat,
              callbacks=[lgb.log_evaluation(period=0)])
    rec["train_seconds"] = round(time.time() - t, 1)

    t = time.time()
    _ = model.predict_proba(X[:500_000])[:, 1]
    rec["score_500k_seconds"] = round(time.time() - t, 1)

    try:
        rec["actual_num_threads"] = model.booster_.params.get("num_threads", "default(all cores)")
    except Exception:
        rec["actual_num_threads"] = "unknown"
    rec["peak_rss_gb"] = round(peak_rss_gb(), 2)
    rec["swap_used_gb_after"] = round(swap_used_gb(), 2)
    rec["free_ram_gb_after"] = round(free_ram_gb(), 2)
    rec["n_trees"] = model.booster_.num_trees()
    rec["held_sides"] = bool(held)
    del held

    out = BENCH / f"bench_{a.features}{a.tag}.json"
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
