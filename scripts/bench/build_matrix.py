#!/usr/bin/env python3
"""Build the E05 feature matrix ONCE and cache it, so the E04/E05 training
benchmark measures only training -- not feature generation.

E05's feature set is a strict superset of E04's (9 added, 0 removed), so the
E04 matrix is recovered exactly by column selection. That keeps the A/B
honest: identical pairs, identical rows, identical values, only the feature
columns differ.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.io import load_ground_truth  # noqa: E402
from models.dataset import add_context, label, sample_negatives  # noqa: E402
from models.train_lgbm import featurize, load_sides  # noqa: E402

OUT = Path("experiments/bench")
CAND = "data_cache/candidates/cand_train_200k_k80.parquet"


def peak_rss_gb() -> float:
    # macOS reports ru_maxrss in bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr = add_context(pl.read_parquet(CAND))
    tr = sample_negatives(label(tr, load_ground_truth()), 8, 4)
    t_pairs = time.time() - t0

    t1 = time.time()
    sides = load_sides("train")
    X, names, meta = featurize(tr, "train", sides=sides)
    t_feat = time.time() - t1

    y = meta["y"].to_numpy().astype(np.int8)
    np.save(OUT / "X_e05.npy", X)
    np.save(OUT / "y.npy", y)
    (OUT / "feature_names.json").write_text(json.dumps(names))

    # per-feature diagnostics: cardinality drives LightGBM bin construction,
    # and NaN/Inf change how it handles splits
    diag = []
    for i, n in enumerate(names):
        col = X[:, i]
        finite = np.isfinite(col)
        diag.append({
            "feature": n,
            "n_unique": int(np.unique(col[finite]).size),
            "n_nan": int(np.isnan(col).sum()),
            "n_inf": int(np.isinf(col).sum()),
            "min": float(col[finite].min()) if finite.any() else None,
            "max": float(col[finite].max()) if finite.any() else None,
        })
    (OUT / "feature_diagnostics.json").write_text(json.dumps(diag, indent=2))

    summary = {
        "n_pairs": int(X.shape[0]),
        "n_features_e05": int(X.shape[1]),
        "n_positive": int(y.sum()),
        "pos_rate": float(y.mean()),
        "matrix_gb": float(X.nbytes / 1e9),
        "dtype": str(X.dtype),
        "pair_build_seconds": round(t_pairs, 1),
        "feature_generation_seconds": round(t_feat, 1),
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    (OUT / "matrix_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
