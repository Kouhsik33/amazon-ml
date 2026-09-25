"""Score a cached candidate file with a saved LightGBM model."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.io import load_ground_truth  # noqa: E402
from models.dataset import add_context, label  # noqa: E402
from models.train_lgbm import featurize, load_sides  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labeled", action="store_true", help="join ground truth (validation folds)")
    a = ap.parse_args()

    t0 = time.time()
    cand = add_context(pl.read_parquet(a.cand))
    if a.labeled:
        cand = label(cand, load_ground_truth())
    sides = load_sides(a.split)
    X, _, meta = featurize(cand, a.split, sides=sides)
    booster = lgb.Booster(model_file=a.model)
    p = booster.predict(X).astype(np.float32)
    out = meta.with_columns(pl.Series("p", p))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out, compression="zstd")
    dt = time.time() - t0
    print(f"scored {out.height:,} pairs in {dt:.0f}s ({out.height/dt:,.0f} pairs/s) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
