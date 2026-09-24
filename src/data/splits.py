"""Validation protocol.

The competition task is: *given an unseen Source-1 entity and the complete
Source-2/3 pools, return its matches.* The validation split must reproduce
exactly that, so the unit of splitting is the **Source-1 entity**, never the
pair and never the S2/S3 record.

    held-out S1 entities  x  the FULL 10.3M S2/S3 pool

Keeping the whole pool is the important part. Shrinking it to "only the S2/S3
records that belong to validation entities" would delete every distractor and
every competing entity, inflating both blocking recall and precision by a wide
margin -- the model would never face the 1.34M unmatched S2 records or the
near-duplicate businesses that generate the real false positives.

Three disjoint folds:

* ``train`` -- model fitting, hard-negative mining, alias/noise table mining.
* ``dev``   -- threshold search, calibration, decision-rule selection.
* ``val``   -- touched once per architecture for an unbiased estimate.

Separating ``dev`` from ``val`` matters here because the decision layer has
many knobs (global/per-source thresholds, margins, top-k) and F_0.5 is sharp
in them; tuning and reporting on the same fold would overstate the score.

Leakage control:

* alias/noise mining takes ``--exclude-s1`` so no held-out entity's ground
  truth shapes normalisation;
* IDF/TF-IDF statistics are unsupervised corpus counts, computed over all
  records exactly as they will be at test time;
* the many-to-one "already claimed" constraint is derived at inference from
  model scores only -- never from ground-truth assignments.

A second, harder split answers the France question. Test contains a country
absent from training, and no amount of in-distribution validation measures
that. ``country_holdout`` trains on one country and evaluates on the other,
giving an honest lower bound for transfer to an unseen country.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.io import load, load_ground_truth, CACHE  # noqa: E402

SPLIT_DIR = CACHE / "splits"
SEED = 20260925
N_DEV = 40_000
N_VAL = 60_000


def _n_matches(gt: pl.DataFrame) -> pl.DataFrame:
    return gt.with_columns(
        pl.when(pl.col("matched_entity_ids") == "")
        .then(0)
        .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)
        .alias("n_match")
    )


def make_splits(force: bool = False) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    if (SPLIT_DIR / "val.txt").exists() and not force:
        print("splits already exist; use --force to rebuild")
        return

    s1 = load("train", "source1").select("entity_id", "country")
    gt = _n_matches(load_ground_truth())
    df = s1.join(gt, left_on="entity_id", right_on="source1_entity_id", how="left")

    # Stratify on (country, match-count bucket) so the singleton rate and the
    # one-to-many shape of each fold mirror the full distribution.
    df = df.with_columns(
        pl.when(pl.col("n_match") == 0).then(pl.lit("0"))
        .when(pl.col("n_match") <= 2).then(pl.lit("1-2"))
        .when(pl.col("n_match") <= 4).then(pl.lit("3-4"))
        .otherwise(pl.lit("5+")).alias("bucket")
    )

    # Deterministic per-entity uniform, then rank inside each stratum so the
    # fold proportions hold exactly within every (country, bucket) cell.
    df = df.with_columns(
        (pl.col("entity_id").hash(seed=SEED) % 1_000_000_007).alias("h")
    )
    total = df.height
    p_dev = N_DEV / total
    p_val = N_VAL / total
    rng = df.with_columns(
        ((pl.col("h").rank("ordinal").over(["country", "bucket"]) - 1)
         / pl.len().over(["country", "bucket"])).alias("frac")
    )
    fold = (
        pl.when(pl.col("frac") < p_dev).then(pl.lit("dev"))
        .when(pl.col("frac") < p_dev + p_val).then(pl.lit("val"))
        .otherwise(pl.lit("train"))
    )
    rng = rng.with_columns(fold.alias("fold"))

    for name in ("train", "dev", "val"):
        ids = rng.filter(pl.col("fold") == name).select("entity_id")
        ids.write_csv(SPLIT_DIR / f"{name}.txt", include_header=False)

    summary = (
        rng.group_by("fold")
        .agg(
            pl.len().alias("n"),
            (pl.col("n_match") == 0).mean().alias("singleton_rate"),
            pl.col("n_match").mean().alias("mean_matches"),
            (pl.col("country") == "US").mean().alias("frac_us"),
        )
        .sort("fold")
    )
    print(summary)

    # cross-country generalisation folds (proxy for the unseen France)
    for c in df.select("country").unique().to_series().to_list():
        ids = df.filter(pl.col("country") == c).select("entity_id")
        ids.write_csv(SPLIT_DIR / f"country_{c}.txt", include_header=False)
        print(f"country fold {c}: {ids.height:,}")

    (SPLIT_DIR / "meta.json").write_text(json.dumps(
        {"seed": SEED, "n_dev": N_DEV, "n_val": N_VAL,
         "policy": "S1-entity holdout against the full S2/S3 pool"}, indent=2))


def load_split(name: str) -> list[str]:
    p = SPLIT_DIR / f"{name}.txt"
    if not p.exists():
        make_splits()
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def split_ground_truth(name: str) -> dict[str, set[str]]:
    """Ground truth restricted to one fold, as {s1_id: set(matched ids)}."""
    ids = set(load_split(name))
    gt = load_ground_truth().filter(pl.col("source1_entity_id").is_in(list(ids)))
    return {
        r[0]: ({t for t in r[1].split(",") if t} if r[1] else set())
        for r in gt.iter_rows()
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    make_splits(force=a.force)
