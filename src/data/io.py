"""Data loading: TSV -> Parquet conversion and cached readers.

The challenge ships 26M rows across 7 TSV files (~2.5 GB). Parquet cuts load
time from minutes to seconds, which matters because every experiment re-reads
the sources.
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "student_resource" / "dataset"
CACHE = Path(os.environ.get("ER_CACHE", REPO / "data_cache"))

SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}


def _tsv(split: str, name: str) -> Path:
    return RAW / split / f"{split}_{name}.tsv"


def convert_all() -> None:
    """Materialise every TSV as Parquet under CACHE. Idempotent."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for src in ("source1", "source2", "source3"):
            out = CACHE / f"{split}_{src}.parquet"
            if out.exists():
                continue
            df = pl.read_csv(
                _tsv(split, src),
                separator="\t",
                quote_char=None,          # addresses contain bare quotes
                schema_overrides=SCHEMA,
                infer_schema_length=0,
            ).with_columns(
                pl.col("business_name").fill_null(""),
                pl.col("business_address").fill_null(""),
                pl.col("country").fill_null(""),
            )
            df.write_parquet(out, compression="zstd")
            print(f"wrote {out.name}: {df.height:,} rows")

    out = CACHE / "train_ground_truth.parquet"
    if not out.exists():
        gt = pl.read_csv(
            _tsv("train", "ground_truth"),
            separator="\t",
            quote_char=None,
            schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
            infer_schema_length=0,
        ).with_columns(pl.col("matched_entity_ids").fill_null(""))
        gt.write_parquet(out, compression="zstd")
        print(f"wrote {out.name}: {gt.height:,} rows")


def load(split: str, src: str) -> pl.DataFrame:
    """Load one source table ('source1'|'source2'|'source3')."""
    p = CACHE / f"{split}_{src}.parquet"
    if not p.exists():
        convert_all()
    return pl.read_parquet(p)


def load_ground_truth() -> pl.DataFrame:
    """Ground truth as (source1_entity_id, matched_entity_ids:str)."""
    p = CACHE / "train_ground_truth.parquet"
    if not p.exists():
        convert_all()
    return pl.read_parquet(p)


def gt_exploded() -> pl.DataFrame:
    """Ground truth as one row per (s1, matched_id) pair. Singletons dropped."""
    return (
        load_ground_truth()
        .filter(pl.col("matched_entity_ids") != "")
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .rename({"matched_entity_ids": "cand_id"})
    )


if __name__ == "__main__":
    convert_all()
