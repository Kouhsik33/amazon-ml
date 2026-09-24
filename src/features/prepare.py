"""Precompute normalised views for every record and cache them as Parquet.

Normalisation is Python-level (romanisation, alias lookup, ordinal folding) and
costs ~47 CPU-minutes over the 26M records in the challenge. Every experiment
needs it, so it is paid once here, in parallel, and cached. Downstream stages
read only Parquet and stay vectorised.

Columns produced per record:

    nm_full     normalised name, all tokens
    nm_core     name minus mined noise tokens (legal suffixes, filler)
    nm_compact  nm_core with spaces removed -- matches 'carneybryant.com'
    nm_skel     consonant skeleton of nm_compact -- vowel typos, transliteration
    ad_full     normalised address, all tokens
    ad_core     address minus mined noise tokens
    ad_num      space-joined numeric address tokens (street no., PIN, unit)
    ad_alpha    space-joined non-numeric address core tokens
    ctry        normalised country label
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.io import CACHE, load  # noqa: E402

CHUNK = 250_000


def _work(args):
    names, addrs, countries = args
    # imported inside the worker so each process builds its own caches
    from features.normalize import (normalize_address, normalize_country,
                                    normalize_name)
    from features.translit import consonant_skeleton

    out = [[] for _ in range(9)]
    for nm, ad, co in zip(names, addrs, countries):
        n = normalize_name(nm)
        a = normalize_address(ad)
        out[0].append(n.full)
        out[1].append(n.core)
        out[2].append(n.compact)
        out[3].append(consonant_skeleton(n.compact))
        out[4].append(a.full)
        out[5].append(a.core)
        out[6].append(" ".join(a.numbers))
        out[7].append(" ".join(a.alpha_tokens))
        out[8].append(normalize_country(co))
    return out


COLS = ["nm_full", "nm_core", "nm_compact", "nm_skel",
        "ad_full", "ad_core", "ad_num", "ad_alpha", "ctry"]


def prepare(split: str, src: str, workers: int, force: bool = False) -> Path:
    out = CACHE / f"norm_{split}_{src}.parquet"
    if out.exists() and not force:
        print(f"{out.name} exists, skipping")
        return out
    df = load(split, src)
    n = df.height
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    ctry = df["country"].to_list()

    chunks = [(names[i:i + CHUNK], addrs[i:i + CHUNK], ctry[i:i + CHUNK])
              for i in range(0, n, CHUNK)]
    acc = [[] for _ in range(9)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, res in enumerate(pool.map(_work, chunks), 1):
            for i in range(9):
                acc[i].extend(res[i])
            print(f"  {out.name}: {min(k * CHUNK, n):,}/{n:,}", end="\r", flush=True)

    res = pl.DataFrame({"entity_id": df["entity_id"], **{c: acc[i] for i, c in enumerate(COLS)}})
    res.write_parquet(out, compression="zstd")
    print(f"\nwrote {out.name}: {res.height:,} rows")
    return out


def load_norm(split: str, src: str, derive: bool = True) -> pl.DataFrame:
    p = CACHE / f"norm_{split}_{src}.parquet"
    if not p.exists():
        prepare(split, src, workers=8)
    df = pl.read_parquet(p)
    return derive_views(df) if derive else df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="e.g. train:source1")
    a = ap.parse_args()
    targets = ([tuple(a.only.split(":"))] if a.only else
               [(s, f"source{i}") for s in ("train", "test") for i in (1, 2, 3)])
    for split, src in targets:
        prepare(split, src, a.workers, a.force)


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------
# nm_full / ad_full / ad_num depend only on the alias tables; nm_core,
# nm_compact, nm_skel, ad_core and ad_alpha depend additionally on the *noise*
# tables. Deriving the latter group in Polars rather than baking it into the
# cache makes the noise vocabulary a cheap hyperparameter -- it can be changed
# and re-measured in seconds instead of re-normalising 26M records for 45
# minutes.

def derive_views(df: pl.DataFrame) -> pl.DataFrame:
    """Recompute the noise-dependent views from the cached full forms."""
    import json
    from features.normalize import CONFIG_DIR

    nm_noise = json.loads((CONFIG_DIR / "name_noise.json").read_text())
    ad_noise = json.loads((CONFIG_DIR / "addr_noise.json").read_text())

    def strip(col: str, noise: list[str]) -> pl.Expr:
        toks = pl.col(col).str.split(" ").list.eval(
            pl.element().filter(pl.element() != ""))
        kept = toks.list.set_difference(pl.lit(noise, dtype=pl.List(pl.Utf8)))
        # a name made only of legal words keeps its original tokens
        return pl.when(kept.list.len() > 0).then(kept).otherwise(toks)

    df = df.with_columns(
        strip("nm_full", nm_noise).alias("_nt"),
        strip("ad_full", ad_noise).alias("_at"),
    )
    return df.with_columns(
        pl.col("_nt").list.join(" ").alias("nm_core"),
        pl.col("_at").list.join(" ").alias("ad_core"),
        pl.col("_at").list.eval(
            pl.element().filter(~pl.element().str.contains(r"\d"))
        ).list.join(" ").alias("ad_alpha"),
        # '07811' and '7811' are the same street number; the generator pads and
        # unpads freely, so leading zeros are stripped before any number key.
        pl.col("ad_num").str.split(" ").list.eval(
            pl.element().str.replace(r"^0+", "").filter(pl.element() != "")
        ).list.unique().list.join(" ").alias("ad_num"),
    ).with_columns(
        pl.col("nm_core").str.replace_all(" ", "").alias("nm_compact"),
    ).with_columns(
        pl.col("nm_compact").str.replace_all("[aeiou]", "").alias("nm_skel"),
    ).drop("_nt", "_at")
