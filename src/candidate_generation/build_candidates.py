"""Blocking -> rerank -> top-K, cached to Parquet.

Structured around one constraint: building the pool-side blocking index costs
~90s and ~600MB per country, so it must be built **once** and reused across
query batches, not rebuilt per call. Queries stream through in batches; each
batch is reranked and truncated to top-K immediately, so peak memory is set by
the batch, not by the total candidate count.

This is the same code path used for validation folds and (later) for the test
set -- only the query id list changes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_generation.blocking import (BlockingConfig, _build_keys,  # noqa: E402
                                           _token_df)
from candidate_generation.rerank import add_rerank, top_k  # noqa: E402
from data.io import CACHE  # noqa: E402
from features.pair_features import Q_COLS  # noqa: E402
from features.prepare import load_norm  # noqa: E402

CAND_DIR = CACHE / "candidates"

# The reranker needs only these four views per side. Joining all nine (as the
# feature stage does) was the memory spike: 18 wide string columns over a
# multi-million-row batch, most of them never read.
RERANK_COLS = ["nm_core", "nm_compact", "ad_core", "ad_num"]


def _attach_lean(pairs: pl.DataFrame, q: pl.DataFrame, c: pl.DataFrame) -> pl.DataFrame:
    qq = q.select([pl.col("entity_id").alias("source1_entity_id")]
                  + [pl.col(x).alias(f"q_{x}") for x in RERANK_COLS])
    cc = c.select([pl.col("entity_id").alias("cand_id")]
                  + [pl.col(x).alias(f"c_{x}") for x in RERANK_COLS])
    return pairs.join(qq, on="source1_entity_id", how="inner") \
                .join(cc, on="cand_id", how="inner")


def _batches(df: pl.DataFrame, size: int) -> Iterator[pl.DataFrame]:
    for lo in range(0, df.height, size):
        yield df[lo:lo + size]


def build(query_ids: list[str], split: str, cfg: BlockingConfig, k: int,
          out: Path, batch_size: int = 10_000, verbose: bool = True) -> pl.DataFrame:
    s1 = load_norm(split, "source1").filter(pl.col("entity_id").is_in(query_ids))
    pool = pl.concat([load_norm(split, "source2"), load_norm(split, "source3")])

    key_cols = ["entity_id", "nm_core", "nm_compact", "nm_skel", "ad_num", "ad_alpha"]
    chunks: list[pl.DataFrame] = []
    t0 = time.time()

    for country in pool["ctry"].unique().to_list():
        p_all = pool.filter(pl.col("ctry") == country)
        q_all = s1.filter(pl.col("ctry") == country)
        if q_all.height == 0 or p_all.height == 0:
            continue

        # ---- pool index: built once per country ----
        p_idx = p_all.select(key_cols).with_row_index("ci")
        rare = _token_df(p_idx)
        pk = _build_keys(p_idx, "ci", cfg, rare)
        post = pk.group_by("k").len().filter(pl.col("len") <= cfg.max_posting)
        pk = pk.join(post.select("k"), on="k", how="inner")
        if verbose:
            print(f"  [{country}] pool={p_all.height:,} index={pk.height:,} keys "
                  f"({time.time() - t0:.0f}s)", flush=True)

        p_side = p_all.select(["entity_id"] + RERANK_COLS)
        p_map = p_idx.select("ci", "entity_id").rename({"entity_id": "cand_id"})

        for b, qb in enumerate(_batches(q_all, batch_size)):
            qi = qb.select(key_cols).with_row_index("qi")
            qk = _build_keys(qi, "qi", cfg, rare)
            pairs = (qk.join(pk, on="k", how="inner")
                       .group_by("qi", "ci").len().rename({"len": "n_keys"}))
            pairs = (pairs
                     .join(qi.select("qi", pl.col("entity_id").alias("source1_entity_id")), on="qi")
                     .join(p_map, on="ci")
                     .select("source1_entity_id", "cand_id", "n_keys"))
            # rerank needs both sides' normalised text
            pairs = _attach_lean(pairs, qb, p_side)
            pairs = top_k(add_rerank(pairs), k)
            chunks.append(pairs.select("source1_entity_id", "cand_id", "n_keys", "rr_score"))
            if verbose:
                print(f"    batch {b}: {qb.height:,} queries -> "
                      f"{chunks[-1].height:,} kept ({time.time() - t0:.0f}s)", flush=True)
        del pk, p_idx, p_side, p_all

    res = pl.concat(chunks) if chunks else pl.DataFrame(
        schema={"source1_entity_id": pl.Utf8, "cand_id": pl.Utf8,
                "n_keys": pl.UInt32, "rr_score": pl.Float32})
    out.parent.mkdir(parents=True, exist_ok=True)
    res.write_parquet(out, compression="zstd")
    print(f"wrote {out.name}: {res.height:,} pairs for {len(query_ids):,} queries "
          f"({res.height / max(len(query_ids),1):.1f}/query) in {time.time() - t0:.0f}s")
    return res


if __name__ == "__main__":
    from data.splits import load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="dev|val|train")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="subsample the fold")
    ap.add_argument("--k", type=int, default=80)
    ap.add_argument("--max-posting", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    ids = load_split(a.fold)
    if a.limit and len(ids) > a.limit:
        import random
        random.Random(7).shuffle(ids)
        ids = ids[:a.limit]
    name = f"cand_{a.fold}{a.tag}_k{a.k}.parquet"
    build(ids, a.split, BlockingConfig(max_posting=a.max_posting, max_cand_per_s1=0),
          a.k, CAND_DIR / name, a.batch_size)
