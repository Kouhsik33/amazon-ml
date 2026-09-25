"""Pair-level training data with hard-negative mining, plus context features.

**Negatives.** Every negative here already survived blocking *and* ranked in
the top-80 of the lexical reranker, so the pool is hard by construction -- a
randomly drawn S2 record from the 10.3M pool would be trivially separable and
teach the model nothing. Within that pool we still stratify:

* ``n_hard`` highest-``rr_score`` non-matches per entity -- the records that
  actually generate false positives (E00 measured 5.12% of orphan records
  colliding *exactly* on normalised name with a real S1 entity);
* ``n_rand`` uniformly sampled from the remainder -- without these the model
  only ever sees the top of the ranking and mis-calibrates on the easy
  candidates it must also score at inference, where all ~63 are fed in.

**Context features.** A pair is not judged in isolation: what matters for
F0.5 is whether *this* candidate is the best explanation for this S1 entity.
Rank, margin to the best candidate, and the best/second-best gap give the
model that, and they are what the singleton logic later keys on. They are
computed over the **full** candidate list before any subsampling, so training
and inference see identical values.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import polars as pl

CTX_COLS = ["rr_score", "rr_rank", "rr_best", "rr_margin", "rr_rel",
            "rr_second", "rr_top_gap", "n_cand", "n_keys"]


def add_context(cand: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 ranking context. Computed on the full candidate list."""
    g = "source1_entity_id"
    out = cand.with_columns(
        pl.col("rr_score").rank("ordinal", descending=True).over(g).alias("rr_rank"),
        pl.col("rr_score").max().over(g).alias("rr_best"),
        pl.len().over(g).alias("n_cand"),
    )
    # second-best score per entity: max of the scores strictly below the best
    second = (out.sort("rr_score", descending=True)
                 .group_by(g, maintain_order=True)
                 .agg(pl.col("rr_score").slice(1, 1).first().alias("rr_second")))
    out = out.join(second, on=g, how="left").with_columns(
        pl.col("rr_second").fill_null(0.0))
    return out.with_columns(
        (pl.col("rr_score") - pl.col("rr_best")).alias("rr_margin"),
        (pl.col("rr_score") / pl.col("rr_best").clip(lower_bound=1e-6)).alias("rr_rel"),
        (pl.col("rr_best") - pl.col("rr_second")).alias("rr_top_gap"),
    )


def label(cand: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    truth = (gt.filter(pl.col("matched_entity_ids") != "")
               .with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids")
               .rename({"matched_entity_ids": "cand_id"})
               .with_columns(pl.lit(1, dtype=pl.Int8).alias("y")))
    return (cand.join(truth, on=["source1_entity_id", "cand_id"], how="left")
                .with_columns(pl.col("y").fill_null(0)))


def sample_negatives(labeled: pl.DataFrame, n_hard: int = 8, n_rand: int = 4,
                     seed: int = 17) -> pl.DataFrame:
    pos = labeled.filter(pl.col("y") == 1)
    neg = labeled.filter(pl.col("y") == 0)

    hard = (neg.sort("rr_score", descending=True)
               .group_by("source1_entity_id", maintain_order=True)
               .head(n_hard))
    rest = neg.join(hard.select("source1_entity_id", "cand_id"),
                    on=["source1_entity_id", "cand_id"], how="anti")
    rnd = (rest.with_columns(pl.col("cand_id").hash(seed=seed).alias("_h"))
               .sort("_h")
               .group_by("source1_entity_id", maintain_order=True)
               .head(n_rand).drop("_h"))
    out = pl.concat([pos, hard, rnd])
    print(f"  train pairs: {pos.height:,} pos + {hard.height:,} hard-neg + "
          f"{rnd.height:,} rand-neg = {out.height:,} "
          f"(pos rate {pos.height / out.height:.3f})")
    return out


def context_matrix(df: pl.DataFrame) -> Tuple[np.ndarray, List[str]]:
    return (df.select(CTX_COLS).to_numpy().astype(np.float32), list(CTX_COLS))
