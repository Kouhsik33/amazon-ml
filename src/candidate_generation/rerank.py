"""Cheap lexical reranker over blocking candidates.

Blocking retrieves ~180 candidates per S1 at a macro-F0.5 ceiling of 0.978,
but truncating that list by key-agreement (``n_keys``) alone drops the ceiling
to 0.955 at K=40. ``n_keys`` is a coarse signal -- it counts how many key
families fired, not how similar the records actually are.

This module scores every surviving candidate with a handful of RapidFuzz
comparisons and re-ranks on that. It is deliberately *not* a model: it runs
before training data exists, it must run over every candidate of every query
(hundreds of millions at test scale), and its only job is to make the top-K
truncation lose as little recall as possible. The learned model (LightGBM)
scores the survivors.

Cost control: all string comparison goes through ``process.cpdist`` (C++,
multithreaded) and pairs are streamed in sub-chunks, because materialising
10M+ Python strings per column at once is what actually exhausts RAM here --
not the arithmetic.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

# Weights are deliberately blunt: name dominates, address disambiguates, a
# shared street number is strong corroboration, and key agreement breaks ties.
# They are validated by Recall@K, not fitted -- fitting them would need labels
# this stage does not have at test time.
W_NAME = 0.50
W_ADDR = 0.28
W_NUM = 0.14
W_KEYS = 0.08

CHUNK = 2_000_000


def _cp(a, b, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def rerank_score(df: pl.DataFrame) -> np.ndarray:
    """Score pairs already carrying q_/c_ normalised columns and ``n_keys``.

    Returns a float32 array in roughly [0, 1], higher = more likely a match.
    """
    n = df.height
    out = np.zeros(n, dtype=np.float32)

    # numeric overlap is a pure set operation -- vectorised, no string cost
    num = df.select(
        pl.col("q_ad_num").str.split(" ").list.eval(
            pl.element().filter(pl.element() != "")).alias("a"),
        pl.col("c_ad_num").str.split(" ").list.eval(
            pl.element().filter(pl.element() != "")).alias("b"),
    ).with_columns(
        pl.col("a").list.set_intersection(pl.col("b")).list.len().alias("ni"),
        pl.col("a").list.len().alias("na"),
        pl.col("b").list.len().alias("nb"),
    )
    ni = num["ni"].to_numpy().astype(np.float32)
    na = num["na"].to_numpy().astype(np.float32)
    nb = num["nb"].to_numpy().astype(np.float32)
    num_score = ni / np.maximum(np.minimum(na, nb), 1.0)
    # a pair where neither side has a number gets a neutral score, not a zero:
    # ~3% of candidates have no address at all and must not be buried.
    num_score = np.where((na == 0) | (nb == 0), 0.5, num_score)
    del num

    keys = df["n_keys"].to_numpy().astype(np.float32)
    keys = np.minimum(keys / 4.0, 1.0)

    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        sl = df[lo:hi]
        q_cmp = sl["q_nm_compact"].to_list()
        c_cmp = sl["c_nm_compact"].to_list()
        q_core = sl["q_nm_core"].to_list()
        c_core = sl["c_nm_core"].to_list()
        q_ad = sl["q_ad_core"].to_list()
        c_ad = sl["c_ad_core"].to_list()

        # order-sensitive and order-insensitive views of the name; the
        # generator transposes tokens often enough that either alone is fragile
        nm = np.maximum(_cp(q_cmp, c_cmp, fuzz.ratio),
                        _cp(q_core, c_core, fuzz.token_set_ratio)) * 0.01
        ad = _cp(q_ad, c_ad, fuzz.token_set_ratio) * 0.01

        out[lo:hi] = (W_NAME * nm + W_ADDR * ad
                      + W_NUM * num_score[lo:hi] + W_KEYS * keys[lo:hi])
        del q_cmp, c_cmp, q_core, c_core, q_ad, c_ad, nm, ad

    return out


def add_rerank(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.Series("rr_score", rerank_score(df)))


def top_k(df: pl.DataFrame, k: int, by: str = "rr_score") -> pl.DataFrame:
    """Keep the k best candidates per S1 entity."""
    return (df.sort(by, descending=True)
              .group_by("source1_entity_id", maintain_order=True)
              .head(k))
