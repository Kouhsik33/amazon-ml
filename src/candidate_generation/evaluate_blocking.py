"""Measure candidate-generation quality on a held-out fold.

Reports, besides the usual recall / reduction ratio:

* **macro-F0.5 ceiling** -- the score a *perfect* matcher would get given this
  candidate set: for each S1, F0.5 with precision 1 and recall
  ``captured/true`` (singletons score 1.0). This is the number that matters,
  because the competition metric is macro-averaged per entity, so losing one
  match on a 2-match entity costs far more than losing one on a 6-match entity.
  Pair-level recall hides that.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_generation.blocking import BlockingConfig, generate_candidates  # noqa: E402
from data.io import load_ground_truth  # noqa: E402
from data.splits import load_split  # noqa: E402
from evaluation.metrics import fbeta  # noqa: E402
from features.prepare import load_norm  # noqa: E402


def evaluate(pairs: pl.DataFrame, gt: pl.DataFrame, n_pool: int,
             query_ids: list[str]) -> dict:
    truth = (gt.filter(pl.col("matched_entity_ids") != "")
               .with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids")
               .rename({"matched_entity_ids": "cand_id"}))

    n_true = truth.height
    hit = truth.join(pairs.with_columns(pl.lit(1).alias("h")),
                     on=["source1_entity_id", "cand_id"], how="left")
    hit = hit.with_columns(pl.col("h").fill_null(0))
    n_hit = int(hit["h"].sum())

    per_src = (hit.with_columns(pl.col("cand_id").str.slice(0, 2).alias("src"))
                  .group_by("src").agg(pl.col("h").mean().alias("recall"),
                                       pl.len().alias("n")).sort("src"))

    # per-entity capture -> macro F0.5 ceiling
    cap = hit.group_by("source1_entity_id").agg(
        pl.col("h").sum().alias("got"), pl.len().alias("tot"))
    cap_map = dict(zip(cap["source1_entity_id"].to_list(),
                       (cap["got"] / cap["tot"]).to_list()))
    n_singleton = 0
    ceil_sum = 0.0
    full_capture = 0
    for qid in query_ids:
        r = cap_map.get(qid)
        if r is None:                       # entity has no true matches
            n_singleton += 1
            ceil_sum += 1.0
            full_capture += 1
        else:
            ceil_sum += fbeta(1.0, r)
            full_capture += int(r >= 1.0)

    cnt = pairs.group_by("source1_entity_id").len()["len"]
    per_q = cnt.to_numpy() if cnt.len() else None
    n_q = len(query_ids)
    zero_cand = n_q - pairs["source1_entity_id"].n_unique()

    import numpy as np
    padded = np.zeros(n_q, dtype=np.int64)
    if per_q is not None:
        padded[:len(per_q)] = per_q

    total_possible = n_q * n_pool
    return {
        "n_queries": n_q,
        "n_pairs": pairs.height,
        "pair_recall": n_hit / n_true if n_true else float("nan"),
        "n_true_pairs": n_true,
        "macro_f05_ceiling": ceil_sum / n_q,
        "entities_fully_captured": full_capture / n_q,
        "entities_zero_candidates": zero_cand / n_q,
        "avg_cand_per_s1": pairs.height / n_q,
        "p50": float(np.percentile(padded, 50)),
        "p95": float(np.percentile(padded, 95)),
        "p99": float(np.percentile(padded, 99)),
        "max": int(padded.max()),
        "reduction_ratio": 1.0 - pairs.height / total_possible,
        "recall_by_source": {r[0]: (r[1], r[2]) for r in per_src.iter_rows()},
        "n_singletons": n_singleton,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="dev")
    ap.add_argument("--rare-df-max", type=int, default=4000)
    ap.add_argument("--max-posting", type=int, default=400)
    ap.add_argument("--disable", default="", help="comma list, e.g. b4,b5")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sweep", default="10,20,30,40,60,0",
                    help="comma list of per-S1 caps to evaluate (0 = uncapped)")
    a = ap.parse_args()

    # generate uncapped once, then evaluate every cap by truncating on the
    # key-agreement rank -- one blocking pass yields the whole trade-off curve.
    cfg = BlockingConfig(rare_df_max=a.rare_df_max, max_posting=a.max_posting,
                         max_cand_per_s1=0)
    for d in filter(None, a.disable.split(",")):
        setattr(cfg, f"use_{d.strip()}", False)

    ids = load_split(a.fold)
    s1 = load_norm("train", "source1").filter(pl.col("entity_id").is_in(ids))
    pool = pl.concat([load_norm("train", "source2"), load_norm("train", "source3")])
    gt = load_ground_truth().filter(pl.col("source1_entity_id").is_in(ids))

    print(f"fold={a.fold} queries={s1.height:,} pool={pool.height:,} cfg={cfg}")
    t0 = time.time()
    pairs = generate_candidates(s1, pool, cfg)
    t_block = time.time() - t0

    print(f"\nblocking took {t_block:.1f}s -> {pairs.height:,} raw pairs")
    ranked = (pairs.sort("n_keys", descending=True)
                   .with_columns(pl.int_range(pl.len()).over("source1_entity_id").alias("rk")))
    for cap in [int(x) for x in a.sweep.split(",")]:
        sub = ranked if cap == 0 else ranked.filter(pl.col("rk") < cap)
        res = evaluate(sub, gt, pool.height, ids)
        res["cap"] = cap or "uncapped"
        res["blocking_seconds"] = round(t_block, 1)
        print(f"\n--- cap={res['cap']} ---")
        for k, v in res.items():
            print(f"  {k:26s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
