"""Recall@K for the lexical reranker, against the key-agreement baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.io import load_ground_truth  # noqa: E402
from data.splits import load_split  # noqa: E402
from evaluation.metrics import fbeta  # noqa: E402


def recall_at_k(cand: pl.DataFrame, truth: pl.DataFrame, ids: list[str],
                k: int, by: str) -> dict:
    ranked = (cand.sort(by, descending=True)
                  .with_columns(pl.int_range(pl.len()).over("source1_entity_id").alias("rk")))
    sub = ranked.filter(pl.col("rk") < k) if k else ranked
    hit = (truth.join(sub.select("source1_entity_id", "cand_id").with_columns(pl.lit(1).alias("h")),
                      on=["source1_entity_id", "cand_id"], how="left")
                .with_columns(pl.col("h").fill_null(0)))
    cap = hit.group_by("source1_entity_id").agg(
        pl.col("h").sum().alias("got"), pl.len().alias("tot"))
    cm = dict(zip(cap["source1_entity_id"].to_list(), (cap["got"] / cap["tot"]).to_list()))
    ceil = sum(1.0 if (r := cm.get(q)) is None else fbeta(1.0, r) for q in ids) / len(ids)
    per_src = (hit.with_columns(pl.col("cand_id").str.slice(0, 2).alias("src"))
                  .group_by("src").agg(pl.col("h").mean().alias("r")).sort("src"))
    return {
        "k": k or "all",
        "by": by,
        "pair_recall": float(hit["h"].mean()),
        "macro_f05_ceiling": ceil,
        "avg_cand": sub.height / len(ids),
        "s2_recall": dict(per_src.iter_rows()).get("S2"),
        "s3_recall": dict(per_src.iter_rows()).get("S3"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--fold", default="dev")
    ap.add_argument("--ks", default="10,20,40,80,0")
    a = ap.parse_args()

    ids = load_split(a.fold)
    cand = pl.read_parquet(a.cand)
    present = set(cand["source1_entity_id"].unique().to_list())
    ids = [i for i in ids if i in present] or ids
    gt = load_ground_truth().filter(pl.col("source1_entity_id").is_in(ids))
    truth = (gt.filter(pl.col("matched_entity_ids") != "")
               .with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids").rename({"matched_entity_ids": "cand_id"}))

    print(f"fold={a.fold} queries={len(ids):,} candidates={cand.height:,} "
          f"true_pairs={truth.height:,}\n")
    hdr = f"{'rank by':>10s} {'K':>5s} {'pair_recall':>12s} {'macroF0.5ceil':>14s} {'avg_cand':>9s} {'S2':>7s} {'S3':>7s}"
    print(hdr); print("-" * len(hdr))
    for by in ("n_keys", "rr_score"):
        for k in [int(x) for x in a.ks.split(",")]:
            r = recall_at_k(cand, truth, ids, k, by)
            print(f"{by:>10s} {str(r['k']):>5s} {r['pair_recall']:>12.4f} "
                  f"{r['macro_f05_ceiling']:>14.4f} {r['avg_cand']:>9.1f} "
                  f"{r['s2_recall']:>7.4f} {r['s3_recall']:>7.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
