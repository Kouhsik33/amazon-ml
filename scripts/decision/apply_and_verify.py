#!/usr/bin/env python3
"""Apply the final decision rule and score it with the REFERENCE scorer.

The fast NumPy engine is used for sweeps; this script independently
reconstructs the prediction sets from the Parquet artifacts and hands them to
src/evaluation/metrics.evaluate_predictions -- the same implementation used in
every earlier experiment -- so the reported number does not depend on the
sweep code being correct.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.io import load_ground_truth  # noqa: E402
from data.splits import load_split  # noqa: E402
from evaluation.metrics import evaluate_predictions  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--rule", required=True)
    a = ap.parse_args()
    r = json.loads(Path(a.rule).read_text())

    ids = load_split(a.fold)
    lf = (pl.scan_parquet(a.scored).select("source1_entity_id", "cand_id", "p")
          .join(pl.scan_parquet(a.cand).select("source1_entity_id", "cand_id", "rr_score"),
                on=["source1_entity_id", "cand_id"], how="left"))
    lf = lf.with_columns(
        pl.col("p").max().over("source1_entity_id").alias("best"),
        pl.when(pl.col("cand_id").str.starts_with("S2-"))
          .then(pl.lit(r["t_s2"])).otherwise(pl.lit(r["t_s3"])).alias("thr"),
    )
    base = (pl.col("p") >= pl.col("thr")) & (pl.col("p") >= pl.col("best") * r["rel_margin"])
    over = (pl.col("rr_score").fill_null(0.0) >= r["override_rr"]) & (pl.col("p") >= r["override_p"])
    keep = lf.filter(base | over).select("source1_entity_id", "cand_id", "p")
    df = keep.collect(streaming=True)
    del lf, keep
    gc.collect()

    if r.get("max_k"):
        df = (df.sort("p", descending=True)
                .group_by("source1_entity_id", maintain_order=True).head(r["max_k"]))
    g = df.group_by("source1_entity_id").agg(pl.col("cand_id"))
    pred = {row[0]: set(row[1]) for row in g.iter_rows()}
    del df, g
    gc.collect()

    gt = load_ground_truth().filter(pl.col("source1_entity_id").is_in(ids))
    truth = {row[0]: ({t for t in row[1].split(",") if t} if row[1] else set())
             for row in gt.iter_rows()}
    del gt
    gc.collect()

    res = evaluate_predictions(truth, pred, entity_universe=ids)
    s2 = evaluate_predictions({k: {x for x in v if x.startswith("S2-")} for k, v in truth.items()},
                              {k: {x for x in v if x.startswith("S2-")} for k, v in pred.items()},
                              entity_universe=ids)
    s3 = evaluate_predictions({k: {x for x in v if x.startswith("S3-")} for k, v in truth.items()},
                              {k: {x for x in v if x.startswith("S3-")} for k, v in pred.items()},
                              entity_universe=ids)
    print(f"=== {a.fold.upper()} — REFERENCE SCORER — rule {r} ===")
    print(res)
    print(f"\n  S2-only macro F0.5 : {s2.macro_f05:.6f}")
    print(f"  S3-only macro F0.5 : {s3.macro_f05:.6f}")
    out = {"fold": a.fold, "rule": r, "macro_f05": res.macro_f05,
           "mean_precision": res.mean_precision, "mean_recall": res.mean_recall,
           "singleton_accuracy": res.singleton_accuracy,
           "s2_f05": s2.macro_f05, "s3_f05": s3.macro_f05,
           "FP": res.false_positives, "FN": res.false_negatives,
           "TP": res.true_positives, "n_predicted": res.n_predicted_matches}
    Path(f"experiments/decision/final_{a.fold}_reference.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
