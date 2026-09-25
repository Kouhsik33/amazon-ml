"""Score a model's candidate scores end-to-end with the official metric.

Threshold search runs on ``dev``; ``val`` is scored once at the chosen setting,
so the headline number is not the maximum of a search.

Source-wise reporting restricts *both* the ground truth and the predictions to
one prefix and re-runs the macro average, which answers "how well do we do on
S2 entities" rather than "what fraction of our S2 predictions are right".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.io import load_ground_truth  # noqa: E402
from data.splits import load_split  # noqa: E402
from decision.decide import DecisionParams, apply, evaluate, search, to_prediction_dict  # noqa: E402
from evaluation.metrics import evaluate_predictions  # noqa: E402


def truth_tables(ids: list[str]):
    gt = load_ground_truth().filter(pl.col("source1_entity_id").is_in(ids))
    truth = (gt.filter(pl.col("matched_entity_ids") != "")
               .with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids")
               .rename({"matched_entity_ids": "cand_id"}))
    counts = dict(truth.group_by("source1_entity_id").len().iter_rows())
    hit = truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    gt_dict = {r[0]: ({t for t in r[1].split(",") if t} if r[1] else set())
               for r in gt.iter_rows()}
    return counts, hit, gt_dict


def per_source(gt_dict, pred_dict, ids, prefix):
    g = {k: {x for x in v if x.startswith(prefix)} for k, v in gt_dict.items()}
    p = {k: {x for x in v if x.startswith(prefix)} for k, v in pred_dict.items()}
    return evaluate_predictions(g, p, entity_universe=ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--fold", default="dev")
    ap.add_argument("--params", default=None, help="JSON of DecisionParams; skips search")
    ap.add_argument("--save-params", default=None)
    ap.add_argument("--experiment-id", default="E04")
    ap.add_argument("--notes", default="")
    ap.add_argument("--ceiling", type=float, default=0.9719)
    a = ap.parse_args()

    t0 = time.time()
    ids = load_split(a.fold)
    scored = pl.read_parquet(a.scored)
    counts, hit, gt_dict = truth_tables(ids)
    print(f"fold={a.fold} entities={len(ids):,} scored_pairs={scored.height:,} "
          f"singletons={sum(1 for i in ids if counts.get(i,0)==0):,}")

    if a.params:
        p = DecisionParams(**json.loads(Path(a.params).read_text()))
        m = evaluate(scored, counts, hit, ids, p)
        print(f"\nusing fixed params {p.as_dict()}")
    else:
        print("\nthreshold search (macro F0.5):")
        p, m = search(scored, counts, hit, ids,
                      grid_t=[round(x, 3) for x in [i / 40 for i in range(8, 40)]],
                      grid_min_best=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                      grid_margin=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    if a.save_params:
        Path(a.save_params).write_text(json.dumps(p.as_dict(), indent=2))

    pred = apply(scored, p)
    pred_dict = to_prediction_dict(pred)
    # independent cross-check against the reference implementation
    res = evaluate_predictions(gt_dict, pred_dict, entity_universe=ids)
    s2 = per_source(gt_dict, pred_dict, ids, "S2-")
    s3 = per_source(gt_dict, pred_dict, ids, "S3-")

    print(f"\n{'='*66}\nDECISION PARAMS: {p.as_dict()}\n{'='*66}")
    print(res)
    print(f"\n  fast-path macro F0.5 : {m['macro_f05']:.6f}  "
          f"(reference: {res.macro_f05:.6f}, delta {abs(m['macro_f05']-res.macro_f05):.2e})")
    print(f"\n  S2-only macro F0.5   : {s2.macro_f05:.6f}  "
          f"(P {s2.mean_precision:.4f} / R {s2.mean_recall:.4f})")
    print(f"  S3-only macro F0.5   : {s3.macro_f05:.6f}  "
          f"(P {s3.mean_precision:.4f} / R {s3.mean_recall:.4f})")
    print(f"\n  blocking ceiling     : {a.ceiling:.4f}")
    print(f"  achieved / ceiling   : {res.macro_f05 / a.ceiling:.4f}")
    print(f"  gap to ceiling       : {a.ceiling - res.macro_f05:.4f}")
    print(f"\n  runtime              : {time.time() - t0:.0f}s")

    out = {
        "experiment_id": a.experiment_id, "fold": a.fold,
        "params": p.as_dict(), "macro_f05": res.macro_f05,
        "singleton_acc": res.singleton_accuracy,
        "s2_f05": s2.macro_f05, "s3_f05": s3.macro_f05,
        "pair_precision": res.micro_precision, "pair_recall": res.micro_recall,
        "ceiling": a.ceiling,
    }
    Path(f"experiments/{a.experiment_id}_{a.fold}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote experiments/{a.experiment_id}_{a.fold}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
