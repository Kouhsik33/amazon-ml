"""Precision-first decision layer, tuned directly on macro F_0.5.

Why not ``p > 0.5``: F_0.5 weights precision twice as heavily as recall, and a
false positive on a singleton costs the entity's whole 1.0. The optimum sits
well above 0.5 and must be found by searching the metric itself.

Four knobs, each answering a failure mode the data actually shows:

* ``t_s2`` / ``t_s3`` -- per-source thresholds. S2 and S3 have different match
  count distributions (mean 1.674 vs 1.787, max 5 vs 6) and different noise, so
  a shared threshold is an assumption, not a default.
* ``min_best`` -- **singleton gate**. If the best candidate for an entity does
  not clear this, the entity is declared a singleton and predicted empty.
  Correct empties are worth a full 1.0 each and 5.6% of entities are singletons.
* ``rel_margin`` -- keep a candidate only if it scores within this fraction of
  the entity's best candidate. One-to-many is real (mean 3.46 matches), so a
  fixed top-k is wrong; a relative margin admits however many candidates are
  *comparably* good and cuts the long tail of stragglers.
* ``max_k`` -- a safety cap. Ground truth never exceeds 11 total matches.

No one-to-one assignment is imposed anywhere: an entity may emit 0, 1 or many
matches, and the same candidate may in principle be claimed by more than one
entity (the many-to-one property is exploited only as a soft signal).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Set

import numpy as np
import polars as pl

from evaluation.metrics import fbeta


@dataclass
class DecisionParams:
    t_s2: float = 0.5
    t_s3: float = 0.5
    min_best: float = 0.0
    rel_margin: float = 0.0     # 0 disables
    max_k: int = 11

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def apply(scored: pl.DataFrame, p: DecisionParams) -> pl.DataFrame:
    """Return the surviving (source1_entity_id, cand_id) predictions."""
    df = scored.with_columns(
        pl.when(pl.col("cand_id").str.starts_with("S2-"))
        .then(pl.lit(p.t_s2)).otherwise(pl.lit(p.t_s3)).alias("_t"),
        pl.col("p").max().over("source1_entity_id").alias("_best"),
    )
    df = df.filter(pl.col("p") >= pl.col("_t"))
    if p.min_best > 0:
        df = df.filter(pl.col("_best") >= p.min_best)
    if p.rel_margin > 0:
        df = df.filter(pl.col("p") >= pl.col("_best") * p.rel_margin)
    if p.max_k:
        df = (df.sort("p", descending=True)
                .group_by("source1_entity_id", maintain_order=True).head(p.max_k))
    return df.select("source1_entity_id", "cand_id")


def _score_fast(pred: pl.DataFrame, truth_counts: Dict[str, int],
                hit_lookup: pl.DataFrame, entity_ids: List[str]) -> Dict[str, float]:
    """Macro F0.5 without materialising per-entity Python sets."""
    j = pred.join(hit_lookup, on=["source1_entity_id", "cand_id"], how="left") \
            .with_columns(pl.col("y").fill_null(0))
    agg = j.group_by("source1_entity_id").agg(
        pl.col("y").sum().alias("tp"), pl.len().alias("n_pred"))
    got = dict(zip(agg["source1_entity_id"].to_list(),
                   zip(agg["tp"].to_list(), agg["n_pred"].to_list())))
    f_sum = 0.0
    n_single = single_ok = 0
    tp_t = fp_t = fn_t = 0
    for eid in entity_ids:
        n_true = truth_counts.get(eid, 0)
        tp, n_pred = got.get(eid, (0, 0))
        tp_t += tp
        fp_t += n_pred - tp
        fn_t += n_true - tp
        if n_true == 0:
            n_single += 1
            ok = n_pred == 0
            single_ok += ok
            f_sum += 1.0 if ok else 0.0
        elif n_pred == 0:
            pass
        else:
            f_sum += fbeta(tp / n_pred, tp / n_true)
    n = len(entity_ids)
    return {
        "macro_f05": f_sum / n,
        "singleton_acc": single_ok / n_single if n_single else float("nan"),
        "pair_precision": tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0.0,
        "pair_recall": tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0.0,
        "n_pred": tp_t + fp_t,
    }


def evaluate(scored: pl.DataFrame, truth_counts: Dict[str, int],
             hit_lookup: pl.DataFrame, entity_ids: List[str],
             p: DecisionParams) -> Dict[str, float]:
    return _score_fast(apply(scored, p), truth_counts, hit_lookup, entity_ids)


def search(scored: pl.DataFrame, truth_counts: Dict[str, int],
           hit_lookup: pl.DataFrame, entity_ids: List[str],
           grid_t: Iterable[float], grid_min_best: Iterable[float],
           grid_margin: Iterable[float], per_source: bool = True,
           verbose: bool = True) -> tuple[DecisionParams, Dict[str, float]]:
    """Coordinate search over the decision knobs, maximising macro F0.5.

    Run on ``dev`` only. ``val`` is scored once at the chosen setting so the
    reported number is not the maximum of a search.
    """
    best_p, best_m = DecisionParams(), {"macro_f05": -1.0}

    for t in grid_t:
        m = evaluate(scored, truth_counts, hit_lookup, entity_ids,
                     DecisionParams(t_s2=t, t_s3=t))
        if m["macro_f05"] > best_m["macro_f05"]:
            best_p, best_m = DecisionParams(t_s2=t, t_s3=t), m
    if verbose:
        print(f"  global threshold -> t={best_p.t_s2:.3f} F0.5={best_m['macro_f05']:.4f}")

    if per_source:
        for t2 in grid_t:
            for t3 in grid_t:
                cand = DecisionParams(t_s2=t2, t_s3=t3)
                m = evaluate(scored, truth_counts, hit_lookup, entity_ids, cand)
                if m["macro_f05"] > best_m["macro_f05"]:
                    best_p, best_m = cand, m
        if verbose:
            print(f"  per-source       -> s2={best_p.t_s2:.3f} s3={best_p.t_s3:.3f} "
                  f"F0.5={best_m['macro_f05']:.4f}")

    for mb in grid_min_best:
        cand = DecisionParams(**{**best_p.as_dict(), "min_best": mb})
        m = evaluate(scored, truth_counts, hit_lookup, entity_ids, cand)
        if m["macro_f05"] > best_m["macro_f05"]:
            best_p, best_m = cand, m
    if verbose:
        print(f"  + singleton gate -> min_best={best_p.min_best:.3f} "
              f"F0.5={best_m['macro_f05']:.4f}")

    for rm in grid_margin:
        cand = DecisionParams(**{**best_p.as_dict(), "rel_margin": rm})
        m = evaluate(scored, truth_counts, hit_lookup, entity_ids, cand)
        if m["macro_f05"] > best_m["macro_f05"]:
            best_p, best_m = cand, m
    if verbose:
        print(f"  + rel margin     -> rel_margin={best_p.rel_margin:.3f} "
              f"F0.5={best_m['macro_f05']:.4f}")

    return best_p, best_m


def to_prediction_dict(pred: pl.DataFrame) -> Dict[str, Set[str]]:
    g = pred.group_by("source1_entity_id").agg(pl.col("cand_id"))
    return {r[0]: set(r[1]) for r in g.iter_rows()}
