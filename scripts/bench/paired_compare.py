#!/usr/bin/env python3
"""Paired per-entity comparison of two scored candidate sets on the same fold.

Macro F0.5 is a mean over entities, and both models are evaluated on the *same*
entities, so the paired difference has far smaller variance than either mean.
An unpaired SE would be ~0.001 here and would call a real +0.00125 shift
"noise"; the paired test is the correct one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.splits import load_split  # noqa: E402
from decision.decide import DecisionParams, apply  # noqa: E402
from evaluation.metrics import fbeta  # noqa: E402
from evaluation.report import truth_tables  # noqa: E402


def per_entity_f05(scored_path: str, params: DecisionParams,
                   counts: dict, truth_pairs: set, ids: list[str]) -> np.ndarray:
    pred = apply(pl.read_parquet(scored_path), params)
    g = pred.group_by("source1_entity_id").agg(pl.col("cand_id"))
    pm = {r[0]: r[1] for r in g.iter_rows()}
    out = np.zeros(len(ids), dtype=np.float64)
    for i, eid in enumerate(ids):
        n_true = counts.get(eid, 0)
        cands = pm.get(eid, [])
        n_pred = len(cands)
        if n_true == 0:
            out[i] = 1.0 if n_pred == 0 else 0.0
        elif n_pred == 0:
            out[i] = 0.0
        else:
            tp = sum(1 for c in cands if (eid, c) in truth_pairs)
            out[i] = fbeta(tp / n_pred, tp / n_true)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--fold", default="val")
    ap.add_argument("--params", default="configs/decision_e05.json")
    a = ap.parse_args()

    ids = load_split(a.fold)
    counts, hit, _ = truth_tables(ids)
    truth_pairs = set(zip(hit["source1_entity_id"].to_list(), hit["cand_id"].to_list()))
    p = DecisionParams(**json.loads(Path(a.params).read_text()))

    fa = per_entity_f05(a.a, p, counts, truth_pairs, ids)
    fb = per_entity_f05(a.b, p, counts, truth_pairs, ids)
    d = fb - fa
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = d.mean() / se if se > 0 else float("nan")

    print(f"fold={a.fold}  entities={n:,}  params={p.as_dict()}")
    print(f"  {a.label_a:12s} macro F0.5 = {fa.mean():.6f}")
    print(f"  {a.label_b:12s} macro F0.5 = {fb.mean():.6f}")
    print(f"  paired difference    = {d.mean():+.6f}  (SE {se:.6f}, t = {t:+.2f})")
    print(f"  95% CI               = [{d.mean()-1.96*se:+.6f}, {d.mean()+1.96*se:+.6f}]")
    print(f"  entities improved    = {int((d > 0).sum()):,}")
    print(f"  entities worsened    = {int((d < 0).sum()):,}")
    print(f"  entities unchanged   = {int((d == 0).sum()):,}")
    unpaired = np.sqrt(fa.var(ddof=1)/n + fb.var(ddof=1)/n)
    print(f"  (unpaired SE would be {unpaired:.6f} -- {unpaired/se:.1f}x larger)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
