"""Retrieval metrics against ground truth.

macro_f05_ceiling is the score a PERFECT ranker would obtain given this
candidate set: per S1, F0.5 with precision 1 and recall captured/true
(singletons score 1.0). It is the quantity that decides whether retrieval or
the downstream model is the binding constraint, and pair-level recall hides it.
"""
from __future__ import annotations


def fbeta(p: float, r: float, beta2: float = 0.25) -> float:
    d = beta2 * p + r
    return 0.0 if d <= 0 else (1 + beta2) * p * r / d


def summarise(retrieved: dict[str, set], truth: dict[str, set], entity_ids: list[str]) -> dict:
    n_cand = sum(len(retrieved.get(e, ())) for e in entity_ids)
    n_true = sum(len(truth.get(e, ())) for e in entity_ids)
    tp = sum(len(retrieved.get(e, set()) & truth.get(e, set())) for e in entity_ids)
    ceil = 0.0
    for e in entity_ids:
        t = truth.get(e, set())
        if not t:
            ceil += 1.0
        else:
            ceil += fbeta(1.0, len(retrieved.get(e, set()) & t) / len(t))
    return {
        "n_entities": len(entity_ids),
        "n_candidates": n_cand,
        "n_true_pairs": n_true,
        "true_retrieved": tp,
        "candidate_recall": tp / n_true if n_true else float("nan"),
        "candidate_precision": tp / n_cand if n_cand else float("nan"),
        "cand_per_entity": n_cand / len(entity_ids),
        "macro_f05_ceiling": ceil / len(entity_ids),
    }
