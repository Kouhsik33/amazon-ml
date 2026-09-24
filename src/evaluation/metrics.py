"""Official competition metric: macro-averaged F_0.5 per Source-1 entity.

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

computed per S1 entity and then averaged over *all* S1 entities in the
evaluation set, singletons included.

Scoring conventions, taken directly from the problem statement:

* true empty, predicted empty  -> 1.0  ("correctly predicting no match is
  worth a full 1.0 on that entity")
* true empty, predicted non-empty -> 0.0
* true non-empty, predicted empty -> 0.0 (recall 0)
* otherwise the F_0.5 formula on the set intersection.

An S1 entity present in the ground truth but absent from the predictions is
scored as an empty prediction -- never skipped -- because the leaderboard
requires a row for every S1 entity and silently dropping hard entities must
not inflate the score.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Mapping, Sequence, Set

BETA = 0.5
_B2 = BETA * BETA  # 0.25


def fbeta(precision: float, recall: float, beta2: float = _B2) -> float:
    denom = beta2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


@dataclass
class EvalResult:
    macro_f05: float
    mean_precision: float
    mean_recall: float
    singleton_accuracy: float
    n_entities: int
    n_singletons: int
    n_predicted_matches: int
    n_true_matches: int
    false_positives: int
    false_negatives: int
    true_positives: int
    micro_precision: float
    micro_recall: float
    micro_f05: float
    macro_f05_singletons: float
    macro_f05_nonsingletons: float
    n_entities_with_prediction: int

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"macro_F0.5          : {self.macro_f05:.6f}\n"
            f"  on singletons     : {self.macro_f05_singletons:.6f} "
            f"({self.n_singletons} entities)\n"
            f"  on non-singletons : {self.macro_f05_nonsingletons:.6f} "
            f"({self.n_entities - self.n_singletons} entities)\n"
            f"mean precision      : {self.mean_precision:.6f}\n"
            f"mean recall         : {self.mean_recall:.6f}\n"
            f"singleton accuracy  : {self.singleton_accuracy:.6f}\n"
            f"entities            : {self.n_entities} "
            f"({self.n_entities_with_prediction} with >=1 prediction)\n"
            f"predicted matches   : {self.n_predicted_matches}\n"
            f"true matches        : {self.n_true_matches}\n"
            f"TP / FP / FN        : {self.true_positives} / "
            f"{self.false_positives} / {self.false_negatives}\n"
            f"micro P / R / F0.5  : {self.micro_precision:.6f} / "
            f"{self.micro_recall:.6f} / {self.micro_f05:.6f}"
        )


def _as_set(value) -> Set[str]:
    """Accept a set, list, or a raw comma-separated string; de-duplicate."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {t for t in (x.strip() for x in value.split(",")) if t}
    return {t for t in (str(x).strip() for x in value) if t}


def evaluate_predictions(
    ground_truth: Mapping[str, Iterable[str]],
    predictions: Mapping[str, Iterable[str]],
    *,
    entity_universe: Sequence[str] | None = None,
) -> EvalResult:
    """Score ``predictions`` against ``ground_truth``.

    Both mappings are ``{source1_entity_id: iterable-of-matched-ids}``; the
    values may also be raw comma-separated strings, which is what the TSVs
    hold. Duplicate IDs inside a list are collapsed, mirroring the scorer
    (the submission validator rejects them outright, so they can never help).

    ``entity_universe`` overrides the set of S1 entities averaged over; by
    default it is the ground truth's key set.
    """
    keys = list(entity_universe) if entity_universe is not None else list(ground_truth.keys())

    f_sum = 0.0
    f_single = 0.0
    f_nonsingle = 0.0
    p_sum = 0.0
    r_sum = 0.0
    n_with_pred = 0
    n_singletons = 0
    singleton_correct = 0
    tp = fp = fn = 0
    n_pred = n_true = 0

    for key in keys:
        truth = _as_set(ground_truth.get(key))
        pred = _as_set(predictions.get(key))

        n_pred += len(pred)
        n_true += len(truth)
        hit = len(truth & pred)
        tp += hit
        fp += len(pred) - hit
        fn += len(truth) - hit

        if not truth:
            n_singletons += 1
            score = 1.0 if not pred else 0.0
            singleton_correct += int(not pred)
            f_single += score
        else:
            if pred:
                precision = hit / len(pred)
                recall = hit / len(truth)
                score = fbeta(precision, recall)
            else:
                score = 0.0
            f_nonsingle += score

        if pred:
            n_with_pred += 1
            p_sum += hit / len(pred)
            r_sum += (hit / len(truth)) if truth else 0.0

        f_sum += score

    n = len(keys)
    n_non = n - n_singletons
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0

    return EvalResult(
        macro_f05=f_sum / n if n else 0.0,
        mean_precision=p_sum / n_with_pred if n_with_pred else 0.0,
        mean_recall=r_sum / n_with_pred if n_with_pred else 0.0,
        singleton_accuracy=singleton_correct / n_singletons if n_singletons else float("nan"),
        n_entities=n,
        n_singletons=n_singletons,
        n_predicted_matches=n_pred,
        n_true_matches=n_true,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f05=fbeta(micro_p, micro_r),
        macro_f05_singletons=f_single / n_singletons if n_singletons else float("nan"),
        macro_f05_nonsingletons=f_nonsingle / n_non if n_non else float("nan"),
        n_entities_with_prediction=n_with_pred,
    )


def load_id_list_tsv(path: str, value_column: int = 1) -> Dict[str, Set[str]]:
    """Read a two-column ``id \\t comma,separated,ids`` TSV into a dict."""
    out: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        fh.readline()  # header
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            key = parts[0].strip()
            raw = parts[value_column] if len(parts) > value_column else ""
            out[key] = _as_set(raw)
    return out
