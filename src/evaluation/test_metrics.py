"""Self-contained checks for the F_0.5 implementation. Run: python -m evaluation.test_metrics"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import evaluate_predictions, fbeta  # noqa: E402

FAIL = []


def check(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got:.6f} want {want:.6f}")
    if not ok:
        FAIL.append(name)


# 1. The worked example from the README.
gt = {"S1-00001": ["S2-00047", "S3-00812"]}
pred = {"S1-00001": ["S2-00047", "S3-00193", "S3-00812"]}
r = evaluate_predictions(gt, pred)
check("README example (P=2/3, R=1 -> 0.714)", r.macro_f05, 0.7142857142857143, 1e-6)

# 2. Singleton handled correctly both ways.
check("singleton + empty pred -> 1.0",
      evaluate_predictions({"a": []}, {"a": []}).macro_f05, 1.0)
check("singleton + any pred -> 0.0",
      evaluate_predictions({"a": []}, {"a": ["S2-1"]}).macro_f05, 0.0)

# 3. Non-singleton with empty prediction -> 0.0 (not skipped, not 1.0).
check("true matches + empty pred -> 0.0",
      evaluate_predictions({"a": ["S2-1"]}, {"a": []}).macro_f05, 0.0)

# 4. Entity missing from predictions is scored as empty, never dropped.
r = evaluate_predictions({"a": ["S2-1"], "b": []}, {})
check("missing rows scored as empty (0.0 + 1.0)/2", r.macro_f05, 0.5)
check("  n_entities stays 2", r.n_entities, 2)

# 5. Perfect prediction.
check("perfect -> 1.0",
      evaluate_predictions({"a": ["S2-1", "S3-2"]}, {"a": ["S3-2", "S2-1"]}).macro_f05, 1.0)

# 6. Duplicate predictions collapse (must not inflate or deflate).
check("duplicate ids collapse",
      evaluate_predictions({"a": ["S2-1"]}, {"a": ["S2-1", "S2-1"]}).macro_f05, 1.0)

# 7. One-to-many preserved: 5 true, 5 predicted, all right.
check("one-to-many 5/5",
      evaluate_predictions({"a": [f"S2-{i}" for i in range(5)]},
                           {"a": [f"S2-{i}" for i in range(5)]}).macro_f05, 1.0)

# 8. Precision weighted 2x over recall: check asymmetry explicitly.
#    half precision (P=.5,R=1) must score WORSE than half recall (P=1,R=.5).
low_p = fbeta(0.5, 1.0)
low_r = fbeta(1.0, 0.5)
print(f"      F0.5(P=.5,R=1)={low_p:.6f}  F0.5(P=1,R=.5)={low_r:.6f}")
check("precision-heavy asymmetry (low_r - low_p > 0)", float(low_r > low_p), 1.0)
check("F0.5(P=.5,R=1) == 5/9", low_p, 5 / 9)
check("F0.5(P=1,R=.5) == 5/6", low_r, 5 / 6)

# 9. Macro != micro: one huge entity must not dominate.
gt = {"big": [f"S2-{i}" for i in range(100)], "small": ["S3-1"]}
pred = {"big": [f"S2-{i}" for i in range(100)], "small": ["S3-999"]}
r = evaluate_predictions(gt, pred)
check("macro averages per entity (1.0 + 0.0)/2", r.macro_f05, 0.5)
check("  micro is dominated by 'big'", r.micro_precision, 100 / 101)

# 10. Aggregate counters.
gt = {"a": ["S2-1", "S2-2"], "b": [], "c": ["S3-9"]}
pred = {"a": ["S2-1", "S2-3"], "b": ["S2-7"], "c": []}
r = evaluate_predictions(gt, pred)
check("TP", r.true_positives, 1)
check("FP", r.false_positives, 2)   # S2-3 on a, S2-7 on b
check("FN", r.false_negatives, 2)   # S2-2 on a, S3-9 on c
check("n_true_matches", r.n_true_matches, 3)
check("n_predicted_matches", r.n_predicted_matches, 3)
check("singleton_accuracy (b was wrong)", r.singleton_accuracy, 0.0)
# a: P=.5 R=.5 -> F=.5 ; b: 0 ; c: 0
check("macro over the three", r.macro_f05, 0.5 / 3)

# 11. Comma-string values (the raw TSV form) behave identically to lists.
check("string values parse",
      evaluate_predictions({"a": "S2-1,S2-2"}, {"a": "S2-1,S2-2"}).macro_f05, 1.0)
check("empty string == singleton",
      evaluate_predictions({"a": ""}, {"a": ""}).macro_f05, 1.0)

print()
if FAIL:
    print(f"{len(FAIL)} FAILURES: {FAIL}")
    sys.exit(1)
print("all metric tests passed")
