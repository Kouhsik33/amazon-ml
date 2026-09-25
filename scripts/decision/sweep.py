#!/usr/bin/env python3
"""Decision-layer sweeps. Tunes on dev only; val is never consulted here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/decision")
from engine import Fold, Rule  # noqa: E402

BASE = Rule(**json.load(open("configs/decision_e05.json")))
f = Fold("dev")
b = f.score(BASE)
B = b["macro_f05"]
rows = []


def ev(name, r):
    m = f.score(r)
    m["name"] = name
    m["delta"] = m["macro_f05"] - B
    m["rule"] = r.as_dict()
    rows.append(m)
    return m


print(f"DEV baseline macro F0.5 = {B:.6f}  (FP {b['FP']:,}  FN {b['FN']:,})\n")

# ---- STEP 5: global threshold, then separate S2/S3 ----
print("STEP 5a  global threshold")
print(f"  {'t':>6s} {'macroF0.5':>11s} {'delta':>9s} {'meanP':>7s} {'meanR':>7s} {'FP':>7s} {'FN':>7s}")
for t in [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95]:
    m = ev(f"global_t={t}", Rule(t_s2=t, t_s3=t))
    print(f"  {t:>6.2f} {m['macro_f05']:>11.6f} {m['delta']:>+9.6f} "
          f"{m['mean_precision']:>7.4f} {m['mean_recall']:>7.4f} {m['FP']:>7,} {m['FN']:>7,}")

print("\nSTEP 5b  separate S2/S3 (coarse)")
best_sep, best_sep_m = None, {"macro_f05": -1}
for t2 in [0.82, 0.84, 0.86, 0.88, 0.90, 0.92]:
    line = []
    for t3 in [0.82, 0.84, 0.86, 0.88, 0.90, 0.92]:
        m = ev(f"sep_s2={t2}_s3={t3}", Rule(t_s2=t2, t_s3=t3))
        line.append(f"{m['macro_f05']:.5f}")
        if m["macro_f05"] > best_sep_m["macro_f05"]:
            best_sep, best_sep_m = (t2, t3), m
    print(f"  s2={t2:.2f} | " + "  ".join(line))
print(f"  best separate: s2={best_sep[0]} s3={best_sep[1]} -> {best_sep_m['macro_f05']:.6f} "
      f"({best_sep_m['delta']:+.6f})")

T2, T3 = best_sep

# ---- rel_margin (already in the baseline rule at 0.9) ----
print("\nSTEP 5c  rel_margin (keep p >= best*rel)")
for rm in [0.0, 0.70, 0.80, 0.85, 0.90, 0.95]:
    m = ev(f"relmargin={rm}", Rule(t_s2=T2, t_s3=T3, rel_margin=rm))
    print(f"  rel={rm:.2f}  {m['macro_f05']:.6f}  {m['delta']:+.6f}  FP {m['FP']:,}  FN {m['FN']:,}")

# ---- STEP 6: best-minus-second gap ----
print("\nSTEP 6  entity gap rule: require (best - second) >= gap")
for g in [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]:
    m = ev(f"gap={g}", Rule(t_s2=T2, t_s3=T3, gap_min=g))
    print(f"  gap={g:.2f}  {m['macro_f05']:.6f}  {m['delta']:+.6f}  FP {m['FP']:,}  FN {m['FN']:,}")

# ---- STEP 7: candidate-count aware thresholds ----
print("\nSTEP 7  stricter threshold for high-candidate-count entities")
for cut in [10, 20, 50]:
    for d in [0.01, 0.02, 0.03, 0.05]:
        m = ev(f"bump_cnt>{cut}_+{d}", Rule(t_s2=T2, t_s3=T3, count_bumps=((cut, d),)))
        print(f"  cnt>{cut:>2d} +{d:.2f}  {m['macro_f05']:.6f}  {m['delta']:+.6f}  "
              f"FP {m['FP']:,}  FN {m['FN']:,}")

# ---- STEP 10: singleton gate ----
print("\nSTEP 10  singleton gate: drop entity entirely if best < min_best")
for mb in [0.0, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96]:
    m = ev(f"min_best={mb}", Rule(t_s2=T2, t_s3=T3, min_best=mb))
    print(f"  min_best={mb:.2f}  {m['macro_f05']:.6f}  {m['delta']:+.6f}  "
          f"singleton_acc {m['singleton_acc']:.4f}  FP {m['FP']:,}  FN {m['FN']:,}")

Path("experiments/decision").mkdir(parents=True, exist_ok=True)
json.dump(rows, open("experiments/decision/sweep_dev.json", "w"), indent=1)
top = sorted(rows, key=lambda r: -r["macro_f05"])[:8]
print(f"\n=== TOP 8 ON DEV (baseline {B:.6f}) ===")
for r in top:
    print(f"  {r['name']:28s} {r['macro_f05']:.6f}  {r['delta']:+.6f}  "
          f"FP {r['FP']:>6,}  FN {r['FN']:>6,}  sing {r['singleton_acc']:.4f}")
