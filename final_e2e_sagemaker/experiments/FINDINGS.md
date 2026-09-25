# Retrieval experiments — measured results

Scope for every number below: **US / S2 only / 1,000-S1 deterministic sample**
(seed 20260926, first 3 IDs `S1-452519725`, `S1-794693624`, `S1-484409229`),
drawn from the `val` split. Normalisation is this repo's cached
`nm_full` / `ad_full`, held identical across all configurations (see Caveats).

`ceiling` = macro-F0.5 a **perfect** ranker would score given the candidate set:
per S1, F0.5 at precision 1.0 and recall `captured/true`.

---

## EXP-01 / EXP-02 — TOP_K sweep at two address DF gates

One variable per column pair. All other parameters frozen at Approach2 values
(`name_share 0.05`, `pool 600`, `cap_frac 0.0002`, `min_cap 20`, `rerank cos`).

| top_k | recall @225,315 | ceiling @225,315 | recall @400,000 | ceiling @400,000 | ceiling Δ |
|---:|---:|---:|---:|---:|---:|
| 5   | 93.0699% | 0.967270 | 94.1641% | 0.971401 | +0.004131 |
| 10  | 96.3526% | 0.980121 | 96.7173% | 0.981844 | +0.001723 |
| **15 (baseline)** | **97.0213%** | **0.984864** | **97.3860%** | **0.986454** | +0.001590 |
| 30  | 97.7508% | 0.988933 | 98.2979% | 0.991587 | +0.002654 |
| 50  | 98.2371% | 0.992420 | 98.6626% | 0.993254 | +0.000834 |
| 100 | 99.0274% | 0.994797 | 99.2097% | 0.995102 | +0.000305 |
| 600 (pool) | 99.3313% | 0.996150 | 99.4529% | 0.997241 | +0.001091 |

Candidate precision falls as expected: 10.70% @ k=15 → 5.44% @ k=30 →
3.32% @ k=50. Filtering that is the ranker's job.

Artifacts: `exp01_topk_sweep.json`, `exp01_topk_sweep_400k.json`.
EXP-02 was run twice (a variable-shadowing bug broke only the JSON write on the
first run); both runs produced identical metrics.

---

## MEASURED RESULT — the binding constraint is TOP_K, not the DF gate

**At the frozen baseline (`top_k=15`, `addr_max_df=225,315`) the ceiling is
0.984864, which is below the 0.987 validation gate.** On this population a
*perfect* ranker cannot reach the target. The constraint is the candidate set,
not the model.

Effect sizes at `top_k=15`, measured on the same sample:

| change | recall Δ | ceiling |
|---|---:|---:|
| DF gate 225,315 → 400,000 | +0.365 pp | 0.986454 (still < 0.987) |
| TOP_K 15 → 30 | +0.730 pp | 0.988933 (clears gate) |
| both | +1.277 pp | 0.991587 |

TOP_K moves recall **~2x** as much as the DF gate, and the two are
**complementary, not redundant** — the DF gate still adds ceiling at every K.

## HYPOTHESIS (not yet tested end-to-end)

`top_k=30` + `addr_max_df=400,000` raises the ceiling to 0.991587 and is the
cheapest configuration that clears 0.987 with headroom. Whether final F0.5
improves is **unproven**: doubling candidates halves candidate precision, and
under F0.5 the ranker must absorb ~2x more false candidates per entity. This
requires an end-to-end run to settle.

## Caveats

- **US / S2 only.** S3, India and France are unmeasured; the full-task ceiling
  will differ.
- **Normalisation is this repo's, not Approach2's `normalize.py`.** Approach2's
  needs `lexicon.json` from `s01_lexicon.py`, which has not been run. Because
  normalisation is held constant across every configuration, the *comparisons*
  are valid; the *absolute* values are not byte-comparable to an Approach2 run.
- Ceiling is an upper bound only. No ranker, no decision layer, no F0.5 here.
- 1,000 S1 / ~1,645 true S2 pairs — small-integer counts carry real uncertainty.

---

## EXP-03 — end-to-end macro F0.5 (retrieval → features → LightGBM → threshold)

Answers what EXP-01/02 could not: the ceiling rose, but does **actual** F0.5?

Splits (entity-level, disjoint, leak-asserted at runtime):
`FIT` 4,823 train-split US S1 → model only · `TUNE` 1,177 train-split US S1 →
threshold only · `EVAL` 1,000 val-split US S1 (the EXP-01 sample) → scored once.
Features: 62 lexical/context + retrieval score + retrieval rank. Negatives are
retrieved false candidates, i.e. hard by construction.

| config | cand recall | cand prec | **macro F0.5** | pair P | pair R | predicted | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A `k=15, df=225,315` (baseline) | 97.0213% | 10.7337% | 0.921903 | 0.9489 | 0.9137 | 1,584 | 81 | 142 |
| B `k=30, df=225,315` | 97.7508% | 5.4544% | **0.930073** | 0.9697 | 0.9143 | 1,551 | 47 | 141 |
| C `k=30, df=400,000` | 98.2979% | 5.4439% | 0.928657 | 0.9682 | 0.9082 | 1,543 | 49 | 151 |

### MEASURED RESULT

1. **TOP_K 15 → 30 improves end-to-end F0.5: +0.008170** (0.921903 → 0.930073).
   Driven by precision (0.9489 → 0.9697): FP fall 81 → 47 while recall holds.
   Doubling candidates gave the ranker better contrast, not more noise.

2. **The DF relaxation does NOT survive ranking: −0.001416** (B → C).
   Candidate recall rises (+0.55 pp) yet F0.5 *falls*: +1 FP and +10 FN.
   The extra candidates it admits are not ones the ranker can exploit.

**The two retrieval changes behave oppositely end-to-end.** Candidate recall
alone was a misleading proxy — C has the best recall of the three and the
second-worst F0.5.

### RECOMMENDATION

Adopt `top_k=30`. Do **not** adopt `addr_max_df=400,000`; keep the frozen
0.03 share.

### Caveats

- US / S2 only. **0.92–0.93 here is not comparable to the 0.987 full-task
  gate** — the restricted task has a different population and denominator.
- Ranker trained on 4,823 entities only; it is deliberately small for a fast
  controlled comparison. The gap to the k=30 ceiling (0.988933 vs 0.930073)
  is now **model-limited, not retrieval-limited**.
- Single seed, 1,000 eval entities; FP/FN differences of ~2 are within noise.
  The +0.0082 for k=30 is well clear of that; the −0.0014 for 400K is not.

### Correctness fix affecting EXP-01/02

`uniq.searchsorted(h)` was used without an exact-match test, so a query key
absent from the B vocabulary aliased to a neighbouring key and contributed its
postings. Fixed in `engine.lookup()` (index and query sides). Re-measurement
shows candidate **recall is unchanged** (97.0213% / 97.7508% / 98.2979% all
reproduce), so the EXP-01/02 recall and ceiling figures stand; candidate
*precision* shifts marginally (10.7035% → 10.7337% at k=15).
