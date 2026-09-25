# E05 Training Runtime Investigation

**Question:** E05 LightGBM training took 1,581s vs E04's 122s after adding 9 fuzzy
numeric features (62 -> 71). Is the feature set responsible?

**Answer: No.** Under controlled benchmarks the feature set costs **1.35x**
(100.3s -> 135.1s). The worst reproducible configuration reaches **2.0x**
(201.3s). The remaining ~6.5x was **memory-pressure/swap degradation specific to
that run**, not a property of the features. E05 model logic was not modified.

## Method

E05's feature set is a strict superset of E04's (9 added, 0 removed), so both
matrices are built **once** from the same 200k-entity candidate file and the
E04 matrix is recovered by column selection. Identical pairs, identical rows,
identical values -- only the feature columns differ.

Each variant runs in its **own process** so peak RSS is attributable and no
allocator state carries over. LightGBM runs at `verbose=1` so its own
threading/binning decisions are captured.

* `scripts/bench/build_matrix.py` -- builds and caches the shared matrix
* `scripts/bench/bench_train.py` -- benchmarks one variant

Shared inputs (identical to the original E05 run):

| | |
|---|---|
| candidate file | `cand_train_200k_k80.parquet` (200,000 S1 entities) |
| pairs | 2,879,651 (642,216 pos + 1,519,009 hard-neg + 718,426 rand-neg) |
| positive rate | 0.2230 |
| matrix | float32, 0.714 GB (E04) / 0.818 GB (E05) |
| feature generation | 287.2s (one-time, shared by both variants) |

LightGBM parameters (unchanged from E05 -- this is the control):
`objective=binary, num_leaves=127, n_estimators=600, learning_rate=0.06,
min_child_samples=60, subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
reg_lambda=1.0, random_state=7, n_jobs=-1`

Hardware: 12 logical / 6 physical cores, 16 GB RAM. `OMP_NUM_THREADS` unset.

## Results

| run | features | threads | side frames resident | Dataset construct | **train** | score 500k | peak RSS | swap delta | free RAM at start |
|---|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| e04 | 62 | 12 | no | 1.7s | **100.3s** | 4.3s | 2.39 GB | -0.70 GB | 1.26 GB |
| e05 | 71 | 12 | no | 2.1s | **135.1s** | 7.1s | 1.71 GB | -0.02 GB | 0.68 GB |
| e04 + held sides | 62 | 12 | yes | 5.6s | **136.0s** | 7.4s | 4.58 GB | +4.24 GB | 0.02 GB |
| e05 + held sides | 71 | 12 | yes | 6.6s | **201.3s** | 7.8s | 5.81 GB | +5.23 GB | 2.74 GB |
| e05 force_row_wise | 71 | 12 | no | 2.7s | **127.4s** | 7.6s | 1.69 GB | -0.61 GB | 0.90 GB |
| e05 threads=6 | 71 | 6 | no | 3.2s | **125.9s** | 11.8s | 1.75 GB | -2.77 GB | 4.00 GB |
| e05 threads=4 | 71 | 4 | no | 4.8s | **239.2s** | 14.0s | 1.68 GB | -0.30 GB | 0.80 GB |

## Attribution by stage

1. **Feature generation** -- 287.2s, shared by both variants, outside the
   training timer. Not a factor in the 122s vs 1,581s comparison.
2. **NumPy conversion** -- 0.0-3.2s (column selection + contiguity). Negligible.
3. **Dataset construction (bin-finding)** -- 1.7s (E04) vs 2.1s (E05); rises to
   5.6-6.6s only when memory is tight. Under 4% of training time in every run.
4. **Histogram construction** -- Total Bins 6906 (E04) vs 7349 (E05), +6.4%.
   Used features 60 vs 69, +15%. Both changes are small and linear.
5. **Training** -- the 2x2 isolates it cleanly:
   * features alone: 100.3 -> 135.1s = **1.35x**
   * residency alone (E04): 100.3 -> 136.0s = **1.36x**
   * both together: 100.3 -> 201.3s = **2.0x** (roughly multiplicative)
6. **Scoring** -- 4.3s -> 7.1s for 500k rows. Proportional, not pathological.
7. **Memory / swapping** -- the dominant term, and the only one capable of
   producing a 13x figure. See below.
8. **Thread configuration** -- all runs auto-chose **row-wise** multi-threading
   (overhead 0.12-0.18s); forcing it saves only that. 12 logical threads gave
   **no benefit over 6 physical** (135.1s vs 125.9s -- 6 was marginally faster,
   consistent with hyperthread contention on 6 physical cores). Dropping to 4
   threads cost 1.8x. Thread configuration did not cause the regression, though
   `n_jobs=6` is mildly preferable to `-1` on this box.

## Why the features are not responsible

The two mechanisms by which 9 extra features could plausibly cause a
super-linear slowdown were both tested and ruled out:

* **High cardinality forcing expensive bin-finding.** The 9 new features
  contribute 454,134 distinct values combined; the E04 features already
  contained 2,564,223, including `rr_rel` (851,704), `rr_margin` (773,189) and
  `rr_score` (409,010). The highest-cardinality new feature
  (`num_first_rel_diff`, 453,757) is *smaller* than three features E04 already
  had. Dataset construction time confirms this: 1.7s -> 2.1s.
* **NaN/Inf triggering missing-value split handling.** There are **zero** NaN
  and **zero** Inf values across all 71 features and 2,879,651 rows
  (`feature_diagnostics.json`).

## What actually happened

This machine degrades by more than an order of magnitude once it swaps, and
there is direct precedent from the same session **with no feature change at
all**: the original E04 run scored 2,526,907 dev pairs in **10,090s**, while
the identical code on identical data later scored 3,795,752 pairs in **281s**
(13,486 pairs/s) once memory was free -- a **51x** swing driven purely by
memory pressure.

The original E05 training ran in a single long-lived process that held the
normalised S1 frame, the 10.3M-row S2/S3 pool, the sampled-pair frame and the
0.82 GB feature matrix simultaneously (peak RSS 5.81 GB reproduced here) on a
16 GB box that was already several GB into swap after hours of large jobs.
A 13x degradation of the boosting loop under those conditions is well inside
the range this machine demonstrably produces.

**Not proven by artificial reproduction.** The worst state reproduced here was
free RAM 0.02 GB / swap +5.23 GB, which yielded 2.0x. Deliberately ballooning
memory further to force the full 13x was judged too disruptive to the machine
to be worth it, since the E04 scoring incident already establishes the
mechanism on identical code.

## Implications for scaling training data

* Training cost scales roughly linearly here; budget **~135s per 2.9M pairs**
  at 71 features on 12 threads, not 1,581s.
* **The binding constraint is RAM, not features.** Before scaling beyond 200k
  entities, drop `sides` and the candidate/label frames before calling `fit()`
  -- they are dead weight during boosting and cost 1.5x on their own.
* Prefer `n_jobs=6` (physical cores) over `-1`.
* `force_row_wise=True` removes a ~0.2s probe; cosmetic.
* Timings taken while the machine is swapping are not comparable to each other;
  record free RAM and swap alongside any future timing.
