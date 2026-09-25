# Final End-to-End Pipeline — SageMaker

Documentation only. **No pipeline code exists in this folder yet.** This README
records the intended execution flow, the frozen baseline, and the safety rules
that govern how the pipeline may be built and run.

Branch: `final-e2e-sagemaker`
Config: [`config/baseline.yaml`](config/baseline.yaml)

---

## 1. Intended end-to-end flow

```
S3
  └─> data validation
        └─> approach2 normalization / blocking / retrieval
              └─> candidate reranking
                    └─> Stage-1 pair features
                          └─> LightGBM Stage-1
                                └─> Stage-2 cache preparation
                                      └─> LightGBM Stage-2
                                            └─> final decision layer
                                                  └─> VALIDATION  ← gate
                                                        └─> full test inference
                                                              └─> matching_results.tsv
                                                              └─> candidate_pairs.tsv
                                                                    └─> official validator
```

Four rules bind this flow:

1. **Validation must reach the agreed gate before any full test inference.** See §7.
2. **~0.99 is a target, not a guarantee.** No score is promised by this design.
3. **The verified approach2 baseline must remain untouched.** It is reference
   material, not a working copy.
4. **Full test inference must be batched and resumable.** ~1.73M test S1
   entities; see §6.

**AWS credits are limited (~$110).** Every stage that costs money must be
justified before it runs, and compute must be stopped afterwards.

---

## 2. Data layout

### Local (actual, as observed in this repository)

```
student_resource/dataset/
    train/          train_source{1,2,3}.tsv, train_ground_truth.tsv
    test/           test_source{1,2,3}.tsv

data_cache/
    norm_train_source1.parquet
    norm_train_source2.parquet
    norm_train_source3.parquet
    norm_test_source1.parquet
    norm_test_source2.parquet
    norm_test_source3.parquet
```

> **The local repository currently has no `dataset/` directory. The S3 layout is
> separate and must be verified before execution.**

Do not create `dataset/normalized/` locally. Do not move or rename data.

Note: approach2 reads **TSV only** — it has no dependency on the `data_cache/`
Parquet files. Those are recorded for the wider project.

### S3 (verified root only)

| | |
|---|---|
| Region | `ap-south-1` |
| Bucket | `sagemaker-ap-south-1-257949589156` |
| Prefix | `amazon-ml-challenge/` |

**The exact dataset and artifact sub-prefixes beneath this root have not been
verified and are deliberately not documented here.** They must be confirmed by a
read-only inspection task before any pipeline stage depends on them.

---

## 3. Frozen baseline

The verified approach2 configuration. Values recorded in
[`config/baseline.yaml`](config/baseline.yaml).

| Parameter | Value |
|---|---|
| `pool` | 600 |
| `top_k` | 15 |
| `cap_frac` | 0.0002 |
| `min_cap` | 20 |
| `rerank` | `cos` |
| `HFRAC` | 0.15 |
| Stage 2 | enabled |

### The address DF threshold is a share, not a constant

**`ADDR_GENERIC_SHARE = 0.03`** is the actual frozen parameter
(`blocking.py:17`). The threshold applied at runtime is computed per country
(`blocking.py:72`):

```
max_df = ADDR_GENERIC_SHARE × n_all     where n_all = |S1| + |S2| + |S3| for that country
```

`225,315` is **not** a universal constant. It is the value this share resolved to
for the **US** portion of the current validation corpus:

```
0.03 × 7,510,506 = 225,315
```

The same share resolves differently elsewhere — for India,
`0.03 × 5,016,534 = 150,496`. **The threshold is country- and
corpus-dependent.** Any config that encodes a single absolute `max_df` cannot
faithfully drive approach2 across countries; it must encode the share.

### The 400,000 experiment is not the final configuration

A retrieval-only comparison at `max_df = 400,000` has been measured on a
1,000-US-S1 sample. **It has not been evaluated through the LightGBM / F0.5
pipeline, and it is not adopted.** It must remain a separate, explicitly named
experiment configuration and must never silently replace the baseline.

### `HFRAC` coupling (do not change independently)

`HFRAC` must remain `0.15`. `s03_train.py` and `x01_learning_curve.py` each
derive the Stage-2 holdout independently, and `s05_stage2_train.py:53` asserts
the two selections are row-for-row identical. `s03` takes `HFRAC` from argv
(default 0.15) while `x01:18` hard-codes 0.15 — running `s03` with any other
value breaks that assert.

### approach2 stage ordering

```
s01_lexicon  →  s02_block train  →  s03_train (HFRAC=0.15)
             →  x01 build_cache  →  s05_stage2_train  →  s06_predict_stage2
```

`x01_learning_curve.py` is the **sole producer** of `WORK_DIR/cache/meta.parquet`
and `cache/X_{country}.npy`, which `s05` reads unconditionally. Only its
`build_cache()` function is required; the rest of that file is a learning-curve
experiment.

---

## 4. Git safety

- Final branch: **`final-e2e-sagemaker`**
- **Only `final_e2e_sagemaker/` may be committed on this branch.**
- **Never** use `git add -A`. **Never** use `git commit -a`. Stage explicit paths only.
- Do not commit: dataset files, generated candidate files, large indexes or
  models, embeddings, FAISS indexes, credentials.

### Pre-existing dirty state

This branch inherits uncommitted state that **predates** the final pipeline work:

- 19 tracked deletions under `work/*.py` (that directory no longer exists on disk;
  its contents now live under `approach2/work/`)
- untracked `approach2/`

**This state must remain untouched unless explicitly handled in a separate
task.** It is the reason blanket staging is forbidden — `git add -A` on this
branch would sweep all of it into a commit.

---

## 5. SageMaker execution philosophy

1. Clone the GitHub repository in SageMaker Studio / Code Editor.
2. Configure S3 paths (after the sub-prefixes are verified — §2).
3. Verify IAM / S3 access via the **SageMaker execution role**.
   No access keys, secrets, or session tokens in code, config, or Git.
4. Install pinned dependencies.
5. Run a small environment + data smoke test.
6. Run validation **before** any full test inference.
7. Persist expensive intermediate artifacts and checkpoints to S3.
8. Run full test inference **only after** validation passes the gate (§7).
9. Generate submission files.
10. Run the official validator.
11. **Stop / terminate compute when finished.** An open Studio session is not a
    reason for a computation to keep running.

### Commands

No commands are given here. The pipeline does not exist yet, and unverified
commands are worse than none. Each will be filled in only after it has been run
successfully end to end:

| Step | Command |
|---|---|
| Environment check | `<VALIDATED_COMMAND>` |
| S3 access / data check | `<VALIDATED_COMMAND>` |
| Validation run | `<VALIDATED_COMMAND>` |
| Stage-1 training | `<VALIDATED_COMMAND>` |
| Stage-2 training | `<VALIDATED_COMMAND>` |
| Validation scoring | `<VALIDATED_COMMAND>` |
| Checkpoint inspection | `<VALIDATED_COMMAND>` |
| Test inference | `<VALIDATED_COMMAND>` |
| Submission generation | `<VALIDATED_COMMAND>` |
| Stop compute | `<VALIDATED_COMMAND>` |

---

## 6. Resumability requirements

Full test inference (~1.73M S1 entities) must:

- process S1 in **batches**
- write a **completed-batch checkpoint** after each batch
- persist outputs and checkpoints to **S3**
- **resume from the last completed batch** on restart
- **never recompute** already-completed batches
- **never load the entire candidate matrix into RAM**

Batch size must be chosen by **profiling**, not hard-coded blindly.

Indexes are built **once** and reused across batches. Rebuilding the S2/S3 index
per batch is a correctness-preserving but economically fatal mistake.

---

## 7. Validation gate

> **No full test inference until validation is satisfactory.**

Current target: **≈ ≥ 0.987 macro F0.5**

> **This is a target/gate, not a guaranteed achievable score.**

Rules:

- Baseline and experimental configurations must be evaluated using the **same
  validation methodology** — same split, same scorer, same decision procedure.
- A configuration change is not an improvement until measured under that
  methodology.
- Passing the gate freezes the code commit, model, feature schema, thresholds,
  config, and experiment ID before inference begins.

---

## 8. Official submission

Required outputs:

```
matching_results.tsv
candidate_pairs.tsv
```

Official validator:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

> ⚠️ **Paths in the command above require verification against the final
> SageMaker layout before execution.** They reflect the challenge's original
> working directory, not this repository's current structure — locally the
> validator lives at `student_resource/utils/validate_submission.py` and the test
> data at `student_resource/dataset/test`. There is no local `dataset/` directory.

Note: this validator checks **format only**. Its own documentation states it
never computes a score. Macro F0.5 must be measured separately.

---

## 9. Safety rules

Do not silently change any of: candidate generation, normalization, blocking
keys, top-K, LightGBM features, thresholds, decision logic, validation split, or
ground-truth interpretation.

**Every methodology change must be explicit and measurable.**
