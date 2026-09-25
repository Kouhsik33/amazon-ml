# Normalized Dataset — Context & Usage Guide

**Audience:** teammates building independent entity-resolution / matching approaches on top of a shared, pre-normalized version of the challenge data.

**Status:** the files described here were inspected directly (schemas, row counts, file sizes, and the normalization code) as of commit `b94a68435e079267f84f7c16708c456a3f25c422` on `2026-09-25`. Every number in this document comes from reading the actual files and code, not from memory or estimation.

---

## 1. Overview

This document describes a **model-agnostic, pre-normalized version** of the "ML Challenge 2026: Business Entity Resolution" dataset. It exists so that every teammate starts feature engineering, blocking, and modeling from the *same* cleaned text — instead of everyone independently re-implementing (and subtly disagreeing on) Unicode transliteration, accent folding, abbreviation handling, and address tokenization.

**What it is:** for every record in every source file (train and test, S1/S2/S3), the raw `business_name`, `business_address`, and `country` fields have been passed through a shared normalization pipeline that produces several comparison-friendly text views per record (see §4–§5). The **original raw fields are not discarded** — `entity_id` and normalized-but-lossless full-text views are always present, so nothing is opaque or unrecoverable.

**Why it's being shared:** normalization (Unicode romanization of Indic scripts, accent stripping, abbreviation/alias resolution, legal-suffix handling, numeric-token extraction) is genuinely fiddly and easy to get subtly wrong or inconsistent across teammates. Sharing one normalization layer means everyone's blocking/matching results are comparable and nobody wastes time re-deriving the same text-cleaning logic.

**How to use it:** treat the normalized Parquet files as your starting point instead of the raw challenge TSVs. Load them with pandas or polars (§14), build your own candidate generation / feature engineering / matching approach on top, and keep your own work in a separate experiment directory (§12) so the shared files stay immutable and reusable by others.

**This is *not* a finished matching pipeline.** It is a shared preprocessing layer only. Blocking, candidate generation, pairwise features, modeling, thresholding, and final match decisions are all still open and are exactly what teammates are expected to build independently (§7).

---

## 2. Challenge Context

The challenge is a three-source **business entity resolution** problem:

- **Source 1 (S1)** — the *deduplicated reference source*. Every S1 record is a distinct real-world business. IDs are prefixed `S1-`.
- **Source 2 (S2)** and **Source 3 (S3)** — two independent, noisier data feeds about (possibly overlapping) real-world businesses. IDs are prefixed `S2-` and `S3-` respectively. Records in S2/S3 may contain typos, abbreviations, transliterations, reordered tokens, partial addresses, or missing fields relative to how the same business appears in S1.

**The task:** for every S1 entity, find all matching records in S2 and S3 — i.e., all S2/S3 records that refer to the *same real-world business* as that S1 entity.

Critically:

- **A S1 entity can have zero, one, or many matching S2/S3 records.** There is no assumption that every business appears exactly once in S2 or S3, or that it appears at all.
- **The final task is explicitly not one-to-one matching.** A single S1 entity may legitimately match several S2/S3 records (the business appears multiple times, under different noise/corruption patterns, across S2 and S3). Nothing in the challenge specification restricts an S1 entity to at most one match.
- A record's source is identified **only** by its `entity_id` prefix (`S1-`/`S2-`/`S3-`) and by which file it came from — there is no separate `source` column in the raw data.

The ground truth (train only) records, for each S1 entity, the comma-separated list of matching S2/S3 `entity_id`s (empty when there are no matches — i.e. the S1 entity is a "singleton" with respect to S2/S3).

---

## 3. Files Included

All files below live under `data_cache/` in the repository. **`data_cache/` is a derived-data cache — none of it is part of the original challenge download.** The original challenge files remain untouched under `student_resource/dataset/`.

### 3.1 Normalized dataset (the files this document is about)

| File | Source dataset | Split | Rows | Columns | Size on disk |
|---|---|---|---:|---:|---:|
| `norm_train_source1.parquet` | Source 1 | train | 2,206,821 | 10 | 207 MB |
| `norm_train_source2.parquet` | Source 2 | train | 5,034,616 | 10 | 475 MB |
| `norm_train_source3.parquet` | Source 3 | train | 5,285,603 | 10 | 492 MB |
| `norm_test_source1.parquet`  | Source 1 | test  | 1,732,544 | 10 | 166 MB |
| `norm_test_source2.parquet`  | Source 2 | test  | 4,887,273 | 10 | 477 MB |
| `norm_test_source3.parquet`  | Source 3 | test  | 5,082,316 | 10 | 484 MB |

All six files share the same 10-column schema (§4.1). Row counts match the raw challenge files exactly — normalization does not drop, merge, or deduplicate any records.

### 3.2 Ground truth (train only — not normalized, format-converted only)

| File | Rows | Columns | Size on disk |
|---|---:|---:|---:|
| `train_ground_truth.parquet` | 2,206,821 | 2 | 55 MB |

This is a **straight TSV→Parquet conversion** of the original `train_ground_truth.tsv` — the two columns (`source1_entity_id`, `matched_entity_ids`) are untouched raw strings, not normalized in any way. There is **no test ground truth** (the test set is unlabeled, as required by the challenge).

### 3.3 Other files present in `data_cache/` — explicitly *not* part of this package

For completeness, and to avoid any confusion about what is and isn't "the normalized dataset":

- `train_source{1,2,3}.parquet`, `test_source{1,2,3}.parquet` — a raw-format Parquet cache of the original challenge TSVs (columns: `entity_id`, `business_name`, `business_address`, `country`). **Not normalized** — just a faster-to-load copy of the original data. Use these only if you specifically want the untouched raw text.
- `candidates/`, `models/`, `splits/` subdirectories — internal experiment artifacts (candidate pair sets, a trained model, an internal train/dev/val split of S1 IDs) from one team member's own modeling work. These are **not** part of the shared normalized-dataset package and are not described further here — do not treat them as required inputs or as "the answer."

---

## 4. Schema

### 4.1 `norm_{train,test}_source{1,2,3}.parquet` — 10 columns, all UTF-8 strings

| Column | Original / Derived | Description |
|---|---|---|
| `entity_id` | **Original** | Unchanged from the raw challenge file. Prefix (`S1-`/`S2-`/`S3-`) identifies the source. |
| `nm_full` | Derived | Fully normalized business name — Indic script romanized, accents stripped, lowercased, punctuation folded to spaces, legal-suffix spelling variants aliased (§5.1), **no tokens removed**. |
| `nm_core` | Derived | `nm_full` with mined "noise" tokens (legal suffixes / filler words such as `private`, `llc`, `center`) removed. ⚠️ **See §5.4 — stale on disk for most rows in the current export.** |
| `nm_compact` | Derived | `nm_core` with all spaces removed (e.g. `"carneybryant.com"` → `carneybryant`). Matches domain-style / run-together name corruptions. ⚠️ Stale on disk (same cause as `nm_core`). |
| `nm_skel` | Derived | `nm_compact` with vowels (`a,e,i,o,u`) removed — a consonant skeleton, tolerant to vowel-level typos and residual transliteration drift. ⚠️ Stale on disk. |
| `ad_full` | Derived | Fully normalized address — same base text pipeline as `nm_full`, plus null-placeholder tokens removed, digit/letter splitting, ordinal canonicalization, and address-specific alias substitution (§5.2). No tokens removed. |
| `ad_core` | Derived | `ad_full` with mined address "noise" tokens removed (residual state names/abbreviations, unit/box designators, etc.). ⚠️ Stale on disk. |
| `ad_num` | Derived | Space-joined **numeric** tokens extracted from the address (street numbers, PIN/postal codes, unit numbers). ⚠️ Stale on disk (see §5.4 — the live version also strips leading zeros and de-duplicates; the cached column does not). |
| `ad_alpha` | Derived | Space-joined **non-numeric** tokens from `ad_core` — the alphabetic part of the address (street/area names) with all digit-containing tokens excluded. ⚠️ Stale on disk. |
| `ctry` | Derived | Normalized country label: accent-stripped, lowercased, whitespace-squeezed version of the original `country` field. Open-set — no mapping table, no filtering (§5.5). |

There are **no** phone, email, or other structured contact-identifier columns anywhere in this dataset. The raw challenge data has exactly four fields per record — `entity_id`, `business_name`, `business_address`, `country` — and no phone/email field exists to normalize. Nothing here fabricates one.

There is also no structured decomposition of the address into separate house-number / street-name / city / state / postal-code columns. The only structural split provided is the numeric/non-numeric split (`ad_num` / `ad_alpha`) described above — full address parsing is left to downstream teammates if they need it (and note: geocoding or external address-validation services are prohibited by the challenge rules, §10).

### 4.2 `train_ground_truth.parquet` — 2 columns, both UTF-8 strings, unchanged from raw

| Column | Original / Derived | Description |
|---|---|---|
| `source1_entity_id` | **Original** | The `S1-` entity ID. |
| `matched_entity_ids` | **Original** | Comma-separated list of matching `S2-`/`S3-` entity IDs, **exactly as shipped** — not parsed into a list, not normalized. Empty string (`""`) means that S1 entity has no matches (a singleton). |

---

## 5. Normalization / Preprocessing

This section documents **exactly** what the normalization code does, verified directly against `src/features/normalize.py` and `src/features/translit.py`. Nothing below is claimed unless it is present in that code.

### 5.1 Business name (`normalize_name` → `nm_full` / `nm_core` / `nm_compact` / `nm_skel`)

Applied in order:

1. **Domain-style name detection.** If the raw name matches a TLD pattern (`.com`, `.net`, `.org`, `.co`, `.io`, `.biz`, `.info`, `.in`, `.us`, `.fr`, `.edu`, `.gov`) and contains no space (e.g. `"carneybryant.com"`), the TLD suffix is stripped before further processing. This targets a real corruption pattern in the data where a business name is rewritten as a bare domain-style string.
2. **Indic-script romanization** (only if the text contains characters in the Devanagari/Bengali/Gurmukhi/Gujarati/Oriya/Tamil/Telugu/Kannada/Malayalam Unicode blocks). Implemented as a pure-code phonetic offset table (`src/features/translit.py`) — **no external transliteration library or dataset is used.** All nine Brahmic scripts share ISCII-derived codepoint layouts, so one Devanagari-offset table romanizes all of them by folding each character back to its block-relative offset.
3. **Accent/diacritic stripping** via Unicode NFD decomposition + combining-mark removal (e.g. `"Bryánt"` → `"Bryant"`).
4. **Lowercasing**, `"&"` → `" and "`, all non-alphanumeric characters collapsed to a single space, whitespace squeezed.
5. **Alias substitution** — each token is looked up in a small mined table (`configs/name_alias.json`, currently 4 entries: `company→co`, `corporation→corp`, `incorporated→inc`, `ltd→limited`) that collapses spelling variants of legal-form words to one canonical spelling.
6. Two output forms are then produced:
   - `nm_full` = all tokens after steps 1–5, joined with spaces.
   - `nm_core` = `nm_full` with tokens present in a mined "noise" vocabulary (`configs/name_noise.json`, currently 134 tokens — legal suffixes and generic filler words such as `private`, `llc`, `services`, `center`) removed. If removing noise tokens would leave nothing, the original tokens are kept instead (a name made entirely of legal words is not emptied).
   - `nm_compact` = `nm_core` with spaces removed.
   - `nm_skel` = `nm_compact` with vowels (`a, e, i, o, u`) removed.

**Why this helps entity resolution:** the challenge's stated noise patterns include exactly the things this pipeline targets — legal-suffix inconsistencies (`Corp` vs `Corporation`), punctuation differences (`&` vs `and`), and transliteration. Producing a "core" view (suffixes removed) alongside a "full" view lets downstream similarity features compare names at the right level rather than being thrown off by a legal suffix that one source includes and another omits.

### 5.2 Business address (`normalize_address` → `ad_full` / `ad_core` / `ad_num` / `ad_alpha`)

Applied in order:

1. Same base text pipeline as name normalization: Indic romanization (if present), accent stripping, lowercasing, `&`→`and`, punctuation folded to spaces.
2. **Placeholder-token removal** — literal missing-value placeholders (`null`, `none`, `nan`, `na`, `n/a`, `-`, `--`, `unknown`) are dropped as tokens (these appear in the raw address text itself, e.g. an address ending in `", null,"`).
3. **Digit/letter splitting** — a token like `"23404b"` is split into `"23404"` and `"b"`, unless the trailing letters form an ordinal suffix (`st`/`nd`/`rd`/`th`, which is handled separately).
4. **Ordinal canonicalization** — both digit-ordinal (`"2nd"`) and word-ordinal (`"second"`) forms are canonicalized to a bare digit (`"2"`), because street names in the data alternate between these forms.
5. **Alias substitution** via a mined table (`configs/addr_alias.json`, currently 62 entries) — e.g. `ave→avenue`, `blvd→boulevard`, `mh→maharashtra`, `dl→delhi`, `alabama→al`, `bombay→mumbai`. The canonical direction is *not* uniformly "abbreviation→full" or "full→abbreviation" — it is whichever spelling the clean S1 reference source uses more often for that concept, determined empirically from the training ground truth (so for some pairs it's abbreviation→full, for others full→abbreviation).
6. Two output forms:
   - `ad_full` = all tokens after steps 1–5, joined with spaces.
   - `ad_core` = `ad_full` with tokens present in a mined address "noise" vocabulary (`configs/addr_noise.json`, currently 49 tokens — e.g. leftover state names/abbreviations not caught by aliasing, unit/apartment/box designators) removed.
   - `ad_num` = the subset of tokens that are purely numeric (street numbers, postal/PIN codes, unit numbers), space-joined.
   - `ad_alpha` = the subset of `ad_core` tokens containing **no** digits, space-joined (the alphabetic street/area-name portion).

**Why this helps entity resolution:** the challenge's stated address noise patterns — abbreviation differences (`Rd` vs `Road`), component reordering, missing components, municipal numbering formats — are exactly what steps 2–5 target. Separating numeric tokens (`ad_num`) from alphabetic tokens (`ad_alpha`) matters because street numbers and postal codes are strong, low-noise identity signals, while state/city names are comparatively unstable (frequently abbreviated, aliased, or dropped) — keeping them as separate views lets downstream feature engineering weight them differently instead of treating the whole address as one opaque string.

### 5.3 Country (`normalize_country` → `ctry`)

Accent-stripped, lowercased, whitespace-squeezed version of the raw `country` field. **No mapping table and no filtering of any kind.** `"France"` normalizes to `"france"` exactly the same way `"US"` normalizes to `"us"` — an unseen country label is preserved as its own distinct value rather than being dropped, mapped to `"unknown"`, or rejected. This is deliberate: see §10 on the France/open-set constraint.

### 5.4 ⚠️ Important: some derived columns in the exported files are stale

The mined alias/noise vocabularies (`configs/name_alias.json`, `configs/name_noise.json`, `configs/addr_alias.json`, `configs/addr_noise.json`) were revised **after** the `norm_*.parquet` files on disk were generated. The pipeline re-derives `nm_core`, `nm_compact`, `nm_skel`, `ad_core`, `ad_num`, `ad_alpha` from `nm_full`/`ad_full` at *load time* using whatever the current config tables say — it does not treat the cached values in the Parquet file as final.

**This means: if you load these Parquet files directly (e.g. `pd.read_parquet(...)`), the values stored on disk for `nm_core`, `nm_compact`, `nm_skel`, `ad_core`, `ad_num`, and `ad_alpha` do not match what the current normalization config would produce.**

Measured directly (2,000,000-row sample of `norm_train_source1.parquet`, comparing the on-disk value to a fresh re-derivation using the current `configs/*.json`):

| Column | Rows where on-disk value ≠ current-config value |
|---|---:|
| `nm_core` | 72.6% |
| `nm_compact` | 72.6% |
| `nm_skel` | 72.6% |
| `ad_core` | 32.8% |
| `ad_num` | 22.3% |
| `ad_alpha` | 30.2% |
| `nm_full`, `ad_full`, `ctry`, `entity_id` | 0% (unaffected — these do not depend on the noise tables) |

Example (entity `S1-755362802`, raw name `"Prabhav Business Center"`):
- `nm_full` (reliable): `"prabhav business center"`
- `nm_core` **on disk**: `"prabhav center"`
- `nm_core` **current config would produce**: `"prabhav"`

**Practical guidance:** treat `nm_full`, `ad_full`, `ctry`, and `entity_id` as reliable as-shipped. For `nm_core`, `nm_compact`, `nm_skel`, `ad_core`, `ad_num`, `ad_alpha`, either (a) re-derive them yourself from `nm_full`/`ad_full` using the current `configs/*.json` tables and the logic in §5.1–§5.2 (the reference implementation is `derive_views()` in `src/features/prepare.py`), or (b) pull the repository code and call `features.prepare.load_norm(split, source)`, which performs this re-derivation automatically on every load. Whoever refreshes the Google Drive export should also consider regenerating these files so the on-disk values match the current config directly.

### 5.5 What is explicitly *not* normalized

- No phone number normalization — no phone field exists in the raw data.
- No email normalization — no email field exists.
- No structured address parsing into house-number/street/city/state/postal-code columns — only the numeric/non-numeric split (`ad_num`/`ad_alpha`) is provided.
- No geocoding, address validation, or lookup against any external service — this is prohibited by the challenge rules (§10) and is not done anywhere in this pipeline.
- No deduplication or merging of S2/S3 records — row counts in the normalized files exactly match the raw files.

---

## 6. What Has Already Been Done

**Done:**

- Raw TSV → Parquet format conversion for all six source files and the training ground truth (fast, lossless — see §3.3).
- Per-record text normalization of `business_name` and `business_address` (§5.1–§5.2): Indic-script romanization, accent/diacritic stripping, lowercasing, punctuation folding, domain-suffix stripping, ordinal canonicalization, digit/letter splitting, missing-value placeholder removal, mined alias substitution, mined noise-token removal, numeric-token extraction, alpha-token extraction, consonant-skeleton derivation.
- Country label normalization (casefold + whitespace squeeze only — still fully open-set).
- Mining of four small vocabulary tables (`name_alias`, `name_noise`, `addr_alias`, `addr_noise`) from the **training** ground truth, used to drive the alias/noise steps above.

**Not done (deliberately left open for independent modeling — see §7):**

- No blocking or candidate generation.
- No pairwise record comparison or pair generation.
- No similarity features, TF-IDF, character n-grams, or embeddings computed.
- No ML model of any kind trained on this normalized data as part of this shared package.
- No matching, scoring, thresholding, or decision logic.
- No prediction or submission file generation.
- No structured address parsing (house number / street / city / state / postal code as separate fields).
- No phone/email normalization (fields don't exist in the source data).
- No deduplication of S2/S3 pools.
- No use of any external database, API, geocoding service, or entity-resolution service anywhere in this pipeline (see §10).

---

## 7. Intended Usage

The normalized files are meant as a **common starting point**, not a prescribed pipeline. Teammates are free to build any of the following (or something else entirely) on top of them:

- Blocking / candidate generation (exact-key blocking, token-based blocking, n-gram indexing, nearest-neighbor retrieval, etc.)
- Lexical similarity (edit distance, Jaro-Winkler, token overlap/Jaccard, etc.)
- Character n-gram or word n-gram similarity
- TF-IDF-based retrieval or scoring
- Gradient-boosted tree models (LightGBM, XGBoost, CatBoost, or similar) on hand-engineered pairwise features
- Neural / deep-learning approaches
- Pretrained text embeddings (subject to the challenge's license constraints, §10)
- Hybrid lexical + embedding approaches
- Learning-to-rank formulations
- Binary pair classification
- Custom threshold or decision logic for turning scores into final matches

**There is no required approach.** Any model, feature set, blocking strategy, or decision rule teammates design independently is equally valid on top of this shared layer — the normalized text is meant to save everyone from re-deriving the same cleaning logic, not to constrain what happens next.

---

## 8. Train Data Usage

`norm_train_source1.parquet`, `norm_train_source2.parquet`, `norm_train_source3.parquet`, and `train_ground_truth.parquet` together let you construct supervised pair-level training data:

- **Positive pairs** — for each row in `train_ground_truth.parquet` with a non-empty `matched_entity_ids`, every listed S2/S3 ID paired with that `source1_entity_id` is a true match.
- **Negative pairs** — any (S1, S2/S3) pair that is *not* listed as a match. In principle this is a huge space (millions of S2/S3 records per S1 entity), so in practice negatives should come from whatever candidate-generation/blocking approach you build (§7), not from exhaustively pairing every S1 against every S2/S3 record.
- **Hard negatives** — S2/S3 records that are lexically/structurally close to an S1 entity (e.g. share a name or address token, or pass your blocking step) but are *not* in that entity's ground-truth match list. These are typically far more informative for training a discriminative matcher than randomly sampled negatives, because random negatives are usually trivially distinguishable.
- **Validation splits** — since the *test* set has no ground truth, any internal validation must come from holding out a portion of the *train* Source-1 entities (and their associated ground truth) before building candidates/features/models, so you can estimate generalization honestly.

A few considerations worth keeping in mind (not prescriptive — your implementation choices are your own):

- **Class imbalance.** True matches are a small fraction of all possible (S1, S2/S3) pairs, and even within a well-blocked candidate set, non-matches typically far outnumber matches. How you handle this (sampling ratio, class weighting, loss choice, etc.) is a modeling decision.
- **Candidate generation determines your recall ceiling.** Whatever blocking/candidate strategy you use to generate the pairs you train and evaluate on will cap the best possible recall your matcher can ever achieve — a match that never appears as a candidate can never be predicted. This is a property of *your own* blocking approach, not something the normalized data enforces.
- **One S1 entity may have many true matches, or none.** Train/validation construction should not assume one match per entity.

---

## 9. Test Data Usage

`norm_test_source1.parquet`, `norm_test_source2.parquet`, and `norm_test_source3.parquet` contain **no ground-truth labels** — there is no test equivalent of `train_ground_truth.parquet`. These files are for **inference only**: generate candidates and final matches for every test S1 entity against the test S2/S3 pools, following whichever pipeline you build.

Note also (§10, §11): the test set contains a country (`France`) not present in training. Test `ctry` values are normalized the same way as train (§5.3) — nothing filters or special-cases any country.

---

## 10. Important Challenge Constraints

These come directly from the challenge specification, not from this pipeline's design choices:

- **External data lookup is strictly prohibited.** No external databases, APIs, or services may be used to look up business identities or resolve entities — this includes commercial entity-resolution APIs, government business-registry lookups, geocoding APIs, or any external data augmentation from internet sources.
- **Public pretrained model weights are permitted**, subject to the challenge's constraints: the final model must be released under an MIT or Apache 2.0 license and have at most 8 billion parameters.
- **Test data includes a country (`France`) not present in training.** Training data covers only `US` and `India`.
- **`country` must be treated as an open-set string field.** Do not hard-code, filter, or one-hot-encode against only `{US, India}` — every test entity, France included, must appear in the final submission. (This is exactly why `normalize_country()` performs no mapping or filtering — see §5.3.)
- **Final predictions must follow the official submission format** specified in the challenge README (`source1_entity_id` → comma-separated `matched_entity_ids`, one row per S1 entity, valid S2/S3 IDs only). This normalized dataset does not produce that output — it is purely a preprocessing layer.

---

## 11. What Teammates Should NOT Assume

- **The normalized data is not the final answer.** It is text cleaning, not entity resolution.
- **Normalized equality does not imply entity equality.** Two records with identical `nm_core` or `ad_core` are not guaranteed to be the same business (common/generic names collide) — treat normalized text as a strong *feature*, not a ground-truth signal.
- **Different normalized values do not necessarily imply different entities.** Corruption in the raw data (typos, partial addresses, transliteration drift) can leave two records that *are* the same business with different normalized text — that's precisely the entity-resolution problem this challenge asks you to solve.
- **Candidate generation and matching are separate problems.** Producing a good candidate set (high recall, manageable size) and deciding which candidates are true matches (high precision) are different tasks with different failure modes; don't conflate them.
- **Do not hard-code `US`/`India`.** The country field is open-set — see §10.
- **Do not assume one-to-one matching** unless you have independently verified this from the challenge requirements or the ground-truth data yourself. The challenge statement explicitly allows an S1 entity to match zero, one, or many S2/S3 records (§2).
- **Do not modify the shared/original files.** See §12.
- **Remember the stale-column caveat (§5.4)** before relying on `nm_core`, `nm_compact`, `nm_skel`, `ad_core`, `ad_num`, or `ad_alpha` as shipped in the current Parquet export.

---

## 12. Recommended Experiment Boundary

To keep the shared normalized dataset usable by everyone:

- Treat `data_cache/norm_*.parquet` and `data_cache/train_ground_truth.parquet` as **immutable, read-only inputs**. Do not overwrite them with your own derived columns or filtered subsets.
- Do your own feature engineering, candidate generation, model training, and evaluation in a **separate experiment directory or script** (e.g. your own working folder, not the shared `data_cache/`).
- If you regenerate or modify the normalization logic itself (rather than just consuming its output), coordinate before overwriting the shared files on Google Drive — other teammates' in-progress work depends on the schema and values staying stable and documented.
- Never modify the original challenge files under `student_resource/dataset/` — everything in this document is derived from those files, never a replacement for them.

---

## 13. Reproducibility

| | |
|---|---|
| Repository commit | `b94a68435e079267f84f7c16708c456a3f25c422` |
| Documentation generated | 2026-09-25 |
| Normalization entry point | `src/features/prepare.py` (`prepare()` writes the cached Parquet; `derive_views()` / `load_norm()` re-derive the noise-dependent columns at load time — see §5.4) |
| Core normalization logic | `src/features/normalize.py` (`normalize_name`, `normalize_address`, `normalize_country`) |
| Indic romanization logic | `src/features/translit.py` |
| Config tables used (current versions) | `configs/name_alias.json` (4 entries), `configs/name_noise.json` (134 entries), `configs/addr_alias.json` (62 entries), `configs/addr_noise.json` (49 entries) |
| Source files normalized | `student_resource/dataset/train/train_source{1,2,3}.tsv`, `student_resource/dataset/test/test_source{1,2,3}.tsv`, `student_resource/dataset/train/train_ground_truth.tsv` |
| Checksums | `docs/NORMALIZED_DATASET_CHECKSUMS.sha256` (SHA-256, covers all 6 `norm_*.parquet` files, `train_ground_truth.parquet`, and the 4 config JSON files) |

To verify a file you downloaded from Google Drive matches what's documented here:

```bash
shasum -a 256 -c docs/NORMALIZED_DATASET_CHECKSUMS.sha256
```

(Run from the repository root, with the Parquet/config files in their documented relative paths.)

---

## 14. Quick Start

### Loading a normalized file directly

```python
import pandas as pd

s1_train = pd.read_parquet("data_cache/norm_train_source1.parquet")
s2_train = pd.read_parquet("data_cache/norm_train_source2.parquet")
s3_train = pd.read_parquet("data_cache/norm_train_source3.parquet")

s1_test = pd.read_parquet("data_cache/norm_test_source1.parquet")
s2_test = pd.read_parquet("data_cache/norm_test_source2.parquet")
s3_test = pd.read_parquet("data_cache/norm_test_source3.parquet")

print(s1_train.columns.tolist())
# ['entity_id', 'nm_full', 'nm_core', 'nm_compact', 'nm_skel',
#  'ad_full', 'ad_core', 'ad_num', 'ad_alpha', 'ctry']
```

Or with polars (what the pipeline itself uses):

```python
import polars as pl

s1_train = pl.read_parquet("data_cache/norm_train_source1.parquet")
```

### Loading the ground truth (train only)

```python
import pandas as pd

gt = pd.read_csv(
    "student_resource/dataset/train/train_ground_truth.tsv",
    sep="\t",
)
# or, equivalently, the pre-converted Parquet cache:
gt = pd.read_parquet("data_cache/train_ground_truth.parquet")

print(gt.columns.tolist())
# ['source1_entity_id', 'matched_entity_ids']

# matched_entity_ids is a raw comma-separated string; split it yourself:
gt["matched_list"] = gt["matched_entity_ids"].apply(
    lambda s: s.split(",") if s else []
)
```

### ⚠️ Getting current (non-stale) values for `nm_core` / `nm_compact` / `nm_skel` / `ad_core` / `ad_num` / `ad_alpha`

Per §5.4, the values cached in the Parquet files for these six columns predate the current config tables. If you have the repository available, the safe way to load fully current normalized data is:

```python
import sys
sys.path.insert(0, "src")
from features.prepare import load_norm

s1_train = load_norm("train", "source1")   # re-derives the noise-dependent columns
```

If you only have the Parquet files (e.g. via Google Drive, without the repo), you can rely on `nm_full`, `ad_full`, `ctry`, and `entity_id` as-is, and should treat the other six columns as approximate until re-derived against the current `configs/*.json` tables.

---

## 15. Google Drive Sharing Note

These files are being shared via **Google Drive** purely for team experimentation convenience — not as a replacement for the repository.

- **Download or copy the files to your own workspace** before modifying, filtering, or joining anything onto them. Do not edit the shared Drive originals in place.
- If you need a fresh copy (e.g. to pick up a normalization fix), re-download rather than assuming your local copy is current — check the checksums (§13) if you need to confirm you have the exact documented version.
- Always cross-reference the **schema and version documented here** (§4, §13) against whatever you actually loaded — if column names, row counts, or checksums don't match this document, you likely have a different (older or newer) export and should re-sync before drawing conclusions from a cross-teammate comparison.

---

## 16. Final Summary

These normalized Parquet files provide a **common, reproducible text-preprocessing layer** over the raw ML Challenge 2026 business entity resolution data — Indic-script romanization, accent folding, abbreviation/alias resolution, legal-suffix and noise-token handling, and numeric/alpha address decomposition, applied identically across all six source files (train/test × S1/S2/S3). Row counts and record identities are preserved exactly; no records are added, removed, merged, or deduplicated.

**Candidate generation, feature engineering, modeling, matching, and final decision logic all remain fully open** for independent experimentation by each teammate. This dataset is a shared starting point, not a shared answer — and (per §5.4) even the "starting point" has one documented caveat worth checking before you build on it.
