# E00 — Initial Data Profile

All numbers below are computed from the shipped files only
(`student_resource/dataset/**`). No external source was consulted.
Reproduce with `src/data/io.py`, `src/data/splits.py` and the scripts in
`scripts/profile/`.

## 1. Inventory and scale

| File | Rows | Bytes |
|---|---:|---:|
| `train/train_source1.tsv` | 2,206,821 | 210 MB |
| `train/train_source2.tsv` | 5,034,616 | 489 MB |
| `train/train_source3.tsv` | 5,285,603 | 504 MB |
| `train/train_ground_truth.tsv` | 2,206,821 | 127 MB |
| `test/test_source1.tsv` | 1,732,544 | 175 MB |
| `test/test_source2.tsv` | 4,887,273 | 509 MB |
| `test/test_source3.tsv` | 5,082,316 | 506 MB |
| **total** | **26,435,994** | **2.5 GB** |

This is the single most important planning fact: a brute-force test-set
comparison is 1,732,544 x 9,969,589 = **1.7 x 10^13 pairs**. Candidate
generation is not an optimisation, it is a precondition.

Schema is uniform and clean: `entity_id, business_name, business_address,
country`, exactly 4 tab-separated fields on every one of the 26.4M rows, no
embedded newlines, no quoting, no ragged rows. Entity IDs are unique within
every file and prefixed `S1-`/`S2-`/`S3-`.

## 2. Missing values and duplicates

| | missing name | missing address | duplicate IDs |
|---|---:|---:|---:|
| train S1 | 0 | 0 | 0 |
| train S2 | 0 | 168,967 (3.36%) | 0 |
| train S3 | 0 | 175,916 (3.33%) | 0 |
| test S1 | 0 | 0 | 0 |
| test S2 | 0 | 129,408 (2.65%) | 0 |
| test S3 | 0 | 136,098 (2.68%) | 0 |

`business_name` is never empty. Addresses are empty ~3% of the time in S2/S3
and never in S1 — so **a matcher must be able to decide on the name alone** for
~3% of candidates. Beyond the truly empty ones, the literal strings `null` /
`none` also occur inside addresses and must be treated as missing components.

Duplicates within a source:

| | unique names | unique (name, address) |
|---|---:|---:|
| S1 | 69.75% | **100.000%** |
| S2 | 87.44% | 99.486% |
| S3 | 88.01% | 99.643% |

S1 is exactly deduplicated on (name, address) — consistent with the problem
statement calling it "the deduplicated reference source". Names alone repeat
often (30% of S1 names are shared), which is the first sign that **name
similarity alone cannot identify an entity**.

## 3. Country

| | US | India | France |
|---|---:|---:|---:|
| train S1 | 1,323,633 (59.98%) | 883,188 (40.02%) | — |
| test S1 | 663,106 (38.27%) | 809,986 (46.75%) | **259,452 (14.98%)** |

Two findings:

1. **Country is a hard constraint.** Of all 7,638,365 ground-truth matched
   pairs, **7,638,365 agree on country and 0 disagree.** Partitioning the
   search space by country is therefore free recall-wise and removes ~60% of
   the comparison space. The pipeline applies it as a generic
   `country_s1 == country_candidate` partition, never as a hard-coded label
   list, so `France` forms its own partition automatically.
2. **The train/test country mix shifts substantially** — train is US-majority,
   test is India-majority with 15% France, a label absent from training. This
   is handled in §8.

## 4. Ground truth structure

| statistic | value |
|---|---:|
| S1 entities | 2,206,821 |
| total matched IDs | 7,638,365 |
| — from S2 | 3,693,619 |
| — from S3 | 3,944,746 |
| mean matches per S1 | 3.461 |
| **singletons (0 matches)** | **123,247 (5.585%)** |
| max matches on one S1 | 11 |
| duplicate IDs inside a list | 0 |

Matches per S1 (total / S2-only / S3-only):

| n | total | S2 | S3 |
|---:|---:|---:|---:|
| 0 | 5.58% | 13.04% | 12.07% |
| 1 | 5.40% | 35.76% | 32.46% |
| 2 | 17.00% | 29.58% | 30.29% |
| 3 | 24.05% | 15.13% | 16.88% |
| 4 | 21.94% | 5.40% | 6.58% |
| 5 | 14.59% | 1.09% | 1.60% |
| 6+ | 11.43% | 0 | 0.13% |

S2 contributes at most 5 matches, S3 at most 6. The distributions are close but
not identical (S3 mean 1.787 vs S2 1.674), which is the first evidence for
treating the two sources separately (Phase 13).

**Two structural properties, both verified rather than assumed:**

* **Not one-to-one.** 94.4% of S1 entities have >= 1 match and 89% have >= 2.
  Any one-to-one assignment would be catastrophic here.
* **Many-to-one: every matched S2/S3 record belongs to exactly one S1.**
  The 7,638,365 matched IDs are 7,638,365 *distinct* IDs — zero reuse across
  the entire training set. This is a strong, exploitable constraint (a
  candidate claimed confidently by one S1 is evidence against every other S1),
  and it is *not* the same as one-to-one matching. It will be used as a soft
  competition signal, never as a hard assignment.

Singleton rate is essentially identical in both countries (US 5.583%, India
5.588%), so singleton-ness is generated country-independently — a good sign for
France transfer.

S2 and S3 emptiness are **positively correlated**: if the two were independent,
P(0 matches at all) would be 0.1304 x 0.1207 = 1.57%, but the observed rate is
5.58% — 3.6x higher. Some entities are simply low-visibility across all
sources, which makes a dedicated singleton detector (Phase 12) worthwhile.

Pool coverage: 73.36% of S2 and 74.63% of S3 records are matched to some S1.
The remaining **1,340,997 S2 + 1,340,857 S3 records are orphans** — real-looking
businesses with no S1 parent.

## 5. Field length distributions

| | mean | p25 | p50 | p75 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| S1 name | 24.0 | 18 | 24 | 30 | 42 | 86 |
| S2 name | 25.1 | 19 | 25 | 31 | 48 | 93 |
| S3 name | 25.2 | 18 | 25 | 31 | 50 | 103 |
| S1 addr | 52.0 | 33 | 41 | 69 | 124 | 244 |
| S2 addr | 46.2 | 30 | 37 | 61 | 118 | 231 |
| S3 addr | 46.7 | 35 | 42 | 54 | 115 | 231 |

S2/S3 names are slightly *longer* on average than S1 — consistent with token
insertion (legal suffixes, filler words) being more common than deletion.
Addresses are *shorter* in S2/S3 — consistent with component dropping.

## 6. The corruption process

**Source 1 is clean.** In S1: 0% domain-style names, 0% bracketed suffixes,
~0% junk prefixes, no Indic script in names. All corruption is applied to
S2/S3 only. Normalisation is therefore a one-directional problem: map a noisy
S2/S3 record onto a clean S1 form.

Measured corruption rates in S2/S3 (300k sample each):

| pattern | S2 | S3 |
|---|---:|---:|
| domain-style name (`carneybryant.com`) | 4.03% | 4.01% |
| bracketed legal suffix (`[LLP] EBK Kg`) | 1.97% | 2.03% |
| junk punctuation prefix (`<< Team Ecole`) | 1.19% | 1.14% |
| Indic script in name | ~8.1% | ~5.1% |
| Indic script in address | ~9.2% | ~8.7% |

Name corruptions, from ground-truth aligned pairs:

* legal-suffix add/drop/swap — `Carney and Bryant` / `... Corporation` / `... INC`
* **domain-ification** — `carneybryant.com`, `brightaut0body.com`
* case folding (S2 skews uppercase), whitespace doubling (`Johnson  Rock`)
* **diacritic injection** on a random character — `Bryánt`, `Chúrch`, `ÚNION`
* **leetspeak character swaps** — `8ig-Bakery!`, `brightaut0body`, `de1ta`,
  `at1antic` (mined automatically: `0`->`o`, `1`->`l`, `5`->`s`, `8`->`b`, `6`->`g`)
* **transliteration to Indic scripts** — `इंटरनेशनल पावर` = `International Power`
* character typos — `Evans Sunstnie`, `VHISQUZ`, `Unhaoin`, `Btaberyn`
* token reorder — `LLC Piedmont Group Holdings`, `inc johnson rock`
* token insertion — `BRIGHT AUTO BODY LP SERVICE`, `Big Bakery! Bakery!`, `The ...`
* token deletion — `Electricians Union Local No` (lost `993`)
* DBA / trade names — `Calocalosol f/k/a Evans Sunshine`, `dba`, `aka`

Address corruptions:

* **component reordering** — `Sherwood, 118 Whitewood Drive, AR` vs
  `WHITEWOOD DRIVE, SHERWOOD, AR`
* street-type abbreviation — `Drive`/`Dr`, `Street`/`St`, `Avenue`/`Ave`
* state name <-> abbreviation <-> transliteration — `Arkansas`/`AR`,
  `Maharashtra`/`MH`/`महाराष्ट्र`, `Kerala`/`Keralam`/`KL`
* **city alias substitution** — `Kochi`/`Cochin`, `Bengaluru`/`Bangalore`,
  `Bombay`/`Mumbai`, and neighbourhood swaps (`Phoenix`/`Sunnyslope`,
  `Mount Pleasant`/`Hawthorne`). **The city can legitimately differ between a
  true pair** — city equality must be a feature, never a filter.
* street-number typo / truncation / loss — `4019`->`019`, `118`->`18`,
  `23404`->`23404b`, or dropped entirely
* ordinal alternation — `Second Street` / `2nd St`
* unit designator injection — `PMB 4043`, `# 133`, `Unit 133`, `PO Box`
* missing components and fully empty addresses

Quantified token instability over 1.5M matched pairs
(`drop_rate = dropped / (kept + dropped)` for S1 tokens):

| token | n | drop rate | reading |
|---|---:|---:|---|
| `unit` | 47,438 | 0.752 | unit designators are noise |
| `maharashtra` | 43,383 | 0.616 | state is highly unstable *before* aliasing |
| `street` | 70,137 | 0.519 | street type is unstable |
| `tx`, `ny`, `nc`, ... | ~20k each | ~0.505 | state abbreviations likewise |
| `no` | 101,103 | 0.180 | house-number marker is stable |
| `pediatric` | 3,602 | 0.086 | rare content words are the identity signal |
| `way` | 7,063 | 0.046 | |

The lesson that drives normalisation: **numbers and rare content tokens carry
identity; type words, unit designators and administrative geography do not.**
After the mined aliases are applied, state instability largely disappears
(`maharashtra`/`mh`/`महाराष्ट्र` collapse to one token), which is why the
alias tables are mined *before* the noise list is scored.

## 7. Discriminative power of the signals

The decisive measurement for architecture choice, on 200k-record samples:

| S2 records whose exact normalised name + country equals **some** S1 | share |
|---|---:|
| among **matched** S2 records | 30.24% |
| among **orphan** (unmatched) S2 records | **5.12%** |

~68,000 orphan S2 records in training collide exactly on normalised name with a
real S1 entity, and they are *not* matches. A name-only matcher would emit all
of them as false positives. Under F0.5 — where precision counts double and a
false positive on a singleton costs a full 1.0 — that is the dominant error
mode. **Address is the disambiguator, and the pairwise model must be able to
veto on address disagreement even when the name matches exactly.**

## 8. Train/test distribution shift

| | train | test |
|---|---:|---:|
| S2 / S1 ratio | 2.281 | 2.821 |
| S3 / S1 ratio | 2.395 | 2.933 |
| S2 records hitting some S1 by exact name | 23.43% | 20.39% |
| S3 records hitting some S1 by exact name | 24.56% | 21.68% |

The test set has ~24% more S2/S3 records per S1 entity. Two hypotheses explain
it: (a) more orphan records, or (b) more matches per entity. Under (a) the
predicted exact-name hit rate is 20.0%, under (b) 23.5%; the measured 20.4% /
21.7% selects **(a) — more orphans**. Estimated test pool composition is ~59%
matched / ~41% orphan, against 74% / 26% in training.

Consequence: per-query competition is nearly unchanged (a test query faces
9.97M pool records, a validation query faces 10.32M), so validation remains
representative — but the pool holds proportionally more orphan look-alikes, so
**test precision will be slightly worse than validation precision at a fixed
threshold.** This argues for erring on the precision side when the threshold
search is close, and is recorded here so the bias is not mistaken for a bug
later.

The second shift is **France**, 15% of test S1 entities, absent from training.
It is a genuine open-set problem: no ground truth exists for French corruption
patterns. Mitigations, none of which involve external data:
* no country literal appears anywhere in the pipeline; country is used only as
  a partition key and an equality feature;
* the mined alias/noise tables key on tokens, so France contributes none and
  simply falls back to the generic mechanisms (accent folding, ordinal
  canonicalisation, abbreviation-prefix logic, IDF from the test corpus itself);
* a **leave-one-country-out** validation fold (train on US, evaluate on India,
  and vice versa) gives an honest lower bound on transfer to an unseen country.
  This is the closest measurable proxy for France and is reported alongside the
  in-distribution score.

French records do show the same corruption families (`SCI Ptit Àmicale` —
diacritic injection; `Fractales Amis Groupe S.A.S` — legal suffix;
`23 Rue Icmre` — typo in `Icare`/similar; `<< Team Ecole` — junk prefix), which
supports the assumption that the generator is country-agnostic.

## 9. Validation protocol

Splitting is by **Source-1 entity**, never by pair and never by S2/S3 record,
because the competition task is "given an unseen S1 entity and the complete
S2/S3 pools, return its matches". Held-out entities are scored against the
**full 10.32M-record pool**; restricting the pool to validation entities' own
matches would delete every orphan and every competing business and inflate both
blocking recall and precision.

| fold | entities | singleton rate | mean matches | frac US |
|---|---:|---:|---:|---:|
| train | 2,106,818 | 5.585% | 3.461 | 0.5998 |
| dev | 40,004 | 5.587% | 3.463 | 0.5998 |
| val | 59,999 | 5.585% | 3.463 | 0.5998 |

Stratified on (country, match-count bucket), so every fold reproduces the
singleton rate and the one-to-many shape to 3 decimal places. `dev` is for
threshold/calibration search, `val` is touched once per architecture — F0.5 is
sharp in the decision-layer knobs, so tuning and reporting on one fold would
overstate the result.

Leakage control: alias/noise mining excludes all 100,003 dev+val entities
(`--exclude-s1`); IDF and token-DF statistics are unsupervised corpus counts
computed exactly as they will be at test time; the many-to-one constraint is
applied at inference from model scores only, never from ground-truth
assignments.

## 10. Consequences for the architecture

1. Scale forces blocking; blocking sets the recall ceiling. Measure the
   **macro-F0.5 ceiling**, not just pair recall — the metric is per-entity, so
   losing one match on a 2-match entity costs far more than on a 6-match one.
2. Country partitions the problem for free and losslessly.
3. Name alone is not identifying (5.12% orphan collision rate). Address
   features are mandatory for precision.
4. ~3% of candidates have no address at all; the model must degrade gracefully.
5. Corruption is lexical and character-level far more than semantic. Character
   n-grams, edit distance and token overlap should dominate; dense embeddings
   must earn their place against that baseline rather than be assumed better.
6. The many-to-one property is a genuine, verified structural constraint worth
   exploiting as a competition/margin signal.
7. Singletons are only 5.6% of entities but each false positive on one costs a
   full 1.0 — they get a dedicated decision stage.
