"""Pairwise similarity features for (S1, candidate) pairs.

The features deliberately lean lexical and character-level. The corruption
process measured in E00 is overwhelmingly orthographic -- typos, diacritics,
leetspeak, abbreviation, token insertion/deletion/reorder, transliteration --
so edit distance, token overlap and character n-grams carry most of the signal.
Semantic similarity is only worth its cost where the surface form is destroyed
(full transliteration, domain-ified names), which is a minority of records.

Three groups:

* **name** -- computed over several views (``nm_core`` tokens, ``nm_compact``
  space-free string, ``nm_skel`` consonant skeleton) so that the same pair is
  judged both order-sensitively and order-insensitively.
* **address** -- token and character overlap plus *explicit numeric features*.
  Street numbers and PIN codes are the most identity-bearing part of an
  address (E00 §6) and are easily drowned out inside a whole-string ratio, so
  they get their own comparisons.
* **cross-field** -- country equality, agreement counts and interaction terms.
  Because 5.12% of orphan records collide exactly on name with a real S1
  entity, the model needs an explicit way to express "names agree but the
  address does not", which is what the interaction terms provide.

Everything is computed batch-wise through ``rapidfuzz.process.cpdist`` (C++,
multithreaded) or vectorised Polars list operations; nothing loops in Python
over pairs.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

WORKERS = -1

Q_COLS = ["nm_full", "nm_core", "nm_compact", "nm_skel",
          "ad_full", "ad_core", "ad_num", "ad_alpha", "ctry"]


def attach_sides(pairs: pl.DataFrame, q: pl.DataFrame, c: pl.DataFrame) -> pl.DataFrame:
    """Join normalised columns for both sides onto a pairs frame."""
    qq = q.select([pl.col("entity_id").alias("source1_entity_id")] +
                  [pl.col(x).alias(f"q_{x}") for x in Q_COLS])
    cc = c.select([pl.col("entity_id").alias("cand_id")] +
                  [pl.col(x).alias(f"c_{x}") for x in Q_COLS])
    return pairs.join(qq, on="source1_entity_id", how="inner") \
                .join(cc, on="cand_id", how="inner")


def _cp(a, b, scorer, scale: float = 1.0) -> np.ndarray:
    return (process.cpdist(a, b, scorer=scorer, workers=WORKERS)
            .astype(np.float32) * scale)


def _toks(df: pl.DataFrame, col: str) -> pl.Series:
    return df[col].str.split(" ").list.eval(
        pl.element().filter(pl.element() != "")).list.unique()


def _set_stats(a: pl.Series, b: pl.Series) -> Tuple[np.ndarray, ...]:
    """|A|, |B|, |A n B| for two list-of-string columns."""
    d = pl.DataFrame({"a": a, "b": b}).with_columns(
        pl.col("a").list.len().alias("na"),
        pl.col("b").list.len().alias("nb"),
        pl.col("a").list.set_intersection(pl.col("b")).list.len().alias("ni"),
    )
    return (d["na"].to_numpy().astype(np.float32),
            d["nb"].to_numpy().astype(np.float32),
            d["ni"].to_numpy().astype(np.float32))


def _ratios(na, nb, ni):
    union = np.maximum(na + nb - ni, 1.0)
    mn = np.maximum(np.minimum(na, nb), 1.0)
    mx = np.maximum(np.maximum(na, nb), 1.0)
    return ni / union, ni / mn, ni / mx


def _char_ngram_jaccard(a: List[str], b: List[str], n: int = 3) -> np.ndarray:
    """Jaccard over character n-gram sets. Robust to typos and reordering."""
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if not x or not y:
            continue
        sx = {x[j:j + n] for j in range(max(len(x) - n + 1, 1))}
        sy = {y[j:j + n] for j in range(max(len(y) - n + 1, 1))}
        u = len(sx | sy)
        if u:
            out[i] = len(sx & sy) / u
    return out


def _len_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mx = np.maximum(np.maximum(a, b), 1.0)
    return np.minimum(a, b) / mx


def compute_features(df: pl.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Return (n_pairs, n_features) float32 matrix and the feature names."""
    names: List[str] = []
    cols: List[np.ndarray] = []

    def add(name: str, arr: np.ndarray) -> None:
        names.append(name)
        cols.append(np.asarray(arr, dtype=np.float32))

    q_core = df["q_nm_core"].to_list()
    c_core = df["c_nm_core"].to_list()
    q_cmp = df["q_nm_compact"].to_list()
    c_cmp = df["c_nm_compact"].to_list()
    q_skl = df["q_nm_skel"].to_list()
    c_skl = df["c_nm_skel"].to_list()
    q_adc = df["q_ad_core"].to_list()
    c_adc = df["c_ad_core"].to_list()
    q_adf = df["q_ad_full"].to_list()
    c_adf = df["c_ad_full"].to_list()

    # ---------------- name ----------------
    add("nm_core_exact", df["q_nm_core"] == df["c_nm_core"])
    add("nm_compact_exact", df["q_nm_compact"] == df["c_nm_compact"])
    add("nm_skel_exact", df["q_nm_skel"] == df["c_nm_skel"])
    add("nm_full_exact", df["q_nm_full"] == df["c_nm_full"])

    nm_lev = _cp(q_cmp, c_cmp, fuzz.ratio, 0.01)
    add("nm_lev_compact", nm_lev)
    add("nm_jw_compact", _cp(q_cmp, c_cmp, distance.JaroWinkler.normalized_similarity))
    add("nm_lev_skel", _cp(q_skl, c_skl, fuzz.ratio, 0.01))
    add("nm_tok_sort", _cp(q_core, c_core, fuzz.token_sort_ratio, 0.01))
    add("nm_tok_set", _cp(q_core, c_core, fuzz.token_set_ratio, 0.01))
    add("nm_partial", _cp(q_cmp, c_cmp, fuzz.partial_ratio, 0.01))
    add("nm_char3_jacc", _char_ngram_jaccard(q_cmp, c_cmp, 3))
    add("nm_char4_jacc", _char_ngram_jaccard(q_cmp, c_cmp, 4))

    qa, ca = _toks(df, "q_nm_core"), _toks(df, "c_nm_core")
    na, nb, ni = _set_stats(qa, ca)
    jac, cont, cov = _ratios(na, nb, ni)
    add("nm_tok_jaccard", jac)
    add("nm_tok_containment", cont)
    add("nm_tok_coverage", cov)
    add("nm_tok_shared", ni)
    add("nm_tok_q", na)
    add("nm_tok_c", nb)
    add("nm_tok_diff", np.abs(na - nb))

    lq = np.array([len(x) for x in q_cmp], dtype=np.float32)
    lc = np.array([len(x) for x in c_cmp], dtype=np.float32)
    add("nm_len_ratio", _len_ratio(lq, lc))
    add("nm_len_q", lq)
    add("nm_len_diff", np.abs(lq - lc))

    # ---------------- address ----------------
    q_empty = (df["q_ad_full"].str.len_chars() == 0).to_numpy().astype(np.float32)
    c_empty = (df["c_ad_full"].str.len_chars() == 0).to_numpy().astype(np.float32)
    add("ad_q_empty", q_empty)
    add("ad_c_empty", c_empty)
    either_empty = np.maximum(q_empty, c_empty)

    add("ad_core_exact", df["q_ad_core"] == df["c_ad_core"])
    ad_lev = _cp(q_adc, c_adc, fuzz.ratio, 0.01)
    add("ad_lev_core", ad_lev)
    add("ad_tok_sort", _cp(q_adc, c_adc, fuzz.token_sort_ratio, 0.01))
    add("ad_tok_set", _cp(q_adc, c_adc, fuzz.token_set_ratio, 0.01))
    add("ad_partial", _cp(q_adf, c_adf, fuzz.partial_ratio, 0.01))
    add("ad_char3_jacc", _char_ngram_jaccard(
        [s.replace(" ", "") for s in q_adc], [s.replace(" ", "") for s in c_adc], 3))

    qb, cb = _toks(df, "q_ad_alpha"), _toks(df, "c_ad_alpha")
    na2, nb2, ni2 = _set_stats(qb, cb)
    jac2, cont2, cov2 = _ratios(na2, nb2, ni2)
    add("ad_tok_jaccard", jac2)
    add("ad_tok_containment", cont2)
    add("ad_tok_coverage", cov2)
    add("ad_tok_shared", ni2)
    add("ad_tok_q", na2)
    add("ad_tok_c", nb2)

    # numeric tokens get their own treatment: a street number is the single
    # most identity-bearing address element and is invisible inside a whole
    # string ratio.
    qn, cn = _toks(df, "q_ad_num"), _toks(df, "c_ad_num")
    na3, nb3, ni3 = _set_stats(qn, cn)
    jac3, cont3, _ = _ratios(na3, nb3, ni3)
    add("num_jaccard", jac3)
    add("num_containment", cont3)
    add("num_shared", ni3)
    add("num_q", na3)
    add("num_c", nb3)
    add("num_any_shared", (ni3 > 0).astype(np.float32))
    add("num_none_either", ((na3 == 0) | (nb3 == 0)).astype(np.float32))

    first_q = df["q_ad_num"].str.split(" ").list.first().fill_null("")
    first_c = df["c_ad_num"].str.split(" ").list.first().fill_null("")
    add("num_first_eq", ((first_q == first_c) & (first_q != "")).to_numpy().astype(np.float32))

    lqa = np.array([len(x) for x in q_adc], dtype=np.float32)
    lca = np.array([len(x) for x in c_adc], dtype=np.float32)
    add("ad_len_ratio", _len_ratio(lqa, lca))
    add("ad_len_diff", np.abs(lqa - lca))

    # ---------------- cross-field ----------------
    add("country_eq", df["q_ctry"] == df["c_ctry"])
    add("is_s3", df["cand_id"].str.starts_with("S3-"))

    nm_best = np.maximum(nm_lev, cols[names.index("nm_tok_set")])
    ad_best = np.maximum(ad_lev, cols[names.index("ad_tok_set")])
    add("nm_x_ad", nm_best * ad_best)
    # "name agrees, address does not" -- the orphan-collision failure mode
    add("nm_minus_ad", nm_best - ad_best)
    add("ad_usable", 1.0 - either_empty)
    add("nm_x_ad_usable", nm_best * (1.0 - either_empty))
    add("exact_agreement", cols[names.index("nm_core_exact")]
        + cols[names.index("ad_core_exact")]
        + cols[names.index("num_first_eq")])

    return np.column_stack(cols).astype(np.float32), names
