"""Candidate generation (blocking).

Exhaustive comparison is 1.73M x 9.97M = 1.7e13 pairs on the test set, so the
candidate set decides the recall ceiling of everything downstream. The design
here is a **union of cheap, complementary keys** resolved by hash join, rather
than one clever similarity index: each key family is blind to a different
corruption, and the union covers what no single family can.

Key families (every key is scoped by country, which the ground truth shows is
never violated -- all 7,638,365 matched pairs agree on country):

    B1 nm_compact      exact normalised core name, spaces removed.
                       Survives spacing noise, punctuation, legal-suffix
                       add/drop and the 'carneybryant.com' domain form.
    B2 nm_skel         consonant skeleton of B1. Survives vowel typos,
                       diacritic injection and Indic romanisation drift
                       ('limiteda' vs 'limited').
    B3 rare name token a single low-document-frequency name token. The
                       workhorse: survives token insertion, deletion and
                       reordering, which all break whole-string keys.
    B4 token x number  a name token conjoined with the first address number.
                       Rescues common-token names ('Prime Money') that B3
                       cannot retrieve without exploding the posting list.
    B5 number x street address number conjoined with a rare street token.
                       The fallback when the name is destroyed beyond
                       recognition but the address survives.
    B6 name prefix     first 8 chars of nm_compact, for trailing-token loss
                       ('Electricians Union Local No 993' -> '... Local No').

A key whose candidate posting list exceeds ``max_posting`` is dropped: such a
key is uninformative (it would admit hundreds of candidates for one S1) and
costs the most to join. This is the reduction-ratio lever.

Everything is per-country, which bounds peak memory and enforces the country
constraint for free. Nothing is conditional on *which* country: an unseen
label simply forms its own partition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import polars as pl


@dataclass
class BlockingConfig:
    rare_df_max: int = 4_000      # a name token is "rare" below this pool DF
    max_posting: int = 120        # drop keys with more posting-list entries
    max_cand_per_s1: int = 40     # keep the top-K by key agreement; 0 = unlimited
    use_b1: bool = True
    use_b2: bool = True
    use_b3: bool = True
    use_b4: bool = True
    use_b5: bool = True
    use_b6: bool = True
    b4_max_tokens: int = 2        # rarest-N name tokens used in the pair key
    b5_max_tokens: int = 2
    max_nums: int = 3             # address numbers used in pair keys
    use_b7: bool = True           # rare name token x rare address token


def _tok_frame(df: pl.DataFrame, col: str, idx: str) -> pl.DataFrame:
    """(idx, token) long frame from a space-joined string column."""
    return (
        df.select(pl.col(idx), pl.col(col).str.split(" ").alias("tok"))
        .explode("tok")
        .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
    )


def _nums(df: pl.DataFrame, idx: str, k: int) -> pl.DataFrame:
    """All address numbers (capped at k), not just the first.

    Using only the first number was a real recall leak: address components are
    routinely reordered, so S1's leading '13' faces the candidate's leading
    '24' even when both records share '24' further along. Every number is a
    separate retrieval chance.
    """
    return (df.select(pl.col(idx),
                      pl.col("ad_num").str.split(" ").list.head(k).alias("num"))
              .explode("num")
              .filter(pl.col("num").is_not_null() & (pl.col("num") != "")))


def _keyed(frame: pl.DataFrame, idx: str, fam: int, expr: pl.Expr) -> pl.DataFrame:
    return frame.select(
        pl.col(idx),
        (expr.hash(seed=fam) ^ pl.lit(fam, dtype=pl.UInt64)).alias("k"),
    )


def _build_keys(df: pl.DataFrame, idx: str, cfg: BlockingConfig,
                rare: pl.DataFrame | None) -> pl.DataFrame:
    """Emit (idx, key) for every blocking key a record participates in."""
    parts: List[pl.DataFrame] = []

    if cfg.use_b1:
        parts.append(_keyed(df.filter(pl.col("nm_compact") != ""), idx, 1,
                            pl.col("nm_compact")))
    if cfg.use_b2:
        parts.append(_keyed(df.filter(pl.col("nm_skel").str.len_chars() >= 4), idx, 2,
                            pl.col("nm_skel")))
    if cfg.use_b6:
        parts.append(_keyed(df.filter(pl.col("nm_compact").str.len_chars() >= 8), idx, 6,
                            pl.col("nm_compact").str.slice(0, 8)))

    name_tok = _tok_frame(df, "nm_core", idx)
    if rare is not None:
        name_tok = name_tok.join(rare, on="tok", how="left").with_columns(
            pl.col("df").fill_null(0)
        )
    else:
        name_tok = name_tok.with_columns(pl.lit(0, dtype=pl.UInt32).alias("df"))

    if cfg.use_b3:
        parts.append(_keyed(name_tok.filter(pl.col("df") <= cfg.rare_df_max), idx, 3,
                            pl.col("tok")))

    if cfg.use_b4:
        # rarest few name tokens x the first address number
        rarest = (name_tok.sort("df")
                  .group_by(idx, maintain_order=True)
                  .head(cfg.b4_max_tokens))
        nums = _nums(df, idx, cfg.max_nums)
        parts.append(_keyed(rarest.join(nums, on=idx), idx, 4,
                            pl.col("tok") + pl.lit("|") + pl.col("num")))

    if cfg.use_b5:
        addr_tok = _tok_frame(df, "ad_alpha", idx)
        if rare is not None:
            addr_tok = addr_tok.join(rare.rename({"df": "adf"}), on="tok", how="left") \
                               .with_columns(pl.col("adf").fill_null(0))
            addr_tok = addr_tok.sort("adf").group_by(idx, maintain_order=True) \
                               .head(cfg.b5_max_tokens)
        nums = _nums(df, idx, cfg.max_nums)
        parts.append(_keyed(addr_tok.join(nums, on=idx), idx, 5,
                            pl.col("num") + pl.lit("|") + pl.col("tok")))

        if cfg.use_b7:
            # rarest name token x rarest address token: the only key that fires
            # when the name is transliterated beyond recognition AND the
            # address carries no usable number.
            rn = name_tok.sort("df").group_by(idx, maintain_order=True).head(1)
            ra = addr_tok.group_by(idx, maintain_order=True).head(1)
            parts.append(_keyed(rn.join(ra, on=idx, suffix="_a"), idx, 7,
                                pl.col("tok") + pl.lit("~") + pl.col("tok_a")))

    return pl.concat([p.select(idx, "k") for p in parts]).unique()


def _token_df(df: pl.DataFrame) -> pl.DataFrame:
    """Document frequency of name/address tokens over the candidate pool."""
    a = _tok_frame(df, "nm_core", "ci").select("tok")
    b = _tok_frame(df, "ad_alpha", "ci").select("tok")
    return (pl.concat([a, b]).group_by("tok").len()
            .rename({"len": "df"}).with_columns(pl.col("df").cast(pl.UInt32)))


def generate_candidates(queries: pl.DataFrame, pool: pl.DataFrame,
                        cfg: BlockingConfig | None = None,
                        verbose: bool = True) -> pl.DataFrame:
    """Return (source1_entity_id, cand_id) candidate pairs.

    ``queries`` and ``pool`` are normalised frames (see features.prepare) that
    must carry ``entity_id`` and ``ctry``. Blocking runs independently inside
    each country partition.
    """
    cfg = cfg or BlockingConfig()
    out: List[pl.DataFrame] = []
    countries = pool["ctry"].unique().to_list()

    for c in countries:
        q = queries.filter(pl.col("ctry") == c)
        p = pool.filter(pl.col("ctry") == c)
        if q.height == 0 or p.height == 0:
            continue
        keep = ["entity_id", "nm_core", "nm_compact", "nm_skel", "ad_num", "ad_alpha"]
        q = q.select(keep).with_row_index("qi")
        p = p.select(keep).with_row_index("ci")

        rare = _token_df(p)
        qk = _build_keys(q, "qi", cfg, rare)
        pk = _build_keys(p, "ci", cfg, rare)

        # drop uninformative keys: huge posting lists add candidates that are
        # never true matches and dominate join cost.
        post = pk.group_by("k").len()
        good = post.filter(pl.col("len") <= cfg.max_posting).select("k")
        pk = pk.join(good, on="k", how="inner")

        # Join multiplicity == how many independent key families agreed on this
        # pair. It is free (a by-product of the join) and strongly ranks true
        # matches above incidental collisions, so it is the cap criterion --
        # capping by key agreement loses far less recall than tightening the
        # keys themselves.
        pairs = (qk.join(pk, on="k", how="inner")
                   .group_by("qi", "ci").len().rename({"len": "n_keys"}))
        if cfg.max_cand_per_s1:
            pairs = (pairs.sort("n_keys", descending=True)
                          .group_by("qi", maintain_order=True)
                          .head(cfg.max_cand_per_s1))
        pairs = (pairs
                 .join(q.select("qi", pl.col("entity_id").alias("source1_entity_id")), on="qi")
                 .join(p.select("ci", pl.col("entity_id").alias("cand_id")), on="ci")
                 .select("source1_entity_id", "cand_id", "n_keys"))
        if verbose:
            print(f"  [{c}] queries={q.height:,} pool={p.height:,} "
                  f"keys(q)={qk.height:,} keys(p)={pk.height:,} pairs={pairs.height:,} "
                  f"({pairs.height / max(q.height,1):.1f}/query)", flush=True)
        out.append(pairs)

    return pl.concat(out) if out else pl.DataFrame(
        {"source1_entity_id": [], "cand_id": [], "n_keys": []},
        schema={"source1_entity_id": pl.Utf8, "cand_id": pl.Utf8, "n_keys": pl.UInt32})
