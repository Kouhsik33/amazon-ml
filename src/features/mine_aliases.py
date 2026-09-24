"""Mine normalisation tables from the training ground truth (no external data).

All evidence comes from files shipped with the challenge.

**Substitution mining.** For every ground-truth matched pair we diff the S1
token multiset against the candidate's. When exactly one token was lost and one
gained, that is evidence of a rewrite: ``avenue``->``ave``, ``maharashtra``->
``mh``, ``bombay``->``mumbai``.

The aggregation rule is **mutual best partner**: ``a`` and ``b`` are aliases
only if ``b`` is the single most frequent replacement observed for ``a`` *and*
``a`` is the most frequent for ``b``, with enough support and confidence on
both sides. This matters. An earlier version grew equivalence classes by
union-find and chained ``bombay->mumbai`` onto ``bangalore->bengaluru`` through
a noisy bridge, merging two different cities, and glued ``center``, ``private``
and ``services`` into one class simply because all three are unstable tokens
that often co-occur in a one-in/one-out diff. A mutual-best graph is a near
matching, so its connected components stay tiny and a coincidental pair loses
to the real partner instead of fusing two classes.

**Instability -> noise vocabulary.** A token a true match keeps is
identity-bearing; one it routinely drops is filler. ``drop_rate = dropped /
(kept + dropped)`` separates ``pediatric`` (0.09) from ``unit`` (0.75). Noise
status additionally requires the token to be *generic* -- present in many
distinct S1 records -- so that a rare business word that happens to be unstable
(``bombay``, ``arbor``) is never discarded as filler.

Nothing is country-conditional: rules are mined from whatever the data holds,
so an unseen country contributes no rules and falls back to the generic
mechanisms (romanisation, accent folding, ordinal and leet canonicalisation).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.io import load, gt_exploded                      # noqa: E402
from features.normalize import CONFIG_DIR, _basic, _NULLS  # noqa: E402
from features.normalize import _canon_ordinal, _split_digit_letter  # noqa: E402

MIN_SUPPORT = 150
MIN_CONF = 0.05          # share of a token's rewrites that must go to its partner
NOISE_MIN_DF = 1_500     # distinct S1 records a token must appear in to be "generic"
NAME_DROP_RATE = 0.45    # legal suffixes / filler; lower-signal generics are left
NOISE_DROP_RATE = {"name": NAME_DROP_RATE, "addr": 0.45}


def _is_abbrev(a: str, b: str) -> bool:
    """True when one token plausibly abbreviates the other.

    Business-name rewrites are overwhelmingly abbreviations (``incorporated``/
    ``inc``, ``limited``/``ltd``), so requiring this relation on the name field
    rejects coincidental pairs such as ``center``/``private`` that survive the
    mutual-best test only because both are unstable filler words. Addresses are
    exempt: genuine city aliases (``bombay``/``mumbai``) share no letters.
    """
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 2 or short == long:
        return False
    if long.startswith(short):
        return True
    it = iter(long)                       # subsequence test: ltd < limited
    return len(short) <= 5 and all(ch in it for ch in short)


def _toks(text: str, address: bool) -> list[str]:
    t = [x for x in _basic(text).split() if x]
    if address:
        t = [x for x in t if x not in _NULLS]
        t = _split_digit_letter(t)
        t = [_canon_ordinal(x) for x in t]
    return t


def mine(n_pairs: int, seed: int, out_dir: Path, exclude_s1: set[str] | None = None) -> None:
    ex = gt_exploded()
    if exclude_s1:
        ex = ex.filter(~pl.col("source1_entity_id").is_in(list(exclude_s1)))
    if ex.height > n_pairs:
        ex = ex.sample(n=n_pairs, seed=seed)

    s1 = (load("train", "source1")
          .rename({"entity_id": "source1_entity_id",
                   "business_name": "n1", "business_address": "a1"}).drop("country"))
    cand = (pl.concat([load("train", "source2"), load("train", "source3")])
            .rename({"entity_id": "cand_id",
                     "business_name": "n2", "business_address": "a2"}).drop("country"))
    df = ex.join(s1, on="source1_entity_id").join(cand, on="cand_id")
    print(f"mining over {df.height:,} matched pairs", flush=True)

    F = ("name", "addr")
    sub = {f: collections.defaultdict(collections.Counter) for f in F}  # a -> Counter(b)
    keep = {f: collections.Counter() for f in F}
    drop = {f: collections.Counter() for f in F}
    add = {f: collections.Counter() for f in F}
    df_s1 = {f: collections.Counter() for f in F}   # document frequency in S1

    for n1, a1, n2, a2 in df.select("n1", "a1", "n2", "a2").iter_rows():
        for field, x, y in (("name", n1, n2), ("addr", a1, a2)):
            is_addr = field == "addr"
            A = collections.Counter(_toks(x, is_addr))
            B = collections.Counter(_toks(y, is_addr))
            lost = list((A - B).elements())
            gained = list((B - A).elements())
            for t, c in (A & B).items():
                keep[field][t] += c
            for t in lost:
                drop[field][t] += 1
            for t in gained:
                add[field][t] += 1
            for t in A:
                df_s1[field][t] += 1
            if len(lost) == 1 and len(gained) == 1 and lost[0] != gained[0]:
                a, b = lost[0], gained[0]
                sub[field][a][b] += 1
                sub[field][b][a] += 1

    out_dir.mkdir(parents=True, exist_ok=True)

    for field in F:
        S = sub[field]
        best = {}
        for a, ctr in S.items():
            b, c = ctr.most_common(1)[0]
            total = sum(ctr.values())
            best[a] = (b, c, c / total)

        alias: dict[str, str] = {}
        accepted = []
        for a, (b, c, conf) in best.items():
            if a >= b:                       # visit each unordered pair once
                continue
            rb = best.get(b)
            if not rb or rb[0] != a:         # not mutual
                continue
            if c < MIN_SUPPORT or conf < MIN_CONF or rb[2] < MIN_CONF:
                continue
            if a.isdigit() != b.isdigit():   # never fold a number into a word
                continue
            if field == "name" and not _is_abbrev(a, b):
                continue
            # canonical form = the spelling the clean source (S1) uses more often
            rep, other = (a, b) if df_s1[field][a] >= df_s1[field][b] else (b, a)
            alias[other] = rep
            accepted.append((c, other, rep))
        accepted.sort(reverse=True)

        # transitive closure is safe now (components are tiny), but resolve
        # anyway so no alias points at another alias.
        for k in list(alias):
            seen = {k}
            v = alias[k]
            while v in alias and v not in seen:
                seen.add(v)
                v = alias[v]
            alias[k] = v

        # ---- pass 2: re-score instability with aliases applied ----
        keep[field] = collections.Counter()
        drop[field] = collections.Counter()
        add[field] = collections.Counter()
        is_addr = field == "addr"
        col_x, col_y = ("a1", "a2") if is_addr else ("n1", "n2")
        for x, y in df.select(col_x, col_y).iter_rows():
            A = collections.Counter(alias.get(t, t) for t in _toks(x, is_addr))
            B = collections.Counter(alias.get(t, t) for t in _toks(y, is_addr))
            for t, c in (A & B).items():
                keep[field][t] += c
            for t in (A - B).elements():
                drop[field][t] += 1
            for t in (B - A).elements():
                add[field][t] += 1

        noise = set()
        for t in set(keep[field]) | set(drop[field]):
            n = keep[field][t] + drop[field][t]
            if n < 300 or df_s1[field][t] < NOISE_MIN_DF:
                continue
            if drop[field][t] / n >= NOISE_DROP_RATE[field]:
                noise.add(t)
        for t, c in add[field].most_common(600):
            # injected by the corruption and essentially absent from clean S1
            if c >= 1500 and df_s1[field][t] < c * 0.20:
                noise.add(t)
        noise -= set(alias)
        noise = sorted(noise)

        (out_dir / f"{field}_alias.json").write_text(
            json.dumps(alias, ensure_ascii=False, indent=0, sort_keys=True))
        (out_dir / f"{field}_noise.json").write_text(
            json.dumps(noise, ensure_ascii=False, indent=0))
        print(f"\n[{field}] {len(alias)} aliases, {len(noise)} noise tokens")
        print(f"  top accepted: {[(o,r,c) for c,o,r in accepted[:18]]}")
        print(f"  noise: {noise[:60]}")

        stats = {
            t: [keep[field][t], drop[field][t], add[field][t], df_s1[field][t]]
            for t in set(keep[field]) | set(drop[field]) | set(add[field])
            if keep[field][t] + drop[field][t] + add[field][t] >= 50
        }
        (out_dir / f"{field}_token_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False))
        print(f"  wrote {field}_token_stats.json ({len(stats)} tokens: keep/drop/add/df)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(CONFIG_DIR))
    ap.add_argument("--exclude-s1", default=None)
    a = ap.parse_args()
    excl = None
    if a.exclude_s1:
        excl = {l.strip() for l in open(a.exclude_s1) if l.strip()}
        print(f"excluding {len(excl):,} held-out S1 ids from mining")
    mine(a.pairs, a.seed, Path(a.out), excl)
