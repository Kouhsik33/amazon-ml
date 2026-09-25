"""IDF-weighted sparse retrieval with every parameter exposed.

Algorithmically equivalent to the Approach2 baseline
(keys -> IDF inverted index -> sum-of-idf^2 scoring -> POOL -> rerank -> TOP_K)
but written against RetrievalConfig so a controlled experiment can vary exactly
one field and hold the rest fixed.

Cost choices that keep this runnable on a 16GB box:
  * postings are retained only for keys the query side actually asks for, so the
    index never materialises ~10M Python lists;
  * per-query scoring accumulates over concatenated postings via bincount on a
    small inverse, instead of a |B|-length vector per entity.
"""
from __future__ import annotations

import gc
from collections import defaultdict
from itertools import combinations

import numpy as np
import polars as pl

from .config import RetrievalConfig

UNIQ = lambda col, alias: (
    pl.col(col).str.split(" ")
      .list.eval(pl.element().filter(pl.element() != ""))
      .list.unique().alias(alias)
)
_H = lambda seq: pl.Series(seq).hash().to_numpy()


def lookup(uniq: np.ndarray, keep: np.ndarray, hashes) -> np.ndarray:
    """Vocabulary positions for `hashes`, EXACT matches only.

    searchsorted alone returns an insertion point, so a key absent from the B
    vocabulary aliases to a neighbouring key and would contribute that key's
    postings. The `uniq[pos] == h` test is what makes the lookup correct.
    """
    h = np.asarray(list(hashes), dtype=np.uint64) if not isinstance(hashes, np.ndarray) \
        else hashes.astype(np.uint64, copy=False)
    if h.size == 0:
        return np.empty(0, np.int64)
    pos = uniq.searchsorted(h)
    ok = pos < len(uniq)
    pos = pos[ok]; h = h[ok]
    exact = uniq[pos] == h            # <- rejects aliased misses
    pos = pos[exact]
    return pos[keep[pos]].astype(np.int64)


def corpus_df(paths: list[str], col: str, ctry: str) -> tuple[dict, int]:
    """Document frequency over one country across the given parquet files."""
    parts, n_all = [], 0
    for p in paths:
        lf = pl.scan_parquet(p).filter(pl.col("ctry") == ctry)
        n_all += lf.select(pl.len()).collect().item()
        parts.append(lf.select(col).with_columns(UNIQ(col, "t")).select("t")
                       .explode("t").drop_nulls().filter(pl.col("t") != "")
                       .group_by("t").len())
    d = pl.concat([p.collect(streaming=True) for p in parts]) \
          .group_by("t").agg(pl.col("len").sum().alias("df"))
    return dict(zip(d["t"].to_list(), d["df"].to_list())), n_all


def _core(tokens, dfc, max_df, top):
    """Approach2 _content: DF-gate, then the `top` rarest by (df, token)."""
    u = {t for t in tokens if dfc.get(t, 0) <= max_df}
    return sorted(u, key=lambda t: (dfc.get(t, 0), t))[:top]


def name_keys(s: str, dfc: dict, max_df: float, cfg: RetrievalConfig) -> list[str]:
    tk = s.split()
    k = {"n" + t for t in tk if len(t) >= 3}
    k |= {"b" + a + "_" + b for a, b in combinations(_core(tk, dfc, max_df, cfg.name_core_top), 2)}
    k |= {"p" + t[:4] for t in tk if len(t) >= 5}
    k |= {"x" + t[-4:] for t in tk if len(t) >= 6}
    return sorted(k)


def addr_keys(s: str, dfc: dict, max_df: float, cfg: RetrievalConfig) -> list[str]:
    tk = s.split()
    k = {"a" + t for t in tk if len(t) >= 4 and not t.isdigit()}
    k |= {"c" + a + "_" + b for a, b in combinations(_core(tk, dfc, max_df, cfg.addr_core_top), 2)}
    k |= {"z" + t for t in tk if t.isdigit() and len(t) >= 5}
    return sorted(k)


class CountryIndex:
    """Inverted IDF index over one (country, source) B partition."""

    def __init__(self, b_nm, b_ad, dfc_n, dfc_a, max_df_n, max_df_a,
                 cfg: RetrievalConfig, query_keys: set[int], log=print):
        self.cfg = cfg
        n_b = len(b_nm)
        self.n_b = n_b
        cap = max(cfg.min_cap, int(cfg.cap_frac * n_b))
        self.cap = cap

        rn, kn, ra, ka = [], [], [], []
        CH = 300_000
        for s in range(0, n_b, CH):
            e = min(s + CH, n_b)
            r1, k1, r2, k2 = [], [], [], []
            for i in range(s, e):
                for x in name_keys(b_nm[i], dfc_n, max_df_n, cfg):
                    r1.append(i); k1.append(x)
                for x in addr_keys(b_ad[i], dfc_a, max_df_a, cfg):
                    r2.append(i); k2.append(x)
            rn.append(np.asarray(r1, np.int32)); kn.append(_H(k1))
            ra.append(np.asarray(r2, np.int32)); ka.append(_H(k2))
        rn = np.concatenate(rn); kn = np.concatenate(kn)
        ra = np.concatenate(ra); ka = np.concatenate(ka)

        uniq, inv, cnt = np.unique(np.concatenate([kn, ka]), return_inverse=True,
                                   return_counts=True)
        keep = cnt <= cap
        idf2 = (np.log(n_b / cnt).astype(np.float64)) ** 2
        iv_n, iv_a = inv[:len(kn)], inv[len(kn):]
        del inv; gc.collect()

        mn, ma = keep[iv_n], keep[iv_a]
        self.mass_b = (np.bincount(rn[mn], weights=idf2[iv_n[mn]], minlength=n_b)
                       + np.bincount(ra[ma], weights=idf2[iv_a[ma]], minlength=n_b))

        want = np.unique(lookup(uniq, keep, np.fromiter(query_keys, np.uint64, len(query_keys))))
        post = {}
        for iv, rr, msk in ((iv_n, rn, mn), (iv_a, ra, ma)):
            sel = np.flatnonzero(msk)
            kk, rv = iv[sel].astype(np.int64), rr[sel]
            hit = np.isin(kk, want, assume_unique=False)
            kk, rv = kk[hit], rv[hit]
            if kk.size == 0:
                continue
            o = np.argsort(kk, kind="stable")
            kk, rv = kk[o], rv[o]
            starts = np.flatnonzero(np.r_[True, kk[1:] != kk[:-1]])
            ends = np.r_[starts[1:], kk.size]
            for st, en in zip(starts, ends):
                k = int(kk[st])
                post[k] = np.concatenate([post[k], rv[st:en]]) if k in post else rv[st:en]
        self.post = post
        self.uniq, self.keep, self.idf2 = uniq, keep, idf2
        log(f"    index n_b={n_b:,} cap={cap} vocab={len(uniq):,} "
            f"kept={keep.mean():.2%} queried_keys={len(self.post):,}")
        del rn, kn, ra, ka, iv_n, iv_a, mn, ma; gc.collect()

    def search_ranked(self, q_keys) -> tuple[np.ndarray, np.ndarray]:
        """POOLed candidates for one S1, SORTED by rerank score (best first).

        Returning the full ranked pool lets one scoring pass answer every TOP_K
        at once: truncating this list at k is exactly the top-k the baseline's
        argpartition would keep.
        """
        cfg = self.cfg
        ks = [int(k) for k in lookup(self.uniq, self.keep, q_keys)]
        arrs = [self.post[k] for k in ks if k in self.post]
        if not arrs:
            return np.empty(0, np.int32), np.empty(0, np.float64)
        mass_a = sum(self.idf2[k] for k in ks)
        rows = np.concatenate(arrs)
        w = np.concatenate([np.full(len(self.post[k]), self.idf2[k]) for k in ks if k in self.post])
        ur, ivr = np.unique(rows, return_inverse=True)
        sc = np.bincount(ivr, weights=w)
        if len(ur) > cfg.pool:
            sel = np.argpartition(-sc, cfg.pool)[:cfg.pool]
            ur, sc = ur[sel], sc[sel]
        if cfg.rerank == "jac":
            r = sc / (mass_a + self.mass_b[ur] - sc + 1e-6)
        else:
            r = sc / np.sqrt(mass_a * self.mass_b[ur] + 1e-6)
        order = np.argsort(-r)
        return ur[order], r[order]

    def search(self, q_keys: np.ndarray, top_k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (row_ids, scores) for one S1, already POOLed and reranked."""
        cfg = self.cfg
        top_k = cfg.top_k if top_k is None else top_k
        ks = [int(k) for k in lookup(self.uniq, self.keep, q_keys)]
        arrs = [self.post[k] for k in ks if k in self.post]
        if not arrs:
            return np.empty(0, np.int32), np.empty(0, np.float64)
        mass_a = sum(self.idf2[k] for k in ks)
        rows = np.concatenate(arrs)
        w = np.concatenate([np.full(len(self.post[k]), self.idf2[k]) for k in ks if k in self.post])
        ur, ivr = np.unique(rows, return_inverse=True)
        sc = np.bincount(ivr, weights=w)
        if len(ur) > cfg.pool:
            sel = np.argpartition(-sc, cfg.pool)[:cfg.pool]
            ur, sc = ur[sel], sc[sel]
        if len(ur) > top_k:
            if cfg.rerank == "jac":
                r = sc / (mass_a + self.mass_b[ur] - sc + 1e-6)
            else:
                r = sc / np.sqrt(mass_a * self.mass_b[ur] + 1e-6)
            sel = np.argpartition(-r, top_k)[:top_k]
            ur, sc = ur[sel], sc[sel]
        return ur, sc
