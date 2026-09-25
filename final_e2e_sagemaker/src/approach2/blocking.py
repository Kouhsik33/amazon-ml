"""Candidate generation: IDF-weighted multi-key blocking implemented with sparse matrix products.

For one country and one target source B (S2 or S3):
  * every record gets a set of hashed keys (name tokens, unordered token pairs, prefixes/suffixes, address tokens, ...)
  * the vocabulary and document frequencies come from B only; keys that are too frequent are dropped
  * S1 chunk @ inverted-index(B) gives sum-of-IDF scores for all pairs that share a key; the top-K per S1 row are kept
"""
import itertools
import math
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

NAME_GENERIC_SHARE = 0.05
ADDR_GENERIC_SHARE = 0.03


def token_df(strings):
    cnt = Counter()
    for s in strings:
        cnt.update(set(s.split()))
    return cnt


def _content(toks, dfc, max_df, top):
    u = {t for t in toks if dfc.get(t, 0) <= max_df}
    return sorted(u, key=lambda t: (dfc.get(t, 0), t))[:top]


def name_keys(s, ctx):
    dfc, max_df = ctx
    toks = s.split()
    keys = {"n" + t for t in toks if len(t) >= 3}
    core = _content(toks, dfc, max_df, 6)
    keys |= {"b" + a + "_" + b for a, b in itertools.combinations(core, 2)}
    keys |= {"p" + t[:4] for t in toks if len(t) >= 5}
    keys |= {"x" + t[-4:] for t in toks if len(t) >= 6}
    return keys


def addr_keys(s, ctx):
    dfc, max_df = ctx
    toks = s.split()
    keys = {"a" + t for t in toks if len(t) >= 4 and not t.isdigit()}
    core = _content(toks, dfc, max_df, 8)
    keys |= {"c" + a + "_" + b for a, b in itertools.combinations(core, 2)}
    keys |= {"z" + t for t in toks if t.isdigit() and len(t) >= 5}
    return keys


def _hash(strs):
    return pd.util.hash_array(np.array(strs, dtype=object))


def gen_keys(strings, fn, ctx, chunk=200_000):
    rows_l, keys_l = [], []
    for s in range(0, len(strings), chunk):
        rows, keys = [], []
        for i, st in enumerate(strings[s:s + chunk], s):
            ks = fn(st, ctx)
            keys.extend(ks)
            rows.extend([i] * len(ks))
        rows_l.append(np.asarray(rows, dtype=np.int32))
        keys_l.append(_hash(keys) if keys else np.zeros(0, np.uint64))
    return np.concatenate(rows_l), np.concatenate(keys_l)


def make_ctx(all_names, all_addrs):
    n = len(all_names)
    return (token_df(all_names), NAME_GENERIC_SHARE * n), (token_df(all_addrs), ADDR_GENERIC_SHARE * n)


def _csr(rows, cols, data, nrows, ncols):
    """CSR from row-sorted entries."""
    indptr = np.zeros(nrows + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=nrows), out=indptr[1:])
    return sp.csr_matrix((data, cols.astype(np.int32), indptr), shape=(nrows, ncols))


def build_index(b_keys, n_b, cap):
    """b_keys = (rows_n, keys_n, rows_a, keys_a). Returns the inverted indexes and vocabulary."""
    rn, kn, ra, ka = b_keys
    allk = np.concatenate([kn, ka])
    uniq, inv, cnt = np.unique(allk, return_inverse=True, return_counts=True)
    del allk
    keep = cnt <= cap
    idf = np.log(n_b / cnt).astype(np.float32)
    V = len(uniq)
    out = []
    for rows, cols in ((rn, inv[: len(kn)]), (ra, inv[len(kn):])):
        m = keep[cols]
        m_rows, m_cols = rows[m], cols[m]
        M = _csr(m_rows, m_cols, idf[m_cols], n_b, V)
        out.append(M.T.tocsr())
    return out[0], out[1], uniq, keep, idf


def _a_chunk(rows, keys, lo, hi, uniq, keep, idf, ncols):
    a, b = np.searchsorted(rows, [lo, hi])
    k, r = keys[a:b], rows[a:b] - lo
    pos = np.searchsorted(uniq, k)
    pos[pos >= len(uniq)] = 0
    hit = (uniq[pos] == k) & keep[pos]
    return _csr(r[hit], pos[hit], idf[pos[hit]], hi - lo, ncols)


def block_source(a_keys, n_a, b_keys, n_b, top_k, cap_frac, min_cap, row_chunk=2000, pool=60, rerank="jac", log=print):
    """Returns DataFrame(i, j, sn, sa, sj): i = row in A (S1), j = row in B.

    The `pool` best pairs per row by raw shared-IDF sum are re-ranked by weighted Jaccard
    sj = shared / (mass_A + mass_B - shared), with squared-IDF masses (penalizes long, generic records); the top_k by sj are kept.
    """
    cap = max(min_cap, int(cap_frac * n_b))
    BnT, BaT, uniq, keep, idf = build_index(b_keys, n_b, cap)
    V = len(uniq)
    mass_b = (np.asarray(BnT.power(2).sum(axis=0)).ravel() + np.asarray(BaT.power(2).sum(axis=0)).ravel()).astype(np.float32)
    log(f"    index built: |B|={n_b:,} vocab={V:,} kept_keys={int(keep.sum()):,} cap={cap}")
    arn, akn, ara, aka = a_keys
    parts = []
    for s in range(0, n_a, row_chunk):
        e = min(s + row_chunk, n_a)
        An = _a_chunk(arn, akn, s, e, uniq, keep, idf, V)
        Aa = _a_chunk(ara, aka, s, e, uniq, keep, idf, V)
        mass_a = (np.asarray(An.power(2).sum(axis=1)).ravel() + np.asarray(Aa.power(2).sum(axis=1)).ravel()).astype(np.float32)
        Sn = (An @ BnT).tocsr()
        Sa = (Aa @ BaT).tocsr()
        P = (Sn + Sa).tocsr()
        indptr, ind, dat = P.indptr, P.indices, P.data
        rr, cc = [], []
        for i in range(P.shape[0]):
            lo, hi = indptr[i], indptr[i + 1]
            if lo == hi:
                continue
            d, cols = dat[lo:hi], ind[lo:hi]
            if len(d) > pool:
                sel = np.argpartition(-d, pool)[:pool]
                d, cols = d[sel], cols[sel]
            if len(d) > top_k:
                sj = d / (mass_a[i] + mass_b[cols] - d + 1e-6) if rerank == "jac" else d / np.sqrt(mass_a[i] * mass_b[cols] + 1e-6)
                sel = np.argpartition(-sj, top_k)[:top_k]
                cols = cols[sel]
            cc.append(cols)
            rr.append(np.full(len(cols), i, dtype=np.int32))
        if rr:
            r, c = np.concatenate(rr), np.concatenate(cc)
            sn = np.asarray(Sn[r, c]).ravel().astype(np.float32)
            sa = np.asarray(Sa[r, c]).ravel().astype(np.float32)
            parts.append(pd.DataFrame({
                "i": r + s, "j": c, "sn": sn, "sa": sa,
                "sj": (sn + sa) / (mass_a[r] + mass_b[c] - sn - sa + 1e-6),
            }))
        if (s // row_chunk) % 50 == 0:
            log(f"    S1 rows {e:,}/{n_a:,}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["i", "j", "sn", "sa", "sj"])
