import itertools
import math
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp


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


def keyify(strings, fn, stop):
    rows, keys = [], []
    for i, s in enumerate(strings):
        ks = fn(s, stop)
        keys.extend(ks)
        rows.extend([i] * len(ks))
    return np.asarray(rows, dtype=np.int32), _hash(keys) if keys else np.zeros(0, dtype=np.uint64)


def build_matrices(sides, max_df):
    """sides: dict name -> (rows_n, keys_n, rows_a, keys_a, nrows). Returns csr matrices sharing one idf-weighted vocab."""
    all_keys = np.concatenate([k for d in sides.values() for k in (d[1], d[3])])
    uniq, inv_all, cnt = np.unique(all_keys, return_inverse=True, return_counts=True)
    n_rec = sum(d[4] for d in sides.values())
    idf = np.log(n_rec / cnt).astype(np.float32)
    keep = cnt <= max_df
    out, pos = {}, 0
    for name, (rn, kn, ra, ka, nrows) in sides.items():
        mats = []
        for r, k in ((rn, kn), (ra, ka)):
            col = inv_all[pos:pos + len(k)]
            pos += len(k)
            m = keep[col]
            mats.append(sp.csr_matrix((idf[col[m]], (r[m], col[m])), shape=(nrows, len(uniq))))
        out[name] = tuple(mats)
    return out


def topk_candidates(A, B, k, chunk=4000):
    An, Aa = A
    Bn, Ba = B
    BnT, BaT = Bn.T.tocsc(), Ba.T.tocsc()
    res = []
    for s in range(0, An.shape[0], chunk):
        Sn = (An[s:s + chunk] @ BnT).tocsr()
        Sa = (Aa[s:s + chunk] @ BaT).tocsr()
        P = (Sn + Sa).tocsr()
        indptr, ind, dat = P.indptr, P.indices, P.data
        rr, cc = [], []
        for i in range(P.shape[0]):
            a, b = indptr[i], indptr[i + 1]
            if a == b:
                continue
            d = dat[a:b]
            if len(d) > k:
                sel = np.argpartition(-d, k)[:k]
            else:
                sel = np.arange(len(d))
            cc.append(ind[a:b][sel])
            rr.append(np.full(len(sel), i + s, dtype=np.int32))
        if not rr:
            continue
        r, c = np.concatenate(rr), np.concatenate(cc)
        ln = r - s
        res.append(pd.DataFrame({
            "i": r, "j": c,
            "sn": np.asarray(Sn[ln, c]).ravel(), "sa": np.asarray(Sa[ln, c]).ravel(),
        }))
    return pd.concat(res, ignore_index=True) if res else pd.DataFrame(columns=["i", "j", "sn", "sa"])
