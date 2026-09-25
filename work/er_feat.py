import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def token_idf_by_country(rec, col):
    """country -> {token: idf} computed over all records of that country (open-set: any country label)."""
    out = {}
    for c, g in rec.groupby("country"):
        df = Counter()
        for s in g[col].values:
            df.update(set(s.split()))
        n = len(g)
        out[c] = {t: math.log(n / v) for t, v in df.items()}
    return out


def _tokinfo(strings, countries, idf_by_country, generic_idf):
    toks, tot, core = [], np.zeros(len(strings), np.float32), []
    for i, (s, c) in enumerate(zip(strings, countries)):
        t = frozenset(s.split())
        idf = idf_by_country[c]
        toks.append(t)
        tot[i] = sum(idf[x] for x in t)
        core.append(frozenset(x for x in t if idf[x] > generic_idf))
    return toks, tot, core


def _numset(toks):
    return [frozenset(x for x in t if any(ch.isdigit() for ch in x)) for t in toks]


def _postal(nums):
    return [frozenset(x for x in t if x.isdigit() and len(x) in (5, 6)) for t in nums]


def build_features(cands, s1, b, generic_idf=2.5):
    """cands: s1_id, cand_id, country, sn, sa. s1/b: entity_id, country, name_n, addr_n, business_name, business_address."""
    rec = pd.concat([s1, b], ignore_index=True)
    idx = pd.Index(rec.entity_id)
    ia = idx.get_indexer(cands.s1_id.values)
    ib = idx.get_indexer(cands.cand_id.values)
    name_idf = token_idf_by_country(rec, "name_n")
    addr_idf = token_idf_by_country(rec, "addr_n")
    countries = rec.country.values
    ntok, ntot, ncore = _tokinfo(rec.name_n.values, countries, name_idf, generic_idf)
    atok, atot, _ = _tokinfo(rec.addr_n.values, countries, addr_idf, 99)
    anum = _numset(atok)
    apost = _postal(anum)
    aseq = [s.split() for s in rec.addr_n.values]
    nseq = [s.split() for s in rec.name_n.values]

    na = rec.name_n.values[ia]
    nb = rec.name_n.values[ib]
    aa = rec.addr_n.values[ia]
    ab = rec.addr_n.values[ib]
    f = pd.DataFrame(index=cands.index)
    f["sn"] = cands.sn.values
    f["sa"] = cands.sa.values
    f["s_tot"] = f.sn + f.sa
    f["is_s3"] = cands.cand_id.str.startswith("S3-").values.astype(np.int8)

    f["n_ratio"] = _cp(na, nb, fuzz.ratio) / 100
    f["n_tsort"] = _cp(na, nb, fuzz.token_sort_ratio) / 100
    f["n_tset"] = _cp(na, nb, fuzz.token_set_ratio) / 100
    f["n_partial"] = _cp(na, nb, fuzz.partial_ratio) / 100
    f["n_jw"] = _cp(na, nb, JaroWinkler.normalized_similarity)
    f["n_lev"] = _cp(na, nb, Levenshtein.normalized_similarity)
    nb_nonlatin = ~pd.Series(rec.business_name.values[ib]).fillna("").map(str.isascii).values
    f["n_nonlatin_b"] = nb_nonlatin.astype(np.int8)
    f["n_untrans_b"] = np.array([sum(not t.isascii() for t in nseq[j]) for j in ib], dtype=np.int8)

    a_missing = pd.Series(ab).eq("").values | pd.Series(aa).eq("").values
    f["a_missing"] = a_missing.astype(np.int8)
    for nm, sc in (("a_ratio", fuzz.ratio), ("a_tsort", fuzz.token_sort_ratio), ("a_tset", fuzz.token_set_ratio)):
        v = _cp(aa, ab, sc) / 100
        v[a_missing] = np.nan
        f[nm] = v

    cols = {k: np.empty(len(cands), np.float32) for k in (
        "n_jac", "n_core_jac", "n_core_ov", "n_core_a", "n_core_b", "n_cov_a", "n_cov_b", "n_first_eq", "n_last_eq",
        "n_len_ratio", "a_jac", "a_cov_a", "a_cov_b", "a_num_ov", "a_num_jac", "a_num_a", "a_num_b",
        "a_first_num_eq", "a_postal_eq", "a_last_eq", "a_tok_a", "a_tok_b")}
    for r, (i, j) in enumerate(zip(ia, ib)):
        A, B = ntok[i], ntok[j]
        inter = A & B
        u = len(A | B) or 1
        cols["n_jac"][r] = len(inter) / u
        CA, CB = ncore[i], ncore[j]
        ci = CA & CB
        cols["n_core_jac"][r] = len(ci) / (len(CA | CB) or 1)
        cols["n_core_ov"][r] = len(ci)
        cols["n_core_a"][r] = len(CA)
        cols["n_core_b"][r] = len(CB)
        idf = name_idf[countries[i]]
        sh = sum(idf[t] for t in inter)
        cols["n_cov_a"][r] = sh / ntot[i] if ntot[i] > 0 else 0
        cols["n_cov_b"][r] = sh / ntot[j] if ntot[j] > 0 else 0
        sa_, sb_ = nseq[i], nseq[j]
        cols["n_first_eq"][r] = 1.0 if sa_ and sb_ and sa_[0] == sb_[0] else 0.0
        cols["n_last_eq"][r] = 1.0 if sa_ and sb_ and sa_[-1] == sb_[-1] else 0.0
        la, lb = len(na[r]), len(nb[r])
        cols["n_len_ratio"][r] = min(la, lb) / max(la, lb, 1)

        A, B = atok[i], atok[j]
        u = len(A | B) or 1
        inter = A & B
        cols["a_jac"][r] = len(inter) / u
        idf = addr_idf[countries[i]]
        sh = sum(idf[t] for t in inter)
        cols["a_cov_a"][r] = sh / atot[i] if atot[i] > 0 else 0
        cols["a_cov_b"][r] = sh / atot[j] if atot[j] > 0 else 0
        NA, NB = anum[i], anum[j]
        cols["a_num_ov"][r] = len(NA & NB)
        cols["a_num_jac"][r] = len(NA & NB) / (len(NA | NB) or 1)
        cols["a_num_a"][r] = len(NA)
        cols["a_num_b"][r] = len(NB)
        qa, qb = aseq[i], aseq[j]
        fa = next((t for t in qa if t in NA), None)
        cols["a_first_num_eq"][r] = 1.0 if fa is not None and fa in NB else 0.0
        cols["a_postal_eq"][r] = 1.0 if apost[i] & apost[j] else 0.0
        cols["a_last_eq"][r] = 1.0 if qa and qb and qa[-1] == qb[-1] else 0.0
        cols["a_tok_a"][r] = len(qa)
        cols["a_tok_b"][r] = len(qb)
    for k, v in cols.items():
        f[k] = v
    for k in [k for k in f.columns if k.startswith("a_") and k not in ("a_missing",)]:
        f.loc[a_missing, k] = np.nan

    tot = f.s_tot
    g = pd.DataFrame({"s1": cands.s1_id.values, "src": f.is_s3.values, "b": cands.cand_id.values, "t": tot.values})
    g1 = g.groupby(["s1", "src"]).t
    f["rank_s1"] = g1.rank(ascending=False, method="first").values
    f["ratio_best_s1"] = (g.t / g1.transform("max")).values
    f["n_cand_s1"] = g1.transform("size").values
    gb = g.groupby("b").t
    f["rank_b"] = gb.rank(ascending=False, method="first").values
    f["ratio_best_b"] = (g.t / gb.transform("max")).values
    f["n_s1_for_b"] = gb.transform("size").values
    second = pd.Series(g.t.values[(f["rank_b"].values == 2)], index=g.b.values[(f["rank_b"].values == 2)])
    f["gap_b"] = (gb.transform("max") - g.b.map(second).fillna(0)).values
    dupn = s1.groupby(["country", "name_n"]).entity_id.transform("size")
    dup_map = pd.Series(dupn.values, index=s1.entity_id.values)
    f["s1_name_dup"] = cands.s1_id.map(dup_map).values
    return f
