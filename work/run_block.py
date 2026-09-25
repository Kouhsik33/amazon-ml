import sys, time
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r"D:\Dataset_ML_C\work")
from er_norm import norm_name, norm_addr, learn_lexicon, apply_lexicon
from er_block import token_df, keyify, name_keys, addr_keys, build_matrices, topk_candidates

W = Path(r"D:\Dataset_ML_C\work")
MAX_DF = int(sys.argv[1]) if len(sys.argv) > 1 else 600
K = int(sys.argv[2]) if len(sys.argv) > 2 else 30

t = time.time()
S = {n: pd.read_pickle(W / f"sample_{n}.pkl") for n in ("s1", "s2", "s3")}
pairs = pd.read_pickle(W / "sample_pairs.pkl")
for n, d in S.items():
    d["name_n"] = [norm_name(x) for x in d.business_name.values]
    d["addr_n"] = [norm_addr(x) for x in d.business_address.values]
MIN_J = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
lp = pd.read_pickle(W / "lex_pairs.pkl")
lex = learn_lexicon(lp.n1.values, lp.nb.values, min_j=MIN_J)
print("lexicon size", len(lex))
for n, d in S.items():
    d["name_n"] = [apply_lexicon(x, lex) if not x.isascii() else x for x in d.name_n.values]
print("normalized", time.time() - t)
pair_set = set(zip(pairs.source1_entity_id, pairs.match_id))

cands = []
for country in sorted(S["s1"].country.unique()):
    sub = {n: d[d.country == country].reset_index(drop=True) for n, d in S.items()}
    allname = np.concatenate([d.name_n.values for d in sub.values()])
    alladdr = np.concatenate([d.addr_n.values for d in sub.values()])
    N = len(allname)
    stop_n = (token_df(allname), 0.05 * N)
    stop_a = (token_df(alladdr), 0.03 * N)
    sides = {}
    for n, d in sub.items():
        rn, kn = keyify(d.name_n.values, name_keys, stop_n)
        ra, ka = keyify(d.addr_n.values, addr_keys, stop_a)
        sides[n] = (rn, kn, ra, ka, len(d))
    M = build_matrices(sides, MAX_DF)
    for src in ("s2", "s3"):
        c = topk_candidates(M["s1"], M[src], K)
        c["s1_id"] = sub["s1"].entity_id.values[c.i.values]
        c["cand_id"] = sub[src].entity_id.values[c.j.values]
        c["country"] = country
        cands.append(c)
    print(country, "done", time.time() - t, flush=True)

C = pd.concat(cands, ignore_index=True)
C["is_match"] = [(a, b) in pair_set for a, b in zip(C.s1_id, C.cand_id)]
n_true = len(pairs)
print(f"MAX_DF={MAX_DF} K={K}  candidates={len(C):,}  per-S1={len(C)/len(S['s1']):.1f}")
print(f"pair recall = {C.is_match.sum()/n_true:.4f}  ({C.is_match.sum():,}/{n_true:,})")
tot = len(S["s1"]) * (len(S["s2"]) + len(S["s3"]))
print(f"reduction ratio = {1 - len(C)/tot:.6f}")
C.to_pickle(W / "cands.pkl")
