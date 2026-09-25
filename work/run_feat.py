import sys, time, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r"D:\Dataset_ML_C\work")
from er_norm import norm_name, norm_addr, learn_lexicon, apply_lexicon
from er_feat import build_features
W = Path(r"D:\Dataset_ML_C\work"); t = time.time()
S = {n: pd.read_pickle(W / f"sample_{n}.pkl") for n in ("s1", "s2", "s3")}
lp = pd.read_pickle(W / "lex_pairs.pkl"); lex = learn_lexicon(lp.n1.values, lp.nb.values, min_j=0.0)
for d in S.values():
    d["name_n"] = [norm_name(x) for x in d.business_name.values]
    d["addr_n"] = [norm_addr(x) for x in d.business_address.values]
    d["name_n"] = [apply_lexicon(x, lex) if not x.isascii() else x for x in d.name_n.values]
    d["business_address"] = d.business_address
C = pd.read_parquet(W / "dev_candidates.parquet")
b = pd.concat([S["s2"], S["s3"]], ignore_index=True)
if len(sys.argv) > 1: C = C.iloc[: int(sys.argv[1])]
print("cands", len(C), time.time() - t, flush=True)
F = build_features(C, S["s1"], b)
print("features", F.shape, time.time() - t)
F["is_match"] = C.is_match.values; F["s1_id"] = C.s1_id.values; F["cand_id"] = C.cand_id.values
F.to_parquet(W / ("dev_features.parquet" if len(sys.argv) == 1 else "dev_features_small.parquet"))
print(F.describe().T[["mean","min","max"]].to_string())
