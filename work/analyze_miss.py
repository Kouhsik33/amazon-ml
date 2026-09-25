import sys, numpy as np, pandas as pd
sys.path.insert(0, r"D:\Dataset_ML_C\work")
from er_norm import norm_name, norm_addr
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",80)
W=r"D:\Dataset_ML_C\work"
S={n:pd.read_pickle(f"{W}/sample_{n}.pkl") for n in ("s1","s2","s3")}
pairs=pd.read_pickle(f"{W}/sample_pairs.pkl"); C=pd.read_pickle(f"{W}/cands.pkl")
found=set(zip(C.s1_id[C.is_match],C.cand_id[C.is_match]))
pairs["found"]=[(a,b) in found for a,b in zip(pairs.source1_entity_id,pairs.match_id)]
info=pd.concat([S["s2"],S["s3"]]).set_index("entity_id")
pairs["src"]=pairs.match_id.str[:2]
pairs["country"]=pairs.match_id.map(info.country)
pairs["name"]=pairs.match_id.map(info.business_name); pairs["addr"]=pairs.match_id.map(info.business_address)
pairs["nonascii"]=pairs.name.fillna("").map(lambda s: not s.isascii())
pairs["addr_missing"]=pairs.addr.isna()
print(pairs.groupby(["country","src"]).found.mean().unstack())
print(pairs.groupby(["nonascii"]).found.mean())
print(pairs.groupby(["addr_missing"]).found.mean())
print(pairs.groupby(["nonascii","addr_missing"]).found.agg(["mean","size"]))
s1=S["s1"].set_index("entity_id")
m=pairs[~pairs.found].sample(25,random_state=3)
for _,r in m.iterrows():
    print(f"S1: {s1.loc[r.source1_entity_id,'business_name']!r} | {s1.loc[r.source1_entity_id,'business_address']!r}\n   {r.match_id}: {r['name']!r} | {r.addr!r}")
