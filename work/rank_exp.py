import sys, json, time
sys.path.insert(0, r"D:\Dataset_ML_C\submission\code\business_entity_resolution\src")
import numpy as np, pandas as pd
from common import WORK_DIR, pack_pair, CAP_FRAC, MIN_CAP
from blocking import *
import s03_train as t
truth, n_true = t.load_truth()
c = "India"
R = pd.read_parquet(WORK_DIR/"train"/c/"records.parquet")
s1 = R[R.src==1].reset_index(drop=True); b = R[R.src==2].reset_index(drop=True)
ctx_n, ctx_a = make_ctx(R.name_n.values, R.addr_n.values)
sub = s1.iloc[:40000]
a_keys = gen_keys(sub.name_n.values, name_keys, ctx_n)[:2] + gen_keys(sub.addr_n.values, addr_keys, ctx_a)[:2]
bk = gen_keys(b.name_n.values, name_keys, ctx_n)[:2] + gen_keys(b.addr_n.values, addr_keys, ctx_a)[:2]
for cap_frac in (0.0002, 0.001):
    r = block_source(a_keys, len(sub), bk, len(b), 200, cap_frac, MIN_CAP, log=lambda m: None)
    r["tot"] = r.sn + r.sa
    r["rank"] = r.groupby("i").tot.rank(ascending=False, method="first")
    key = pack_pair(sub.num.values[r.i.values], b.num.values[r.j.values], np.full(len(r), 2))
    r["hit"] = np.isin(key, truth)
    s1n = set(sub.num.values)
    # true pairs of these S1 in S2
    tk = truth[(truth & 1) == 0]
    tk = tk[np.isin(tk >> 33, sub.num.values)]
    print(f"cap_frac={cap_frac}: true S2 pairs for subset={len(tk)}; found within top-200: {r.hit.sum()} ({r.hit.sum()/len(tk):.4f})")
    for K in (15, 25, 40, 60, 100, 200):
        print(f"   recall@{K}: {r[(r['rank']<=K)].hit.sum()/len(tk):.4f}")
