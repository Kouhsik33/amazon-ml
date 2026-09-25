import sys
sys.path.insert(0, r"D:\Dataset_ML_C\submission\code\business_entity_resolution\src")
import numpy as np, pandas as pd
from common import WORK_DIR, pack_pair, MIN_CAP
from blocking import *
import s03_train as t
truth, n_true = t.load_truth()
R = pd.read_parquet(WORK_DIR/"train"/"India"/"records.parquet")
s1 = R[R.src==1].reset_index(drop=True); b = R[R.src==2].reset_index(drop=True)
ctx_n, ctx_a = make_ctx(R.name_n.values, R.addr_n.values)
sub = s1.iloc[:40000]
a_keys = gen_keys(sub.name_n.values, name_keys, ctx_n)[:2] + gen_keys(sub.addr_n.values, addr_keys, ctx_a)[:2]
bk = gen_keys(b.name_n.values, name_keys, ctx_n)[:2] + gen_keys(b.addr_n.values, addr_keys, ctx_a)[:2]
tk = truth[(truth & 1) == 0]; tk = tk[np.isin(tk >> 33, sub.num.values)]
for pool, K, rr in ((300,15,"cos"),(300,20,"cos"),(600,15,"cos"),(600,20,"cos")):
    r = block_source(a_keys, len(sub), bk, len(b), K, 0.0002, MIN_CAP, pool=pool, rerank=rr, log=lambda m: None)
    key = pack_pair(sub.num.values[r.i.values], b.num.values[r.j.values], np.full(len(r), 2))
    print(f"{rr} pool={pool} K={K}: recall={np.isin(key, truth).sum()/len(tk):.4f}  cands/S1={len(r)/len(sub):.1f}", flush=True)
