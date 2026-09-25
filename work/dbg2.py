import sys
sys.path.insert(0, r"D:\Dataset_ML_C\submission\code\business_entity_resolution\src")
import numpy as np, pandas as pd
from common import WORK_DIR, pack_pair, MIN_CAP
from blocking import *
import s03_train as t
truth, n_true = t.load_truth()
R = pd.read_parquet(WORK_DIR/"train"/"India"/"records.parquet")
s1 = R[R.src==1].reset_index(drop=True).iloc[:3000]; b = R[R.src==2].reset_index(drop=True)
ctx_n, ctx_a = make_ctx(R.name_n.values, R.addr_n.values)
a_keys = gen_keys(s1.name_n.values, name_keys, ctx_n)[:2] + gen_keys(s1.addr_n.values, addr_keys, ctx_a)[:2]
bk = gen_keys(b.name_n.values, name_keys, ctx_n)[:2] + gen_keys(b.addr_n.values, addr_keys, ctx_a)[:2]
r = block_source(a_keys, len(s1), bk, len(b), 15, 0.0002, MIN_CAP, pool=60, log=lambda m: None)
key = pack_pair(s1.num.values[r.i.values], b.num.values[r.j.values], np.full(len(r), 2))
r["hit"] = np.isin(key, truth)
print(r.head(20)); print(r.describe().T)
print(r.groupby("hit")[["sn","sa","sj"]].mean())
print("-----")
from blocking import build_index, _a_chunk
cap = max(MIN_CAP, int(0.0002*len(b)))
BnT, BaT, uniq, keep, idf = build_index(bk, len(b), cap)
print(BnT.shape, BnT.nnz, BaT.nnz)
mb = np.asarray(BnT.sum(axis=0)).ravel() + np.asarray(BaT.sum(axis=0)).ravel()
print("mass_b stats", mb.min(), mb.mean(), mb.max(), mb.shape)
An = _a_chunk(a_keys[0], a_keys[1], 0, 2000, uniq, keep, idf, len(uniq))
Aa = _a_chunk(a_keys[2], a_keys[3], 0, 2000, uniq, keep, idf, len(uniq))
ma = np.asarray(An.sum(axis=1)).ravel() + np.asarray(Aa.sum(axis=1)).ravel()
print("mass_a stats", ma.min(), ma.mean(), ma.max())
Sn = (An @ BnT); Sa = (Aa @ BaT)
print("sn+sa row0 max", (Sn+Sa)[0].max(), "mass_a[0]", ma[0])
