import sys
sys.path.insert(0, r"D:\Dataset_ML_C\submission\code\business_entity_resolution\src")
import numpy as np, pandas as pd
from common import WORK_DIR, pack_pair, DATA_DIR
import s03_train as t
HF = 0.15
truth, n_true = t.load_truth()
out = []
for c in ("India", "US"):
    d = WORK_DIR / "train" / c
    cand = pd.read_parquet(d / "cand.parquet", columns=["s1_num", "b_num", "src"])
    recs = pd.read_parquet(d / "records.parquet")
    s1 = recs[recs.src == 1]
    u1 = t.u01(s1.num.to_numpy())
    ids = s1.num.to_numpy()[(u1 >= HF * 0.7) & (u1 < HF)]
    tr = truth[np.isin(truth >> 33, ids)]
    ck = pack_pair(cand.s1_num.to_numpy(), cand.b_num.to_numpy(), cand.src.to_numpy())
    miss = tr[~np.isin(tr, ck)]
    m = pd.DataFrame({"s1": miss >> 33, "b": (miss >> 1) & ((1 << 32) - 1), "src": np.where(miss & 1, 3, 2)})
    out.append((c, m))
raw = {i: pd.read_csv(DATA_DIR / "train" / f"train_source{i}.tsv", sep="\t", dtype=str).assign(num=lambda d: d.entity_id.str[3:].astype(np.int64)).set_index("num") for i in (1, 2, 3)}
for c, m in out:
    for s in (2, 3):
        mm = m[m.src == s].sample(12 if c == "India" else 6, random_state=1)
        print(f"\n===== {c} S{s} blocking misses =====")
        for _, r in mm.iterrows():
            a, b = raw[1].loc[r.s1], raw[s].loc[r.b]
            print(f"S1: {a.business_name!r} | {a.business_address!r}\n    {b.business_name!r} | {b.business_address!r}")
