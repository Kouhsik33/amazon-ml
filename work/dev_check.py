import sys, time
sys.path.insert(0, r"D:\Dataset_ML_C\submission\code\business_entity_resolution\src")
import pandas as pd, numpy as np
import s02_block
from common import WORK_DIR, id_num, pack_pair
frames = {n: pd.read_pickle(rf"D:\Dataset_ML_C\work\sample_{n}.pkl") for n in ("s1","s2","s3")}
s02_block.run("dev", frames=frames)
pairs = pd.read_pickle(r"D:\Dataset_ML_C\work\sample_pairs.pkl")
src = pairs.match_id.str[1].astype(int)
tk = set(pack_pair(id_num(pairs.source1_entity_id), id_num(pairs.match_id), src.values))
tot = hit = 0
for c in ("India","US"):
    C = pd.read_parquet(WORK_DIR/"dev"/c/"cand.parquet")
    k = pack_pair(C.s1_num.values, C.b_num.values, C.src.values)
    hit += np.isin(k, np.fromiter(tk, dtype=np.int64)).sum(); tot += len(C)
print("cands", tot, "hits", hit, "recall", hit/len(pairs))
