import sys
sys.path.insert(0, r"D:\Dataset_ML_C\submission\code\business_entity_resolution\src")
import lightgbm  # noqa
import numpy as np
import pandas as pd
from common import WORK_DIR, pack_pair
import s03_train as t
from features import ALL_COLS
from stage2 import STAGE2_EXTRA

HF = 0.15
truth, n_true = t.load_truth()
pd.set_option("display.width", 220)

# ---------- (a) true pairs that never reach the candidate set (blocking misses) ----------
rows, tot_true = [], 0
miss_frames = []
for c in ("India", "US"):
    d = WORK_DIR / "train" / c
    cand = pd.read_parquet(d / "cand.parquet", columns=["s1_num", "b_num", "src"])
    u = t.u01(cand.s1_num.to_numpy())
    hold_c = cand[(u >= HF * 0.7) & (u < HF)]
    recs = pd.read_parquet(d / "records.parquet")
    s1 = recs[recs.src == 1]
    u1 = t.u01(s1.num.to_numpy())
    hold_ids = s1.num.to_numpy()[(u1 >= HF * 0.7) & (u1 < HF)]
    tr = truth[np.isin(truth >> 33, hold_ids)]
    ck = pack_pair(hold_c.s1_num.to_numpy(), hold_c.b_num.to_numpy(), hold_c.src.to_numpy())
    miss = tr[~np.isin(tr, ck)]
    tot_true += len(tr)
    mdf = pd.DataFrame({"s1_num": miss >> 33, "b_num": (miss >> 1) & ((1 << 32) - 1), "src": np.where(miss & 1, 3, 2), "country": c})
    tdf = pd.DataFrame({"src": np.where(tr & 1, 3, 2)})
    for s in (2, 3):
        rows.append({"country": c, "source": f"S{s}", "true pairs": int((tdf.src == s).sum()), "not in candidates": int((mdf.src == s).sum())})
    # attach text
    s1i = s1.set_index("num")[["name_n", "addr_n"]].add_prefix("s1_")
    parts = []
    for s in (2, 3):
        b = recs[recs.src == s].set_index("num")[["name_n", "addr_n", "name_nonlatin"]].add_prefix("b_")
        parts.append(mdf[mdf.src == s].join(b, on="b_num"))
    m = pd.concat(parts).join(s1i, on="s1_num")
    miss_frames.append(m)
    print(c, "done", flush=True)
blk = pd.DataFrame(rows)
blk["miss rate"] = blk["not in candidates"] / blk["true pairs"]
print("\n(a) blocking misses on held-out entities"); print(blk.round(4).to_string())
M = pd.concat(miss_frames, ignore_index=True)


def jac(a, b):
    A, B = set(a.split()), set(b.split())
    return len(A & B) / max(len(A | B), 1)


samp = M.sample(min(len(M), 20000), random_state=0)
samp["addr_missing"] = samp.b_addr_n == ""
samp["name_jac"] = [jac(a, b) for a, b in zip(samp.s1_name_n, samp.b_name_n)]
samp["addr_jac"] = [jac(a, b) if b else np.nan for a, b in zip(samp.s1_addr_n, samp.b_addr_n)]
print("\nblocking-miss profile (sample of", len(samp), ")")
print("B address missing:", round(samp.addr_missing.mean(), 3), "| non-Latin B name:", round(samp.b_name_nonlatin.mean(), 3))
print("name Jaccard   <0.2:", round((samp.name_jac < 0.2).mean(), 3), " 0.2-0.6:", round(((samp.name_jac >= 0.2) & (samp.name_jac < 0.6)).mean(), 3), " >=0.6:", round((samp.name_jac >= 0.6).mean(), 3))
ad = samp[~samp.addr_missing]
print("addr Jaccard (address present)  <0.2:", round((ad.addr_jac < 0.2).mean(), 3), " 0.2-0.5:", round(((ad.addr_jac >= 0.2) & (ad.addr_jac < 0.5)).mean(), 3), " >=0.5:", round((ad.addr_jac >= 0.5).mean(), 3))
both_weak = ((samp.name_jac < 0.3) & ((samp.addr_jac < 0.3) | samp.addr_missing)).mean()
print("both name and address evidence weak (name J<0.3 and addr J<0.3 or missing):", round(both_weak, 3))
print("name similar (>=0.6) but not retrieved:", round((samp.name_jac >= 0.6).mean(), 3), "-> these are mostly ambiguous same-name entities")

# ---------- (b) in candidates but rejected by stage 2 ----------
X2 = np.load(WORK_DIR / "cache" / "X2.npy", mmap_mode="r")
m2 = pd.read_parquet(WORK_DIR / "cache" / "meta2_oof.parquet")
names = ALL_COLS + list(STAGE2_EXTRA)
pos = m2.y.to_numpy() & (m2.p2.to_numpy() < 0.7)
idx = np.flatnonzero(pos)
cols = {k: names.index(k) for k in ("a_missing", "n_tset", "a_tset", "n_cov_b", "a_num_jac", "n_nonlatin_b", "s1_name_dup", "b_n_s1", "b_rank_p")}
F = pd.DataFrame(np.asarray(X2[idx][:, list(cols.values())]), columns=list(cols))
F["p2"] = m2.p2.to_numpy()[idx]
F["country"] = m2.country.to_numpy()[idx]
print("\n(b) in-candidate true pairs rejected by stage 2:", len(F), "of", int(m2.y.sum()), f"({len(F)/m2.y.sum():.3%})")
print(pd.cut(F.p2, [0, 0.05, 0.2, 0.4, 0.6, 0.7], include_lowest=True).value_counts().sort_index().to_string())
print("share with B address missing:", round((F.a_missing == 1).mean(), 3), "| non-Latin B name:", round((F.n_nonlatin_b == 1).mean(), 3))
print("name token-set sim >=0.8:", round((F.n_tset >= 0.8).mean(), 3), "| address token-set sim >=0.8 (present):", round((F.a_tset[F.a_missing == 0] >= 0.8).mean(), 3))
print("strong name (>=0.8) but weak/missing address:", round(((F.n_tset >= 0.8) & ((F.a_missing == 1) | (F.a_tset < 0.6))).mean(), 3))
print("strong address (>=0.8) but weak name (<0.6):", round(((F.a_tset >= 0.8) & (F.n_tset < 0.6)).mean(), 3))
print("S1 name duplicated (dup>=2):", round((F.s1_name_dup >= 2).mean(), 3), "| B claimed by >1 S1 (b_n_s1>=2):", round((F.b_n_s1 >= 2).mean(), 3), "| not top-ranked on its B (b_rank_p>1):", round((F.b_rank_p > 1).mean(), 3))
print("by country:", F.country.value_counts().to_dict())
