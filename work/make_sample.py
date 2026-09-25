import numpy as np, pandas as pd
from pathlib import Path
T = Path(r"D:\Dataset_ML_C\student_resource\dataset\train")
W = Path(r"D:\Dataset_ML_C\work")
s1 = pd.read_csv(T/"train_source1.tsv", sep="\t", dtype=str)
s2 = pd.read_csv(T/"train_source2.tsv", sep="\t", dtype=str)
s3 = pd.read_csv(T/"train_source3.tsv", sep="\t", dtype=str)
gt = pd.read_csv(T/"train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)

# long-form ground truth: one row per (s1, matched id)
gl = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
gl = gl[gl.m.notna() & (gl.m != "")].rename(columns={"m": "match_id"})[["source1_entity_id", "match_id"]]
print("pairs:", len(gl), "unique matched ids:", gl.match_id.nunique())

# country consistency of matched pairs
c = pd.concat([s1.set_index("entity_id").country, s2.set_index("entity_id").country, s3.set_index("entity_id").country])
gl["c1"] = gl.source1_entity_id.map(c); gl["c2"] = gl.match_id.map(c)
print("cross-country matched pairs:", (gl.c1 != gl.c2).sum())

FRAC, rng = 0.10, np.random.default_rng(42)
s1_samp = s1[rng.random(len(s1)) < FRAC].reset_index(drop=True)
ids = set(s1_samp.entity_id)
gl_s = gl[gl.source1_entity_id.isin(ids)]
matched_all = set(gl.match_id); matched_s = set(gl_s.match_id)
def pool(df):
    keep_m = df.entity_id.isin(matched_s)
    un = df[~df.entity_id.isin(matched_all)]
    keep_d = un.sample(frac=FRAC, random_state=42)
    return pd.concat([df[keep_m], keep_d]).sample(frac=1, random_state=1).reset_index(drop=True)
s2_samp, s3_samp = pool(s2), pool(s3)
gt_s = gt[gt.source1_entity_id.isin(ids)].reset_index(drop=True)
print(len(s1_samp), len(s2_samp), len(s3_samp), len(gt_s))
for n, d in [("s1", s1_samp), ("s2", s2_samp), ("s3", s3_samp), ("gt", gt_s)]:
    d.to_pickle(W/f"sample_{n}.pkl")
gl_s.to_pickle(W/"sample_pairs.pkl")
