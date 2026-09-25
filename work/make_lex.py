import sys, pandas as pd
sys.path.insert(0, r"D:\Dataset_ML_C\work")
from er_norm import norm_name
from pathlib import Path
T = Path(r"D:\Dataset_ML_C\student_resource\dataset\train")
W = Path(r"D:\Dataset_ML_C\work")
dev = set(pd.read_pickle(W/"sample_s1.pkl").entity_id)
s1 = pd.read_csv(T/"train_source1.tsv", sep="\t", dtype=str, usecols=["entity_id","business_name"]).set_index("entity_id").business_name
gt = pd.read_csv(T/"train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[~gt.source1_entity_id.isin(dev) & (gt.matched_entity_ids != "")]
gl = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
gl = gl.rename(columns={"m": "match_id"})[["source1_entity_id", "match_id"]]
b = pd.concat([pd.read_csv(T/f"train_source{i}.tsv", sep="\t", dtype=str, usecols=["entity_id","business_name"]) for i in (2,3)]).set_index("entity_id").business_name
gl["nb"] = gl.match_id.map(b)
gl = gl[gl.nb.fillna("").map(lambda s: not s.isascii())]
gl["n1"] = gl.source1_entity_id.map(s1)
gl["n1"] = [norm_name(x) for x in gl.n1]; gl["nb"] = [norm_name(x) for x in gl.nb]
print(len(gl)); gl[["n1","nb"]].to_pickle(W/"lex_pairs.pkl")
