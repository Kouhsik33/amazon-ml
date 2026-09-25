"""Second-stage features built from first-stage probabilities (all label-free at prediction time).

For every candidate pair (s1, b) with first-stage probability p:
  * S1-level context : how much match evidence the entity has elsewhere (sum / max / counts of other high-p candidates)
  * B-level context  : how strongly other S1 entities claim the same B record
  * anchor agreement : similarity between B and the entity's strongest *other* candidate ("cluster coherence"):
                       true matches of one business tend to resemble each other even when the S1 name is noisy
"""
import numpy as np
import pandas as pd

ANCHOR_COLS = ["n_ratio", "n_tsort", "n_tset", "n_jw", "n_cov_a", "n_cov_b", "a_missing", "a_ratio", "a_tsort", "a_tset",
               "a_jac", "a_cov_a", "a_cov_b", "a_num_jac", "a_first_num_eq", "a_postal_eq", "a_last_eq"]
CTX_COLS = ["p1", "logit_p1", "s1_p_sum_other", "s1_p_max_other", "s1_n50_other", "s1_n90_other", "s1_n50_s2", "s1_n50_s3",
            "s1_rank_p", "p_over_s1_max", "b_rank_p", "b_p_max_other", "b_n_s1", "p_anchor", "anchor_same_src"]
STAGE2_EXTRA = CTX_COLS + ["anc_" + c for c in ANCHOR_COLS]


def context_features(df):
    """df: s1_num, b_num, src, p  (all candidate rows of one country). Returns (DataFrame of CTX_COLS, anchor row index array or -1)."""
    p = df.p.to_numpy(np.float32)
    s1 = df.s1_num.to_numpy()
    is3 = (df.src.to_numpy() == 3)
    bk = df.b_num.to_numpy().astype(np.int64) * 2 + is3
    g = pd.DataFrame({"s1": s1, "bk": bk, "p": p, "h50": p >= 0.5, "h90": p >= 0.9, "h50_s2": (p >= 0.5) & ~is3, "h50_s3": (p >= 0.5) & is3})
    gs = g.groupby("s1", sort=False)
    out = pd.DataFrame(index=df.index)
    out["p1"] = p
    out["logit_p1"] = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    s_sum = gs.p.transform("sum").to_numpy()
    out["s1_p_sum_other"] = s_sum - p
    out["s1_n50_other"] = gs.h50.transform("sum").to_numpy() - g.h50.to_numpy()
    out["s1_n90_other"] = gs.h90.transform("sum").to_numpy() - g.h90.to_numpy()
    out["s1_n50_s2"] = gs.h50_s2.transform("sum").to_numpy()
    out["s1_n50_s3"] = gs.h50_s3.transform("sum").to_numpy()
    rank = gs.p.rank(ascending=False, method="first").to_numpy()
    out["s1_rank_p"] = rank
    p_max = gs.p.transform("max").to_numpy()
    out["p_over_s1_max"] = p / np.maximum(p_max, 1e-6)
    # strongest and second strongest row per S1 -> "best other" for each row
    order = np.lexsort((-p, s1))
    s1_sorted = s1[order]
    first = np.r_[True, s1_sorted[1:] != s1_sorted[:-1]]
    start = np.flatnonzero(first)
    grp_id = np.cumsum(first) - 1
    top1 = order[start]
    top2_pos = start + 1
    has2 = top2_pos < len(order)
    has2[has2] = s1_sorted[top2_pos[has2]] == s1_sorted[start[has2]]
    top2 = np.full(len(start), -1, np.int64)
    top2[has2] = order[top2_pos[has2]]
    gid = np.empty(len(df), np.int64)
    gid[order] = grp_id
    anchor = np.where(rank == 1, top2[gid], top1[gid])
    has_anchor = anchor >= 0
    out["s1_p_max_other"] = np.where(has_anchor, p[np.maximum(anchor, 0)], 0.0)
    out["p_anchor"] = out["s1_p_max_other"].to_numpy()
    out["anchor_same_src"] = np.where(has_anchor, (is3[np.maximum(anchor, 0)] == is3).astype(np.float32), np.nan)
    gb = g.groupby("bk", sort=False)
    out["b_rank_p"] = gb.p.rank(ascending=False, method="first").to_numpy()
    out["b_n_s1"] = gb.p.transform("size").to_numpy()
    order_b = np.lexsort((-p, bk))
    bk_sorted = bk[order_b]
    first_b = np.r_[True, bk_sorted[1:] != bk_sorted[:-1]]
    start_b = np.flatnonzero(first_b)
    gid_b = np.empty(len(df), np.int64)
    gid_b[order_b] = np.cumsum(first_b) - 1
    top1b = order_b[start_b]
    top2b_pos = start_b + 1
    has2b = top2b_pos < len(order_b)
    has2b[has2b] = bk_sorted[top2b_pos[has2b]] == bk_sorted[start_b[has2b]]
    top2b = np.full(len(start_b), -1, np.int64)
    top2b[has2b] = order_b[top2b_pos[has2b]]
    best_b_other = np.where(out["b_rank_p"].to_numpy() == 1, top2b[gid_b], top1b[gid_b])
    out["b_p_max_other"] = np.where(best_b_other >= 0, p[np.maximum(best_b_other, 0)], 0.0)
    return out[CTX_COLS].astype(np.float32), anchor
