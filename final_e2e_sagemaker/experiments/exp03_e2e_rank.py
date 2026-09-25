#!/usr/bin/env python3
"""EXP-03: end-to-end macro F0.5 for K=15/30 x addr_max_df 225,315/400,000.

Question EXP-01/02 could not answer: the ceiling rose, but does ACTUAL F0.5?

Pipeline: retrieval -> pair features -> LightGBM -> per-entity threshold -> F0.5.

Split discipline (the point of the experiment):
  FIT    train-split US S1, hash bucket 0-7   -> model fitting only
  TUNE   train-split US S1, hash bucket 8-9   -> threshold selection only
  EVAL   val-split US S1, the same 1,000-entity sample as EXP-01 (seed 20260926)
         -> touched once, never used to fit or tune.
FIT/TUNE/EVAL are disjoint entity sets, so no pair of the same S1 spans splits.

Scope is US / S2 only, so "true matches" means true S2 matches and an entity
with no true S2 match is a singleton in this restricted task. Identical for
every configuration, so the comparison is fair; absolute values are not
comparable to a full-task score.

Hard negatives: every negative here survived IDF retrieval AND the cosine
rerank, so the training pool is hard by construction. No extra mining.
"""
from __future__ import annotations

import gc, json, sys, time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "final_e2e_sagemaker" / "src"))
sys.path.insert(0, str(ROOT / "src"))

from new_model.retrieval.config import BASELINE                                   # noqa: E402
from new_model.retrieval.engine import CountryIndex, _H, addr_keys, corpus_df, name_keys  # noqa: E402
from new_model.evaluate.retrieval_metrics import fbeta, summarise                 # noqa: E402
from data.splits import load_split                                                # noqa: E402
from data.io import load_ground_truth                                             # noqa: E402
from features.pair_features import attach_sides, compute_features                 # noqa: E402
from features.prepare import load_norm                                            # noqa: E402

DC = ROOT / "data_cache"
SEED, CTRY, N_EVAL, N_TRAIN = 20260926, "us", 1000, 6000
t0 = time.time(); L = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
BUCKET = lambda ids: (pl.Series(ids).hash(seed=99) % 10).to_numpy()


def retrieve(idx, bid, S1, ents, dfc_n, dfc_a, mx_n, mx_a, cfg, k):
    out = {}
    for e in ents:
        nm, ad = S1[e]
        q = np.concatenate([_H(name_keys(nm, dfc_n, mx_n, cfg)),
                            _H(addr_keys(ad, dfc_a, mx_a, cfg))])
        rows, sc = idx.search_ranked(q)
        out[e] = [(bid[r], float(s), i) for i, (r, s) in enumerate(zip(rows[:k], sc[:k]))]
    return out


def featurize(cand: dict, s1n, pooln):
    rec = [(e, c, s, r) for e, lst in cand.items() for c, s, r in lst]
    df = pl.DataFrame({"source1_entity_id": [x[0] for x in rec],
                       "cand_id": [x[1] for x in rec],
                       "rr": [x[2] for x in rec], "rank": [x[3] for x in rec]})
    j = attach_sides(df.select("source1_entity_id", "cand_id"), s1n, pooln)
    j = j.join(df, on=["source1_entity_id", "cand_id"], how="left")
    X, names = compute_features(j)
    extra = np.column_stack([j["rr"].to_numpy(), j["rank"].to_numpy()]).astype(np.float32)
    return np.hstack([X, extra]), names + ["rr_score", "rr_rank"], j.select("source1_entity_id", "cand_id")


def macro_f05(pred: dict, truth: dict, ents: list) -> tuple:
    tot = tp = fp = fn = 0; s = 0.0
    for e in ents:
        t = truth.get(e, set()); p = pred.get(e, set())
        hit = len(t & p); tp += hit; fp += len(p) - hit; fn += len(t) - hit; tot += len(p)
        if not t:   s += 1.0 if not p else 0.0
        elif p:     s += fbeta(hit / len(p), hit / len(t))
    return s / len(ents), tot, fp, fn, tp


def run_cfg(tag, max_df_abs, k, idx_cache, S1, dfc_n, dfc_a, mx_n, cfg_base,
            fit, tune, ev, truth, s1n, pooln, bid, results):
    cfg = replace(cfg_base, top_k=k) if max_df_abs is None else \
          replace(cfg_base, top_k=k, addr_max_df_abs=max_df_abs)
    n_all = idx_cache["n_all"]; mx_a = cfg.addr_max_df(n_all)
    key = f"{mx_a:.0f}"
    if key not in idx_cache:
        L(f"  [{tag}] collecting query keys ...")
        allq = fit + tune + ev
        need = set()
        for e in allq:
            nm, ad = S1[e]
            need.update(_H(name_keys(nm, dfc_n, mx_n, cfg)).tolist())
            need.update(_H(addr_keys(ad, dfc_a, mx_a, cfg)).tolist())
        L(f"  [{tag}] {len(need):,} query keys; building index ...")
        idx_cache[key] = CountryIndex(idx_cache["nm"], idx_cache["ad"], dfc_n, dfc_a,
                                      mx_n, mx_a, cfg, need, log=L)
    idx = idx_cache[key]
    cand = {sp: retrieve(idx, bid, S1, e_, dfc_n, dfc_a, mx_n, mx_a, cfg, k)
            for sp, e_ in (("fit", fit), ("tune", tune), ("eval", ev))}
    L(f"  [{tag}] retrieved")

    Xf, names, mf = featurize(cand["fit"], s1n, pooln)
    yf = np.array([1 if c in truth.get(e, ()) else 0
                   for e, c in zip(mf["source1_entity_id"], mf["cand_id"])], np.int8)
    m = lgb.LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.06,
                           num_leaves=63, min_child_samples=50, subsample=0.85,
                           subsample_freq=1, colsample_bytree=0.85, reg_lambda=1.0,
                           n_jobs=6, verbose=-1, random_state=7).fit(Xf, yf, feature_name=names)
    L(f"  [{tag}] trained on {len(yf):,} pairs ({int(yf.sum()):,} pos)")
    del Xf, yf; gc.collect()

    def score(split):
        X, _, mm = featurize(cand[split], s1n, pooln)
        p = m.predict_proba(X)[:, 1]
        d = defaultdict(list)
        for e, c, pi in zip(mm["source1_entity_id"], mm["cand_id"], p):
            d[e].append((c, pi))
        del X; gc.collect()
        return d
    sc_t, sc_e = score("tune"), score("eval")

    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.20, 0.96, 0.02):
        pr = {e: {c for c, pi in v if pi >= t} for e, v in sc_t.items()}
        f = macro_f05(pr, truth, tune)[0]
        if f > best_f: best_f, best_t = f, float(t)
    L(f"  [{tag}] threshold {best_t:.2f} (TUNE F0.5={best_f:.6f})")

    pr = {e: {c for c, pi in v if pi >= best_t} for e, v in sc_e.items()}
    f, npred, fp, fn, tp = macro_f05(pr, truth, ev)
    rmet = summarise({e: {c for c, _, _ in v} for e, v in cand["eval"].items()}, truth, ev)
    results.append({"config": tag, "top_k": k, "addr_max_df": float(mx_a),
                    "threshold": best_t, "tune_f05": best_f,
                    "candidate_recall": rmet["candidate_recall"],
                    "candidate_precision": rmet["candidate_precision"],
                    "macro_f05": f, "pair_precision": tp / npred if npred else 0.0,
                    "pair_recall": tp / (tp + fn) if tp + fn else 0.0,
                    "n_predicted": npred, "false_positives": fp, "missed_true": fn})
    print(f"  >> {tag}: macroF0.5={f:.6f} pairP={tp/npred if npred else 0:.4f} "
          f"pairR={tp/(tp+fn) if tp+fn else 0:.4f} pred={npred} FP={fp} FN={fn}")
    del cand, sc_t, sc_e; gc.collect()


def main() -> int:
    cfg_base = BASELINE
    srcs = [str(DC / f"norm_train_source{i}.parquet") for i in (1, 2, 3)]
    dfc_a, n_all = corpus_df(srcs, "ad_full", CTRY)
    dfc_n, _ = corpus_df(srcs, "nm_full", CTRY)
    mx_n = cfg_base.name_max_df(n_all)
    L(f"ctx n_all={n_all:,}")

    s1n = load_norm("train", "source1")
    pooln = load_norm("train", "source2")
    us = s1n.filter(pl.col("ctry") == CTRY)
    S1 = {r[0]: (r[1], r[2]) for r in us.select("entity_id", "nm_full", "ad_full").iter_rows()}

    val_ids = set(load_split("val")); tr_ids = set(load_split("train"))
    ev_frame = sorted(e for e in S1 if e in val_ids)
    ev = [ev_frame[i] for i in np.random.default_rng(SEED).choice(len(ev_frame), size=N_EVAL, replace=False)]
    L(f"EVAL sample={len(ev)} first3={ev[:3]}")
    tr_frame = sorted(e for e in S1 if e in tr_ids)
    rng = np.random.default_rng(SEED + 1)
    pick = [tr_frame[i] for i in rng.choice(len(tr_frame), size=N_TRAIN, replace=False)]
    b = BUCKET(pick)
    fit = [e for e, x in zip(pick, b) if x < 8]; tune = [e for e, x in zip(pick, b) if x >= 8]
    assert not (set(fit) | set(tune)) & set(ev), "LEAK: train/eval overlap"
    L(f"FIT={len(fit):,} TUNE={len(tune):,} EVAL={len(ev):,} (disjoint, leak-check passed)")

    L("loading ground truth ...")
    gt = load_ground_truth().filter(pl.col("source1_entity_id").is_in(fit + tune + ev))
    L("ground truth loaded")
    truth = defaultdict(set)
    for e, mm in gt.iter_rows():
        if mm:
            for x in mm.split(","):
                if x.startswith("S2-"): truth[e].add(x)

    L("loading B partition ...")
    B = (pl.scan_parquet(DC / "norm_train_source2.parquet")
           .filter(pl.col("ctry") == CTRY).select("entity_id", "nm_full", "ad_full")).collect()
    bid = B["entity_id"].to_list()
    cache = {"nm": B["nm_full"].to_list(), "ad": B["ad_full"].to_list(), "n_all": n_all}
    L(f"B partition ready: {len(bid):,} records")
    del B; gc.collect()

    res = []
    run_cfg("A_k15_df225315", None, 15, cache, S1, dfc_n, dfc_a, mx_n, cfg_base,
            fit, tune, ev, truth, s1n, pooln, bid, res)
    run_cfg("B_k30_df225315", None, 30, cache, S1, dfc_n, dfc_a, mx_n, cfg_base,
            fit, tune, ev, truth, s1n, pooln, bid, res)
    run_cfg("C_k30_df400000", 400000.0, 30, cache, S1, dfc_n, dfc_a, mx_n, cfg_base,
            fit, tune, ev, truth, s1n, pooln, bid, res)

    out = ROOT / "final_e2e_sagemaker" / "experiments" / "exp03_e2e_rank.json"
    out.write_text(json.dumps({"scope": {"country": CTRY, "source": "S2",
        "n_eval": N_EVAL, "n_fit": len(fit), "n_tune": len(tune), "seed": SEED},
        "results": res}, indent=2))
    print("\n=== SUMMARY ===")
    for r in res:
        print(f"  {r['config']:<18s} candR={r['candidate_recall']:.4%} candP={r['candidate_precision']:.4%} "
              f"F0.5={r['macro_f05']:.6f} P={r['pair_precision']:.4f} R={r['pair_recall']:.4f} "
              f"pred={r['n_predicted']} FP={r['false_positives']} FN={r['missed_true']}")
    L(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
