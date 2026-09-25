#!/usr/bin/env python3
"""EXP-01: TOP_K sweep at the frozen baseline DF gate.

ONE variable: top_k. Everything else is RetrievalConfig defaults (= Approach2
baseline: share 0.03/0.05, pool 600, cap_frac 0.0002, rerank cos).

Question: is TOP_K=15 the binding constraint on retrieval, and what is the
macro-F0.5 ceiling the downstream model is working under?

Scope: US / S2 / 1,000-S1 deterministic sample (seed 20260926) -- the same
population used for the already-measured baseline-vs-400K comparison.
"""
from __future__ import annotations

import gc, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "final_e2e_sagemaker" / "src"))
sys.path.insert(0, str(ROOT / "src"))

from dataclasses import replace                                      # noqa: E402
from new_model.retrieval.config import BASELINE                      # noqa: E402
from new_model.retrieval.engine import CountryIndex, UNIQ, _H, addr_keys, corpus_df, name_keys  # noqa: E402
from new_model.evaluate.retrieval_metrics import summarise           # noqa: E402
from data.splits import load_split                                   # noqa: E402
from data.io import load_ground_truth                                # noqa: E402

DC = ROOT / "data_cache"
SEED, CTRY, N = 20260926, "us", 1000
KS = [5, 10, 15, 30, 50, 100, 600]
t0 = time.time(); L = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr-max-df", type=float, default=None,
                    help="absolute address DF gate; omit to use the 0.03 share")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()   # NOT `a`: `a` is the S1 loop variable below
    cfg = BASELINE if args.addr_max_df is None else replace(
        BASELINE, name=f"addr_max_df_{int(args.addr_max_df)}", addr_max_df_abs=args.addr_max_df)
    srcs = [str(DC / f"norm_train_source{i}.parquet") for i in (1, 2, 3)]
    dfc_a, n_all = corpus_df(srcs, "ad_full", CTRY)
    dfc_n, _ = corpus_df(srcs, "nm_full", CTRY)
    mx_a, mx_n = cfg.addr_max_df(n_all), cfg.name_max_df(n_all)
    L(f"ctx n_all={n_all:,} addr_max_df={mx_a:,.0f} name_max_df={mx_n:,.0f}")

    s1 = (pl.read_parquet(DC / "norm_train_source1.parquet",
                          columns=["entity_id", "nm_full", "ad_full", "ctry"])
            .filter(pl.col("entity_id").is_in(load_split("val")) & (pl.col("ctry") == CTRY)))
    S1 = {r[0]: (r[1], r[2]) for r in s1.iter_rows()}
    frame = sorted(S1)
    sample = [frame[i] for i in np.random.default_rng(SEED).choice(len(frame), size=N, replace=False)]
    L(f"sample={len(sample)} first3={sample[:3]}")

    qk, need = {}, set()
    for a in sample:
        nm, ad = S1[a]
        h = np.concatenate([_H(name_keys(nm, dfc_n, mx_n, cfg)),
                            _H(addr_keys(ad, dfc_a, mx_a, cfg))])
        qk[a] = h; need.update(h.tolist())

    B = (pl.scan_parquet(DC / "norm_train_source2.parquet")
           .filter(pl.col("ctry") == CTRY).select("entity_id", "nm_full", "ad_full")).collect()
    bid = B["entity_id"].to_list()
    idx = CountryIndex(B["nm_full"].to_list(), B["ad_full"].to_list(),
                       dfc_n, dfc_a, mx_n, mx_a, cfg, need, log=L)
    del B; gc.collect()

    ranked = {a: idx.search_ranked(qk[a])[0] for a in sample}
    L("scored 1,000 S1")

    gt = load_ground_truth().filter(pl.col("source1_entity_id").is_in(sample))
    truth = defaultdict(set)
    for a, m in gt.iter_rows():
        if m:
            for x in m.split(","):
                if x.startswith("S2-"):
                    truth[a].add(x)

    rows = []
    for k in KS:
        got = {a: {bid[r] for r in ranked[a][:k]} for a in sample}
        m = summarise(got, truth, sample); m["top_k"] = k
        rows.append(m)
        print(f"  top_k={k:<4d} recall={m['candidate_recall']:.4%} "
              f"prec={m['candidate_precision']:.4%} cand/S1={m['cand_per_entity']:6.1f} "
              f"ceiling={m['macro_f05_ceiling']:.6f}")
    out = ROOT / "final_e2e_sagemaker" / "experiments" / f"exp01_topk_sweep{args.tag}.json"
    out.write_text(json.dumps({"config": cfg.as_dict(), "scope":
        {"country": CTRY, "source": "S2", "n_s1": N, "seed": SEED}, "results": rows}, indent=2))
    L(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
