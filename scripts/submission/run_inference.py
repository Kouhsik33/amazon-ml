#!/usr/bin/env python3
"""Frozen E07 inference over the test set, streamed to TSV.

Nothing here is retrained or retuned. The E06 booster and the E07 decision
rule are loaded as-is and applied to test.

Why streaming: 1.73M test S1 entities x ~63 candidates is ~109M pairs, whose
feature matrix would be ~31 GB. So the pipeline never materialises the test
candidate set. It processes one country at a time (building that country's
blocking index once) and, inside it, one batch of entities at a time:

    block -> rerank -> top-K -> features -> predict -> decide -> append rows

An entity's candidates are always wholly inside one batch, which matters
because the context features (rr_rank/rr_best/...) and the decision rule's
best_p are both per-entity quantities. Everything is freed between batches.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from candidate_generation.blocking import (BlockingConfig, _build_keys,  # noqa: E402
                                           _token_df)
from candidate_generation.build_candidates import RERANK_COLS, _attach_lean  # noqa: E402
from candidate_generation.rerank import add_rerank, top_k  # noqa: E402
from features.pair_features import Q_COLS, attach_sides, compute_features  # noqa: E402
from features.prepare import load_norm  # noqa: E402
from models.dataset import add_context, context_matrix  # noqa: E402


# blocking needs these views; the reranker needs RERANK_COLS. They differ.
KEY_COLS = ["entity_id", "nm_core", "nm_compact", "nm_skel", "ad_num", "ad_alpha"]


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def free_ram_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    free = inact = 0
    for line in out.splitlines():
        if "Pages free" in line:
            free = int(line.split(":")[1].strip().rstrip("."))
        elif "Pages inactive" in line:
            inact = int(line.split(":")[1].strip().rstrip("."))
    return (free + inact) * 4096 / 1e9


def decide(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Apply the frozen E07 rule. df carries source1_entity_id, cand_id, p, rr_score."""
    d = df.with_columns(
        pl.col("p").max().over("source1_entity_id").alias("best"),
        pl.when(pl.col("cand_id").str.starts_with("S2-"))
          .then(pl.lit(cfg["s2_threshold"])).otherwise(pl.lit(cfg["s3_threshold"])).alias("thr"),
    )
    ov = cfg["evidence_override"]
    base = (pl.col("p") >= pl.col("thr")) & (pl.col("p") >= pl.col("best") * cfg["rel_margin"])
    over = (pl.col("rr_score") >= ov["rr_min"]) & (pl.col("p") >= ov["p_min"])
    d = d.filter(base | over)
    if cfg.get("max_k"):
        d = (d.sort("p", descending=True)
               .group_by("source1_entity_id", maintain_order=True).head(cfg["max_k"]))
    return d.select("source1_entity_id", "cand_id")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data_cache/models/lgbm_e06_500k.txt")
    ap.add_argument("--config", default="configs/decision_e06.json")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--k", type=int, default=80)
    ap.add_argument("--max-posting", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N entities")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    booster = lgb.Booster(model_file=a.model)
    # the saved feature-name list is the contract with the trained model
    expected = json.loads(
        Path(str(a.model).replace(".txt", ".features.json")).read_text())

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    m_path = out / f"matching_results{a.tag}.tsv"
    c_path = out / f"candidate_pairs{a.tag}.tsv"

    s1 = load_norm("test", "source1")
    if a.limit:
        s1 = s1.head(a.limit)
    pool = pl.concat([load_norm("test", "source2"), load_norm("test", "source3")])
    all_ids = s1["entity_id"].to_list()
    print(f"test S1 entities: {len(all_ids):,}   pool: {pool.height:,}", flush=True)

    bcfg = BlockingConfig(max_posting=a.max_posting, max_cand_per_s1=0)
    seen = set()
    n_pairs_total = 0
    t0 = time.time()
    feat_order_checked = False

    with open(m_path, "w", encoding="utf-8") as fm, open(c_path, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for country in pool["ctry"].unique().to_list():
            q_all = s1.filter(pl.col("ctry") == country)
            p_all = pool.filter(pl.col("ctry") == country)
            if q_all.height == 0 or p_all.height == 0:
                continue
            p_idx = p_all.select(KEY_COLS).with_row_index("ci")
            rare = _token_df(p_idx)
            pk = _build_keys(p_idx, "ci", bcfg, rare)
            post = pk.group_by("k").len().filter(pl.col("len") <= bcfg.max_posting)
            pk = pk.join(post.select("k"), on="k", how="inner")
            del post
            gc.collect()
            p_map = p_idx.select("ci", pl.col("entity_id").alias("cand_id"))
            p_lean = p_all.select(["entity_id"] + RERANK_COLS)
            p_full = p_all.select(["entity_id"] + Q_COLS)
            print(f"  [{country}] queries={q_all.height:,} pool={p_all.height:,} "
                  f"index={pk.height:,} ({time.time()-t0:.0f}s, RAM free {free_ram_gb():.1f}GB)",
                  flush=True)

            for lo in range(0, q_all.height, a.batch_size):
                qb = q_all[lo:lo + a.batch_size]
                qi = qb.select(KEY_COLS).with_row_index("qi")
                qk = _build_keys(qi, "qi", bcfg, rare)
                pairs = (qk.join(pk, on="k", how="inner")
                           .group_by("qi", "ci").len().rename({"len": "n_keys"}))
                del qk
                pairs = (pairs
                         .join(qi.select("qi", pl.col("entity_id").alias("source1_entity_id")), on="qi")
                         .join(p_map, on="ci")
                         .select("source1_entity_id", "cand_id", "n_keys"))
                pairs = top_k(add_rerank(_attach_lean(pairs, qb, p_lean)), a.k)
                cand = pairs.select("source1_entity_id", "cand_id", "n_keys", "rr_score")
                del pairs, qi
                gc.collect()
                n_pairs_total += cand.height

                # write the candidate set actually fed to the model
                cg = cand.group_by("source1_entity_id").agg(pl.col("cand_id"))
                for eid, ids in cg.iter_rows():
                    fc.write(f"{eid}\t{','.join(ids)}\n")
                    seen.add(eid)
                del cg

                # features -> probability
                ctx = add_context(cand)
                j = attach_sides(ctx.select("source1_entity_id", "cand_id"), qb, p_full)
                j = j.join(ctx, on=["source1_entity_id", "cand_id"], how="left")
                X_sim, n_sim = compute_features(j)
                X_ctx, n_ctx = context_matrix(j)
                X = np.hstack([X_sim, X_ctx])
                if not feat_order_checked:
                    assert (n_sim + n_ctx) == expected, "feature order differs from training!"
                    feat_order_checked = True
                    print(f"    feature order verified against {Path(a.model).stem}"
                          f".features.json ({len(expected)} features)", flush=True)
                del X_sim, X_ctx
                p = booster.predict(X, num_threads=a.n_jobs).astype(np.float32)
                del X
                scored = j.select("source1_entity_id", "cand_id").with_columns(
                    pl.Series("p", p), pl.Series("rr_score", j["rr_score"]))
                del j, ctx
                gc.collect()

                matched = decide(scored, cfg)
                mg = matched.group_by("source1_entity_id").agg(pl.col("cand_id"))
                got = {}
                for eid, ids in mg.iter_rows():
                    got[eid] = ids
                for eid in qb["entity_id"].to_list():
                    fm.write(f"{eid}\t{','.join(got.get(eid, []))}\n")
                del scored, matched, mg, got, cand, p
                gc.collect()

                done = lo + qb.height
                if (lo // a.batch_size) % 10 == 0:
                    el = time.time() - t0
                    print(f"    [{country}] {done:,}/{q_all.height:,}  pairs={n_pairs_total:,}  "
                          f"{el:.0f}s  RAM free {free_ram_gb():.1f}GB  peakRSS {peak_rss_gb():.1f}GB",
                          flush=True)
            del pk, p_idx, p_map, p_lean, p_full, p_all, rare
            gc.collect()

        # entities that produced no candidates still need a row in both files
        missing = [e for e in all_ids if e not in seen]
        for eid in missing:
            fc.write(f"{eid}\t\n")
        print(f"  entities with zero candidates: {len(missing):,}", flush=True)

    print(f"\nDONE in {time.time()-t0:.0f}s  peak RSS {peak_rss_gb():.2f} GB")
    print(f"  {m_path}  /  {c_path}")
    print(f"  total candidate pairs scored: {n_pairs_total:,}")
    json.dump({"seconds": time.time()-t0, "peak_rss_gb": peak_rss_gb(),
               "n_pairs": n_pairs_total, "n_entities": len(all_ids),
               "n_zero_cand": len(missing)},
              open(out / f"inference_stats{a.tag}.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
