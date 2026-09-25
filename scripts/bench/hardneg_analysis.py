#!/usr/bin/env python3
"""Does scaling training coverage add examples near the decision boundary?

Scores the 200k and 500k *training* candidate sets with the frozen E05 model
and compares how the negatives distribute relative to E05's operating
thresholds. If the extra 300k entities mostly contribute easy negatives, the
boundary region grows only proportionally and extra data buys little; if it
grows super-proportionally, the extra coverage is genuinely informative.

Uses the E05 model (not E06) so the yardstick is identical for both sets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.io import load_ground_truth  # noqa: E402
from models.dataset import add_context, label, sample_negatives  # noqa: E402
from models.train_lgbm import featurize, load_sides  # noqa: E402

T_S2, T_S3 = 0.875, 0.85          # frozen E05 operating point
MODEL = "data_cache/models/lgbm_e05.txt"


def analyse(cand_path: str, tag: str, sides, booster, gt) -> dict:
    tr = add_context(pl.read_parquet(cand_path))
    tr = sample_negatives(label(tr, gt), 8, 4)
    X, _, meta = featurize(tr, "train", sides=sides,
                           mmap_path=f"/tmp/erprof/hn_{tag}.mmap")
    p = booster.predict(X).astype(np.float32)
    del X

    y = meta["y"].to_numpy()
    is_s2 = meta["cand_id"].str.starts_with("S2-").to_numpy()
    thr = np.where(is_s2, T_S2, T_S3)

    neg, pos = y == 0, y == 1
    # "boundary band": within +/-0.15 probability of that pair's threshold
    band = np.abs(p - thr) <= 0.15
    n_ent = tr["source1_entity_id"].n_unique()

    res = {
        "tag": tag,
        "n_entities": int(n_ent),
        "n_pairs": int(len(p)),
        "n_pos": int(pos.sum()),
        "n_neg": int(neg.sum()),
        "pos_rate": float(pos.mean()),
        # negatives the model already scores above threshold = active FP risk
        "neg_above_thr": int((neg & (p >= thr)).sum()),
        "neg_above_thr_per_entity": float((neg & (p >= thr)).sum() / n_ent),
        # positives below threshold = active FN risk
        "pos_below_thr": int((pos & (p < thr)).sum()),
        "pos_below_thr_per_entity": float((pos & (p < thr)).sum() / n_ent),
        # everything close to the boundary, either label
        "in_band": int(band.sum()),
        "in_band_per_entity": float(band.sum() / n_ent),
        "neg_in_band": int((neg & band).sum()),
        "pos_in_band": int((pos & band).sum()),
        "frac_pairs_in_band": float(band.mean()),
        # how hard are the hard negatives overall
        "neg_p_mean": float(p[neg].mean()),
        "neg_p_p99": float(np.percentile(p[neg], 99)),
        "neg_p_p999": float(np.percentile(p[neg], 99.9)),
    }
    del meta, tr
    Path(f"/tmp/erprof/hn_{tag}.mmap").unlink(missing_ok=True)
    return res


def main() -> int:
    gt = load_ground_truth()
    sides = load_sides("train")
    booster = lgb.Booster(model_file=MODEL)
    out = []
    for path, tag in [("data_cache/candidates/cand_train_200k_k80.parquet", "200k"),
                      ("data_cache/candidates/cand_train_500k_k80.parquet", "500k")]:
        r = analyse(path, tag, sides, booster, gt)
        out.append(r)
        print(json.dumps(r, indent=2), flush=True)

    a, b = out
    scale = b["n_entities"] / a["n_entities"]
    print("\n=== scaling 200k -> 500k (entity scale factor "
          f"{scale:.2f}x) ===")
    for k in ("n_pairs", "n_pos", "n_neg", "in_band", "neg_in_band",
              "pos_in_band", "neg_above_thr", "pos_below_thr"):
        growth = b[k] / a[k] if a[k] else float("nan")
        print(f"  {k:20s} {a[k]:>12,} -> {b[k]:>12,}   {growth:5.2f}x  "
              f"({'super' if growth > scale * 1.02 else 'sub' if growth < scale * 0.98 else 'exactly'}-proportional)")
    Path("experiments/bench/hardneg_analysis.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
