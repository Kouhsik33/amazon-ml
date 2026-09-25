#!/usr/bin/env python3
"""Compress E06 predictions into a compact NumPy bundle for decision analysis.

The scored Parquet files are dominated by two wide string columns. For decision
work none of that text is needed -- only which entity a row belongs to, which
source it came from, and a few floats. Factorising the entity id to int32 and
dropping cand_id entirely takes val from 40.5 MB on disk / ~1 GB joined in RAM
down to a ~60 MB array bundle, after which every sweep is a vectorised NumPy
pass with no further I/O.

Run once per fold, then exit so the join memory is returned to the OS.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.io import load_ground_truth  # noqa: E402
from data.splits import load_split  # noqa: E402

OUT = Path("experiments/decision")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=["dev", "val"])
    ap.add_argument("--scored", required=True)
    ap.add_argument("--cand", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ids = load_split(a.fold)                      # every entity, incl. zero-candidate
    idx = {e: i for i, e in enumerate(ids)}
    n_ent = len(ids)

    # true match count per entity, from ground truth -- NOT from the candidate
    # set, because blocking misses must still count against recall
    gt = load_ground_truth().filter(pl.col("source1_entity_id").is_in(ids))
    n_true = np.zeros(n_ent, dtype=np.int16)
    for eid, m in gt.iter_rows():
        if m:
            n_true[idx[eid]] = m.count(",") + 1
    del gt
    gc.collect()

    sc = pl.read_parquet(a.scored, columns=["source1_entity_id", "cand_id", "y", "p"])
    cd = pl.read_parquet(a.cand, columns=["source1_entity_id", "cand_id", "n_keys", "rr_score"])
    j = sc.join(cd, on=["source1_entity_id", "cand_id"], how="left")
    del sc, cd
    gc.collect()

    gidx = j["source1_entity_id"].replace_strict(idx, return_dtype=pl.Int32).to_numpy()
    is_s2 = j["cand_id"].str.starts_with("S2-").to_numpy()
    p = j["p"].to_numpy().astype(np.float32)
    y = j["y"].to_numpy().astype(np.int8)
    n_keys = j["n_keys"].fill_null(0).to_numpy().astype(np.uint8)
    rr = j["rr_score"].fill_null(0.0).to_numpy().astype(np.float32)
    del j
    gc.collect()

    order = np.lexsort((-p, gidx))                # group rows, best-first inside group
    out = OUT / f"compact_{a.fold}_e06.npz"
    np.savez_compressed(
        out, gidx=gidx[order].astype(np.int32), is_s2=is_s2[order],
        p=p[order], y=y[order], n_keys=n_keys[order], rr=rr[order],
        n_true=n_true, entity_ids=np.array(ids, dtype=object),
    )
    mb = out.stat().st_size / 1e6
    print(f"{a.fold}: {len(p):,} pairs, {n_ent:,} entities, "
          f"{int((n_true == 0).sum()):,} singletons, "
          f"{int(y.sum()):,} true pairs in candidates, "
          f"{int(n_true.sum()):,} true pairs total -> {out.name} ({mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
