"""Step 2: normalize + block. Usage: python s02_block.py {train,test} [country ...]

Writes, per country, to WORK_DIR/<split>/<country>/ :
  records.parquet   normalized records of all three sources (src, num, name_n, addr_n, name_nonlatin)
  idf.pkl           token IDF tables (name / address) used by the pair features
  cand.parquet      candidate pairs (s1_num, b_num, src, sn, sa)
"""
import gc
import json
import math
import pickle
import sys
import time

import numpy as np
import pandas as pd

from blocking import block_source, gen_keys, make_ctx, name_keys, addr_keys
from common import CAP_FRAC, MIN_CAP, POOL, RERANK, TOP_K, WORK_DIR, country_dir, id_num, read_split
from normalize import normalize_frame

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def idf_table(counter, n):
    return {t: math.log(n / c) for t, c in counter.items()}


def run(split, only=None, frames=None):
    data = frames if frames is not None else read_split(split)
    with open(WORK_DIR / "lexicon.json", encoding="utf-8") as f:
        lexicon = json.load(f)
    countries = sorted(data["s1"].country.dropna().unique())
    for country in countries:
        if only and country not in only:
            continue
        log(f"=== {split} / {country}")
        recs = {}
        for n in ("s1", "s2", "s3"):
            sub = data[n][data[n].country == country].reset_index(drop=True)
            r = normalize_frame(sub, None if n == "s1" else lexicon)
            r["num"] = id_num(r.entity_id)
            recs[n] = r.drop(columns=["entity_id"])
        log("normalized: " + ", ".join(f"{n}={len(r):,}" for n, r in recs.items()))
        untranslated = {t for n in ("s2", "s3") for s in recs[n].name_n.values if not s.isascii() for t in s.split() if not t.isascii()}
        log(f"non-Latin tokens left untranslated (not in lexicon): {len(untranslated)}")

        all_names = np.concatenate([r.name_n.values for r in recs.values()])
        all_addrs = np.concatenate([r.addr_n.values for r in recs.values()])
        ctx_name, ctx_addr = make_ctx(all_names, all_addrs)
        n_all = len(all_names)
        idf = {"name": idf_table(ctx_name[0], n_all), "addr": idf_table(ctx_addr[0], n_all), "n": n_all}
        del all_names, all_addrs
        log("token statistics done")

        out_dir = country_dir(split, country)
        rec_all = pd.concat([r.assign(src=int(n[1])) for n, r in recs.items()], ignore_index=True)
        rec_all.to_parquet(out_dir / "records.parquet", index=False)
        with open(out_dir / "idf.pkl", "wb") as f:
            pickle.dump(idf, f, protocol=4)
        del rec_all

        a_keys = gen_keys(recs["s1"].name_n.values, name_keys, ctx_name)[:2] + gen_keys(recs["s1"].addr_n.values, addr_keys, ctx_addr)[:2]
        log(f"S1 keys: {len(a_keys[1]) + len(a_keys[3]):,}")
        cands = []
        for src in ("s2", "s3"):
            b = recs[src]
            rn, kn = gen_keys(b.name_n.values, name_keys, ctx_name)
            ra, ka = gen_keys(b.addr_n.values, addr_keys, ctx_addr)
            log(f"{src} keys: {len(kn) + len(ka):,}")
            c = block_source(a_keys, len(recs["s1"]), (rn, kn, ra, ka), len(b), TOP_K, CAP_FRAC, MIN_CAP, pool=POOL, rerank=RERANK, log=log)
            del rn, kn, ra, ka
            gc.collect()
            cands.append(pd.DataFrame({
                "s1_num": recs["s1"].num.values[c.i.values].astype(np.int32),
                "b_num": b.num.values[c.j.values].astype(np.int32),
                "src": np.int8(int(src[1])),
                "sn": c.sn.values, "sa": c.sa.values, "sj": c.sj.values,
            }))
            log(f"{src}: {len(c):,} candidate pairs ({len(c) / len(recs['s1']):.1f} per S1)")
            del c
        pd.concat(cands, ignore_index=True).to_parquet(out_dir / "cand.parquet", index=False)
        del recs, cands, a_keys, ctx_name, ctx_addr, idf
        gc.collect()
        log(f"{country} done")


if __name__ == "__main__":
    run(sys.argv[1], set(sys.argv[2:]) or None)
