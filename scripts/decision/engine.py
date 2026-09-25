"""Vectorised macro-F0.5 evaluator over frozen E06 probabilities.

Rows arrive pre-sorted by (entity, -probability), so per-entity best/second/
count are slice reads and every candidate decision rule is a boolean mask plus
two bincounts. A full sweep point costs ~20 ms on 2.5M rows, which is what
makes an honest grid affordable without touching the model.

Scoring convention is identical to src/evaluation/metrics.py: singleton +
empty = 1.0, singleton + any prediction = 0.0, non-singleton + empty = 0.0,
otherwise F0.5 on the intersection. n_true comes from ground truth, so
blocking misses correctly count as recall loss.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

DEC = Path("experiments/decision")


@dataclass
class Rule:
    t_s2: float = 0.875
    t_s3: float = 0.85
    rel_margin: float = 0.9          # keep p >= best * rel_margin
    min_best: float = 0.0            # singleton gate: drop entity if best < this
    max_k: int = 11
    gap_min: float = 0.0             # require (best - second) >= this
    # evidence override: additionally accept a pair the thresholds rejected when
    # the stored lexical reranker score / blocking agreement is strong enough.
    # rr_score bundles name (0.50), address (0.28), numeric (0.14) and key
    # agreement (0.08); it is the only similarity signal persisted, since the
    # per-feature matrix was not saved.
    override_rr: float = 2.0         # 2.0 = disabled (rr is <= 1)
    override_p: float = 1.1          # 1.1 = disabled
    # candidate-count-dependent threshold bumps: list of (count_gt, delta)
    count_bumps: tuple = ()

    def as_dict(self):
        d = asdict(self)
        d["count_bumps"] = list(self.count_bumps)
        return d


class Fold:
    def __init__(self, fold: str):
        z = np.load(DEC / f"compact_{fold}_e06.npz", allow_pickle=True)
        self.fold = fold
        self.gidx = z["gidx"]
        self.is_s2 = z["is_s2"]
        self.p = z["p"]
        self.y = z["y"].astype(bool)
        self.n_keys = z["n_keys"]
        self.rr = z["rr"]
        self.n_true = z["n_true"].astype(np.int32)
        self.entity_ids = z["entity_ids"]
        self.n_ent = len(self.n_true)

        # group boundaries (rows already sorted by gidx, then -p)
        uniq, starts, counts = np.unique(self.gidx, return_index=True, return_counts=True)
        best = np.zeros(self.n_ent, np.float32)
        second = np.zeros(self.n_ent, np.float32)
        cnt = np.zeros(self.n_ent, np.int32)
        best[uniq] = self.p[starts]
        has2 = counts >= 2
        second[uniq[has2]] = self.p[starts[has2] + 1]
        cnt[uniq] = counts
        self.best, self.second, self.cnt = best, second, cnt
        self.gap = best - second

        # per-source true counts, for S2-/S3-only macro scores
        cache = DEC / f"ntrue_src_{fold}.npz"
        if cache.exists():
            c = np.load(cache)
            self.n_true_s2, self.n_true_s3 = c["s2"], c["s3"]
        else:
            from data.io import load_ground_truth
            idx = {e: i for i, e in enumerate(self.entity_ids)}
            s2 = np.zeros(self.n_ent, np.int32)
            s3 = np.zeros(self.n_ent, np.int32)
            gt = load_ground_truth().filter(
                pl.col("source1_entity_id").is_in(list(self.entity_ids)))
            for eid, m in gt.iter_rows():
                if not m:
                    continue
                i = idx[eid]
                for t in m.split(","):
                    if t.startswith("S2-"):
                        s2[i] += 1
                    else:
                        s3[i] += 1
            np.savez_compressed(cache, s2=s2, s3=s3)
            self.n_true_s2, self.n_true_s3 = s2, s3

    def accept_mask(self, r: Rule) -> np.ndarray:
        thr = np.where(self.is_s2, r.t_s2, r.t_s3).astype(np.float32)
        if r.count_bumps:
            bump = np.zeros(self.n_ent, np.float32)
            for cnt_gt, delta in r.count_bumps:
                bump[self.cnt > cnt_gt] = delta
            thr = thr + bump[self.gidx]
        m = self.p >= thr
        if r.rel_margin > 0:
            m &= self.p >= self.best[self.gidx] * r.rel_margin
        if r.min_best > 0:
            m &= self.best[self.gidx] >= r.min_best
        if r.gap_min > 0:
            m &= self.gap[self.gidx] >= r.gap_min
        if r.override_rr <= 1.0:
            m |= (self.rr >= r.override_rr) & (self.p >= r.override_p)
        if r.max_k:
            # rows are best-first within a group; keep the first max_k accepted
            rank = np.zeros(len(m), np.int32)
            sel = np.flatnonzero(m)
            if sel.size:
                g = self.gidx[sel]
                newg = np.ones(len(g), bool)
                newg[1:] = g[1:] != g[:-1]
                rank[sel] = np.arange(len(g)) - np.maximum.accumulate(
                    np.where(newg, np.arange(len(g)), 0))
                m = m.copy()
                m[sel[rank[sel] >= r.max_k]] = False
        return m

    def score(self, r: Rule) -> dict:
        m = self.accept_mask(r)
        g = self.gidx[m]
        n_pred = np.bincount(g, minlength=self.n_ent).astype(np.int32)
        tp = np.bincount(self.gidx[m & self.y], minlength=self.n_ent).astype(np.int32)
        return self._metrics(n_pred, tp, m)

    def _f05(self, tp, n_pred, n_true):
        f = np.zeros(self.n_ent, np.float64)
        single = n_true == 0
        f[single & (n_pred == 0)] = 1.0
        ok = (~single) & (n_pred > 0)
        P = np.zeros(self.n_ent); R = np.zeros(self.n_ent)
        P[ok] = tp[ok] / n_pred[ok]
        R[ok] = tp[ok] / n_true[ok]
        den = 0.25 * P + R
        good = ok & (den > 0)
        f[good] = 1.25 * P[good] * R[good] / den[good]
        return f

    def _metrics(self, n_pred, tp, m) -> dict:
        f = self._f05(tp, n_pred, self.n_true)
        single = self.n_true == 0
        s2m = m & self.is_s2
        s3m = m & (~self.is_s2)
        np2 = np.bincount(self.gidx[s2m], minlength=self.n_ent).astype(np.int32)
        tp2 = np.bincount(self.gidx[s2m & self.y], minlength=self.n_ent).astype(np.int32)
        np3 = np.bincount(self.gidx[s3m], minlength=self.n_ent).astype(np.int32)
        tp3 = np.bincount(self.gidx[s3m & self.y], minlength=self.n_ent).astype(np.int32)
        TP, FP = int(tp.sum()), int(n_pred.sum() - tp.sum())
        FN = int(self.n_true.sum() - TP)
        has = n_pred > 0
        return {
            "macro_f05": float(f.mean()),
            "mean_precision": float((tp[has] / n_pred[has]).mean()) if has.any() else 0.0,
            "mean_recall": float(np.divide(tp[has], np.maximum(self.n_true[has], 1)).mean()) if has.any() else 0.0,
            "singleton_acc": float((n_pred[single] == 0).mean()),
            "s2_f05": float(self._f05(tp2, np2, self.n_true_s2).mean()),
            "s3_f05": float(self._f05(tp3, np3, self.n_true_s3).mean()),
            "TP": TP, "FP": FP, "FN": FN,
            "n_pred": int(n_pred.sum()),
            "singleton_fp_entities": int(((n_pred > 0) & single).sum()),
        }
