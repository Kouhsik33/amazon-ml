import sys
sys.path.insert(0, r"D:\Dataset_ML_C\submission\code\business_entity_resolution\src")
import lightgbm as lgb
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from common import WORK_DIR, pack_pair
from features import COMP_COLS, PairFeaturizer, competition_features
import s03_train as t

def main():
    HFRAC = 0.15
    truth, n_true = t.load_truth()
    booster = lgb.Booster(model_file=str(WORK_DIR / "model.txt"))
    h = pd.read_parquet(WORK_DIR / "hold_pred.parquet")
    res = {}
    for country in ("US", "India"):
        cdir = WORK_DIR / "train" / country
        cand = pd.read_parquet(cdir / "cand.parquet")
        ra = pd.read_parquet(cdir / "records.parquet")
        recs = {s: ra[ra.src == s].reset_index(drop=True) for s in (1, 2, 3)}
        dup = recs[1].groupby("name_n").num.transform("size")
        comp = competition_features(cand, pd.Series(dup.to_numpy(), index=recs[1].num.to_numpy()))[COMP_COLS].to_numpy(np.float32)
        u = t.u01(cand.s1_num.to_numpy())
        bk = cand.b_num.to_numpy().astype(np.int64) * 2 + (cand.src.to_numpy() == 3)
        hold_b = np.unique(bk[(u >= HFRAC * 0.7) & (u < HFRAC)])
        train_rows = (u < HFRAC * 0.7) & ~np.isin(bk, hold_b)
        sel = np.flatnonzero(train_rows)
        sel = sel[np.random.default_rng(0).permutation(len(sel))[:1_500_000]]
        c = cand.iloc[sel].reset_index(drop=True)
        pf = PairFeaturizer(str(cdir / "idf.pkl"), recs)
        base = pf.compute(c); pf.close()
        X = np.concatenate([base, comp[sel]], axis=1)
        y = np.isin(pack_pair(c.s1_num.to_numpy(), c.b_num.to_numpy(), c.src.to_numpy()), truth)
        p = booster.predict(X, num_iteration=booster.best_iteration)
        hh = h[(h.country == country) & h.hold_s1]
        def m(y, p):
            acc = p >= 0.7
            return {"rows": len(y), "positives": int(y.sum()), "AUC": roc_auc_score(y, p), "AP": average_precision_score(y, p),
                    "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "precision@0.7": (y & acc).sum() / max(acc.sum(), 1), "recall@0.7": (y & acc).sum() / y.sum()}
        res[f"{country} training rows"] = m(y, p)
        res[f"{country} held-out rows"] = m(hh.y.to_numpy(), hh.p.to_numpy())
        print(country, "done", flush=True)
        del cand, comp, X, base
    out = pd.DataFrame(res).T
    pd.set_option("display.width", 250)
    print(out.round(5).to_string())


if __name__ == '__main__':
    main()
