import lightgbm as lgb
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

W = Path(r"D:\Dataset_ML_C\work")
t0 = time.time()
F = pd.read_parquet(W / "dev_features.parquet")
pairs = pd.read_pickle(W / "sample_pairs.pkl")
s1_all = pd.read_pickle(W / "sample_s1.pkl").entity_id.values
feats = [c for c in F.columns if c not in ("is_match", "s1_id", "cand_id")]
y = F.is_match.values.astype(np.float32)
X = F[feats].to_numpy(dtype=np.float32)

rng = np.random.default_rng(0)
fold_of = pd.Series(rng.integers(0, 4, len(s1_all)), index=s1_all)
fold = F.s1_id.map(fold_of).values

params = dict(objective="binary", learning_rate=0.06, num_leaves=127, min_child_samples=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=12)
oof = np.zeros(len(F))
imp = np.zeros(len(feats))
for k in range(4):
    tr, va = fold != k, fold == k
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr], feature_name=feats), num_boost_round=400)
    oof[va] = m.predict(X[va])
    imp += m.feature_importance("gain")
    print(f"fold {k} auc={roc_auc_score(y[va] > 0, oof[va]):.5f} ap={average_precision_score(y[va] > 0, oof[va]):.5f}  ({time.time()-t0:.0f}s)", flush=True)
print(pd.Series(imp / imp.sum(), index=feats).sort_values(ascending=False).head(20).round(4).to_string())
F["p"] = oof
F[["s1_id", "cand_id", "p", "is_match"]].to_parquet(W / "dev_oof.parquet")

n_true = pairs.groupby("source1_entity_id").size().reindex(s1_all).fillna(0).astype(int)


def macro_f05(sel):
    """sel: DataFrame with s1_id, is_match for the predicted pairs. Returns macro F0.5 over all dev S1 entities."""
    npred = sel.groupby("s1_id").size().reindex(s1_all).fillna(0)
    tp = sel.groupby("s1_id").is_match.sum().reindex(s1_all).fillna(0)
    nt = n_true
    prec = np.where(npred > 0, tp / npred.clip(lower=1), 0.0)
    rec = np.where(nt > 0, tp / nt.clip(lower=1), 0.0)
    f = np.where((prec + rec) > 0, 1.25 * prec * rec / (0.25 * prec + rec + 1e-12), 0.0)
    f = np.where(nt == 0, (npred == 0).astype(float), f)
    return f.mean(), prec[nt > 0].mean(), rec[nt > 0].mean()


def owner_dedupe(d):
    return d.loc[d.groupby("cand_id").p.idxmax()]


print("\n--- decision rules (OOF, all dev S1 entities) ---")
ub = F[F.is_match]
print("oracle within candidates (all true pairs in candidate set):", round(macro_f05(ub)[0], 4))
best = (0, None)
for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    a = macro_f05(F[F.p >= t])[0]
    d = owner_dedupe(F[F.p >= 0.05])
    b = macro_f05(d[d.p >= t])
    print(f"t={t}: threshold-only F0.5={a:.4f} | + one-owner-per-B F0.5={b[0]:.4f} (P={b[1]:.4f} R={b[2]:.4f})")
