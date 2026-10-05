"""LightGBM + zero-shot Chronos forecast as an extra feature (foundation-model forecast fed to a trained detector).
Chronos was never trained on any panel country, so its output is a valid feature in country-held-out folds."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from common import case_features, country_folds
DS = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else 'H3_R1.5'
df = pd.read_parquet(f'panel_{DS}_{TAG}.parquet').reset_index(drop=True)
P = pd.read_parquet(f'preds/{DS}_{TAG}.parquet')
assert (P.iso3.values == df.iso3.values).all()
df['chronos_p'] = P.chronos_prob.values
F = case_features(df) + ['chronos_p']
X = df[F].values.astype(np.float32); y = df.y.values.astype(int)
for seed in (0, 1, 2):
    pred = np.full(len(df), np.nan)
    for tr, te in country_folds(df.iso3.values, 5, seed):
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=2)
        m.fit(X[tr], y[tr]); pred[te] = m.predict_proba(X[te])[:, 1]
    P[f'lgbm_chronos_s{seed}'] = pred
P.to_parquet(f'preds/{DS}_{TAG}.parquet')
print('saved lgbm_chronos', DS)
