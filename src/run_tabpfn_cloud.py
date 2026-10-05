"""TabPFN v2 via the Prior Labs cloud API (tabpfn-client), country-held-out.
Per fold: random 10,000-row training subsample of training countries (TabPFN's design limit),
predictions for every held-out country-week. Token read from ~/.tabpfn_token (never stored in the repo).
usage: python run_tabpfn_cloud.py flu|covid"""
import os, sys, time, numpy as np, pandas as pd
import tabpfn_client as tc
from tabpfn_client import TabPFNClassifier
from sklearn.impute import SimpleImputer
from common import case_features, country_folds
tc.set_access_token(open(os.path.expanduser('~/.tabpfn_token')).read().strip())
DS = sys.argv[1]; TAG = 'H3_R1.5'; NTRAIN = 10000
df = pd.read_parquet(f'panel_{DS}_{TAG}.parquet').reset_index(drop=True)
F = case_features(df); X = df[F].values.astype(np.float32); y = df.y.values.astype(int)
out = {}
for seed in (0, 1, 2):
    pred = np.full(len(df), np.nan); t0 = time.time()
    for k, (tr, te) in enumerate(country_folds(df.iso3.values, 5, seed)):
        rng = np.random.RandomState(seed * 10 + k)
        sub = rng.choice(tr, min(NTRAIN, len(tr)), replace=False)
        imp = SimpleImputer(strategy='median').fit(X[sub])
        for attempt in range(4):
            try:
                m = TabPFNClassifier(random_state=seed)
                m.fit(imp.transform(X[sub]), y[sub])
                parts = [m.predict_proba(imp.transform(X[te[i:i + 10000]]))[:, 1] for i in range(0, len(te), 10000)]
                pred[te] = np.concatenate(parts); break
            except Exception as e:
                print('retry', seed, k, str(e)[:120], flush=True); time.sleep(30 * (attempt + 1))
        print(f'{DS} seed {seed} fold {k} {time.time() - t0:.0f}s', flush=True)
    out[f'tabpfn_s{seed}'] = pred
pd.DataFrame(out).to_parquet(f'preds/{DS}_{TAG}_tabpfn.parquet')   # merged into the main file later (avoids write races)
print('saved', DS, {c: float(np.isnan(v).mean()) for c, v in out.items()}, flush=True)
print(tc.get_api_usage())
