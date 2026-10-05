"""Episode-level results with frozen, leakage-free thresholds. usage: python run_frozen.py flu|covid"""
import sys, os, json, time, numpy as np, pandas as pd
from common import case_features, episodes, sens_at_spec
from frozen import row_thresholds, crossing_weeks, first_alerts, episode_metrics, rowlevel
DS = sys.argv[1]; TAG = 'H3_R1.5'; R = 1.5; A = 5.0 if DS == 'flu' else 10.0; LOG = DS == 'covid'
df = pd.read_parquet(f'panel_{DS}_{TAG}.parquet').reset_index(drop=True)
P = pd.read_parquet(f'preds/{DS}_{TAG}.parquet'); P['date'] = pd.to_datetime(P.date)
assert (P.iso3.values == df.iso3.values).all()
y = P.y.values.astype(int); g = P.iso3.values
F = case_features(df); X = df[F].values.astype(np.float32)
Xc = np.column_stack([X, P.chronos_prob.values.astype(np.float32)])
eps = crossing_weeks(P, episodes(P), R, A, LOG)
MODELS = {'growth': 'train', 'expgrowth': 'train', 'chronos': 'train', 'logreg': 'nested', 'rf': 'otherfold',
          'lgbm': 'nested', 'cnnlstm_rf': 'otherfold', 'tabpfn': 'otherfold', 'lgbm_chronos': 'nested', 'tft': 'otherfold'}
rows = []; t0 = time.time()
for m, method in MODELS.items():
    per = []
    for seed in (0, 1, 2):
        col = f'{m}_s{seed}' if f'{m}_s{seed}' in P else m
        s = P[col].values
        fn = f'thr_cache/{DS}_{m}_{method}_{seed}.npy'
        if os.path.exists(fn):
            thr = np.load(fn)
        else:
            thr = row_thresholds(method, s, y, g, seed, name=m, X=Xc if m == 'lgbm_chronos' else X); np.save(fn, thr)
        r = rowlevel(y, s, thr); r.update(episode_metrics(eps, first_alerts(P, s > thr, eps)))
        if m == 'lgbm':     # validation of the cheap 'otherfold' rule against full nesting
            thr2 = row_thresholds('otherfold', s, y, g, seed)
            r2 = episode_metrics(eps, first_alerts(P, s > thr2, eps)); r2.update(rowlevel(y, s, thr2))
            r.update({f'otherfold_{k}': v for k, v in r2.items()})
        per.append(r)
    out = {k: float(np.mean([d[k] for d in per])) for k in per[0]}
    out.update({k + '_sd': float(np.std([d[k] for d in per])) for k in ('sens', 'detected', 'ge1')})
    out['model'] = m; out['threshold_method'] = method
    # previous (test-set oracle) threshold, for comparison
    sc = P[[c for c in P.columns if c.startswith(m + '_s')]].mean(1).values if f'{m}_s0' in P else P[m].values
    _, t_or = sens_at_spec(y, sc, 0.95)
    out.update({'oracle_' + k: v for k, v in episode_metrics(eps, first_alerts(P, sc > t_or, eps)).items()})
    rows.append(out); print(m, method, {k: round(v, 3) for k, v in out.items() if isinstance(v, float)}, f'{time.time()-t0:.0f}s', flush=True)
res = pd.DataFrame(rows).set_index('model'); res['episodes'] = len(eps)
res.to_csv(f'results_frozen_{DS}.csv')
print(res.round(3).to_string())
