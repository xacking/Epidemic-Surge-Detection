"""Evaluate prediction files. usage: python evaluate.py flu H3_R1.5"""
import sys, json, numpy as np, pandas as pd
from common import metrics, cluster_boot_diff, episodes, sens_at_spec

DS = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else 'H3_R1.5'
H = int(TAG.split('_')[0][1:]); R = float(TAG.split('_')[1][1:])
A = 5.0 if DS == 'flu' else 10.0
LOG = DS == 'covid'
P = pd.read_parquet(f'preds/{DS}_{TAG}.parquet')
P['date'] = pd.to_datetime(P.date)
NAMES = {'growth': 'Growth-rate rule', 'expgrowth': 'Exponential-growth extrapolation', 'chronos': 'Chronos-Bolt (zero-shot)',
         'logreg': 'Logistic regression', 'rf': 'Random forest', 'lgbm': 'LightGBM', 'cnnlstm_rf': 'CNN-LSTM-RF (thesis hybrid)',
         'tabpfn': 'TabPFN v2', 'lgbm_chronos': 'LightGBM + Chronos forecast feature', 'tft': 'Temporal Fusion Transformer'}
models = [m for m in NAMES if m in P.columns or any(c.startswith(m + '_s') for c in P.columns)]
y = P.y.values.astype(int)
rows = []
for m in models:
    seeds = [c for c in P.columns if c.startswith(m + '_s')]
    if seeds:
        per = [metrics(y[P[c].notna()], P.loc[P[c].notna(), c].values, P.loc[P[c].notna(), c].values) for c in seeds]
        P[m] = P[seeds].mean(1)
        r = {k: np.mean([d[k] for d in per]) for k in per[0]}
        r.update({k + '_sd': np.std([d[k] for d in per]) for k in ('roc_auc', 'pr_auc', 'sens_at_95spec')})
    else:
        r = metrics(y, P[m].values, P[m + '_prob'].values if m + '_prob' in P else None)
        r.update(roc_auc_sd=0.0, pr_auc_sd=0.0, sens_at_95spec_sd=0.0)
    r['model'] = m; r['name'] = NAMES[m]
    rows.append(r)
res = pd.DataFrame(rows).set_index('model')

# per-model cluster-bootstrap 95% CI (countries resampled) for PR-AUC and ROC-AUC of seed-averaged scores
from sklearn.metrics import average_precision_score as _ap, roc_auc_score as _roc
_rng = np.random.RandomState(1); _grp = P.groupby('iso3').indices; _keys = np.array(list(_grp.keys()))
_samples = [np.concatenate([_grp[k] for k in _rng.choice(_keys, len(_keys))]) for _ in range(500)]
for m in models:
    s = P[m].values
    pr = [_ap(y[i], s[i]) for i in _samples]; rc = [_roc(y[i], s[i]) for i in _samples]
    res.loc[m, 'pr_lo'], res.loc[m, 'pr_hi'] = np.percentile(pr, [2.5, 97.5])
    res.loc[m, 'roc_lo'], res.loc[m, 'roc_hi'] = np.percentile(rc, [2.5, 97.5])
    res.loc[m, 'pr_avg'] = _ap(y, s); res.loc[m, 'roc_avg'] = _roc(y, s)

# paired cluster bootstrap vs LightGBM (seed-averaged scores)
ref = 'lgbm'
for m in models:
    if m == ref:
        continue
    for met in ('pr_auc', 'roc_auc'):
        d, lo, hi, p = cluster_boot_diff(P, m, ref, met, B=1000)
        res.loc[m, f'd_{met}_vs_lgbm'] = d; res.loc[m, f'd_{met}_lo'] = lo; res.loc[m, f'd_{met}_hi'] = hi; res.loc[m, f'p_{met}'] = p

# ---------- episode-level detection and lead time ----------
eps = episodes(P)
by = {c: g.sort_values('date').reset_index(drop=True) for c, g in P.groupby('iso3')}
cross = []
for _, e in eps.iterrows():
    g = by[e.iso3]; s0 = g[g.date == e.start].iloc[0]
    b = s0.baseline; thr = max(R * b, b + A)
    x = np.expm1(g.x_lag0) if LOG else g.x_lag0
    after = g[(g.date > e.start) & (x >= thr)]
    cross.append(after.date.iloc[0] if len(after) else pd.NaT)
eps['cross'] = cross
eps = eps.dropna(subset=['cross'])
lead = {}
for m in models:
    _, thr = sens_at_spec(y, P[m].values, 0.95)
    fa = []
    for _, e in eps.iterrows():
        g = by[e.iso3]
        w = g[(g.date >= e.start - pd.Timedelta(weeks=4)) & (g.date <= e.end)]
        hit = w[w[m] > thr]
        fa.append(hit.date.iloc[0] if len(hit) else pd.NaT)
    fa = pd.Series(fa, index=eps.index)
    L = (eps.cross - fa).dt.days / 7
    eps[m + '_lead'] = L
    det = L.notna()
    lead[m] = dict(episodes=int(len(eps)), detected=float(det.mean()), median_lead=float(L[det].median()),
                   mean_lead=float(L[det].mean()), frac_ge1wk=float((L >= 1).mean()))
res = res.join(pd.DataFrame(lead).T)
res.to_csv(f'results_{DS}_{TAG}.csv')
eps.to_csv(f'episodes_{DS}_{TAG}.csv', index=False)
pd.set_option('display.width', 250)
print(res[['name', 'roc_auc', 'roc_auc_sd', 'pr_auc', 'pr_auc_sd', 'sens_at_95spec', 'brier', 'd_pr_auc_vs_lgbm', 'd_pr_auc_lo', 'd_pr_auc_hi',
           'detected', 'median_lead', 'mean_lead', 'frac_ge1wk']].round(3).to_string())
print('prevalence', y.mean().round(3), 'countries', P.iso3.nunique(), 'rows', len(P), 'episodes', len(eps))
