"""Country AND time held out (reviewer R1-2): train on training countries' weeks before CUTOFF, evaluate on
held-out countries' weeks from CUTOFF onward. Thresholds: nested inner CV on the pre-cutoff training rows
(zero-shot scorers: quantile of pre-cutoff training-country scores). usage: python run_timesplit.py flu|covid"""
import sys, os, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from common import case_features, country_folds, metrics, episodes
from frozen import q_neg, make_model, crossing_weeks, first_alerts, episode_metrics
DS = sys.argv[1]; TAG = 'H3_R1.5'; R = 1.5; A = 5.0 if DS == 'flu' else 10.0; LOG = DS == 'covid'
CUTOFF = pd.Timestamp('2020-01-01' if DS == 'flu' else '2021-07-01')
df = pd.read_parquet(f'panel_{DS}_{TAG}.parquet').reset_index(drop=True)
P0 = pd.read_parquet(f'preds/{DS}_{TAG}.parquet')
y = df.y.values.astype(int); g = df.iso3.values; d = df.date.values
F = case_features(df); X = df[F].values.astype(np.float32)
Xc = np.column_stack([X, P0.chronos_prob.values.astype(np.float32)])
P = df[['iso3', 'date', 'y', 'baseline', 'x_lag0']].copy()
eps = crossing_weeks(P, episodes(P), R, A, LOG); eps = eps[eps.start >= CUTOFF].reset_index(drop=True)


def tabpfn_model(seed):
    import tabpfn_client as tc
    from tabpfn_client import TabPFNClassifier
    tc.set_access_token(open(os.path.expanduser('~/.tabpfn_token')).read().strip())
    return TabPFNClassifier(random_state=seed)


def fit_predict(name, Xtr, ytr, Xte, seed):
    if name == 'tabpfn':
        rng = np.random.RandomState(seed); idx = rng.choice(len(ytr), min(10000, len(ytr)), replace=False)
        imp = SimpleImputer(strategy='median').fit(Xtr[idx]); m = tabpfn_model(seed); m.fit(imp.transform(Xtr[idx]), ytr[idx])
        Xi = imp.transform(Xte); return np.concatenate([m.predict_proba(Xi[i:i + 10000])[:, 1] for i in range(0, len(Xi), 10000)])
    m = make_model(name, seed); m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]


rows = []
for name in ['growth', 'expgrowth', 'chronos', 'logreg', 'lgbm', 'lgbm_chronos', 'tabpfn']:
    per = []
    for seed in ((0,) if name == 'tabpfn' else (0, 1, 2)):
        s = np.full(len(df), np.nan); thr = np.full(len(df), np.nan)
        XX = Xc if name == 'lgbm_chronos' else X
        for tr, te in country_folds(g, 5, seed):
            tr = tr[d[tr] < np.datetime64(CUTOFF)]; te = te[d[te] >= np.datetime64(CUTOFF)]
            if name in ('growth', 'expgrowth', 'chronos'):
                sc = P0[name].values; s[te] = sc[te]; thr[te] = q_neg(sc[tr], y[tr]); continue
            fn = f'thr_cache/ts_{DS}_{name}_{seed}_{len(te)}_{int(te[0])}.npz'
            if os.path.exists(fn):
                z = np.load(fn); s[te] = z['s']; thr[te] = float(z['t']); continue
            s[te] = fit_predict(name, XX[tr], y[tr], XX[te], seed)
            inner = np.full(len(tr), np.nan)
            for itr, ite in country_folds(g[tr], 4, 100 + seed):
                inner[ite] = fit_predict(name, XX[tr][itr], y[tr][itr], XX[tr][ite], seed)
            thr[te] = q_neg(inner, y[tr]); np.savez(fn, s=s[te], t=thr[te][0])
        m = ~np.isnan(s)
        r = metrics(y[m], s[m]); r.update(episode_metrics(eps, first_alerts(P, (s > thr) & m, eps)))
        r['spec'] = float((~(s[m] > thr[m]))[y[m] == 0].mean()); r['sens'] = float((s[m] > thr[m])[y[m] == 1].mean())
        per.append(r)
        if name in ('growth', 'expgrowth', 'chronos'):
            per = per * 3; break
    if len(per) == 1: per = per * 3      # TabPFN: one partition only (API cost)
    out = {k: float(np.mean([p[k] for p in per])) for k in per[0]}; out['pr_auc_sd'] = float(np.std([p['pr_auc'] for p in per]))
    out['model'] = name; rows.append(out); print(name, {k: round(v, 3) for k, v in out.items() if isinstance(v, float)}, flush=True)
res = pd.DataFrame(rows).set_index('model'); res['episodes'] = len(eps); res['cutoff'] = str(CUTOFF.date())
res['test_rows'] = int(((d >= np.datetime64(CUTOFF))).sum())
res.to_csv(f'results_timesplit_{DS}.csv'); print(res.round(3).to_string())
