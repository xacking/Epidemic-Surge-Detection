"""Second-stream analysis, revised (reviewer points 1, 3, 5, 7 / R2-1, R2-3).
usage: python run_fusion2.py VARIANT [STREAM]
VARIANT: base | delay1 | delay2 | window | time
  base   : as in the paper, thresholds now leakage-free (nested inner CV for both models)
  delayD : case-based features available only up to week t-D (simulated reporting delay); the second stream
           remains real-time; labels unchanged
  window : Google Trends features computed inside single downloaded windows (no cross-window chaining, no
           level-vs-52-week feature, first 4 weeks of each window excluded)
  time   : country AND time held out: train on training countries before CUTOFF, test on held-out countries
           from CUTOFF onward
STREAM: trends (default) | ili
Paired lead, 2x2 detection table and >=1 / >=2-week-early proportions of ALL episodes."""
import sys, glob, numpy as np, pandas as pd, lightgbm as lgb
from common import case_features, country_folds, metrics, cluster_boot_diff, episodes
from frozen import q_neg, make_model, crossing_weeks, first_alerts
VAR = sys.argv[1]; STREAM = sys.argv[2] if len(sys.argv) > 2 else 'trends'
TAG = 'H3_R1.5'; R = 1.5; A = 5.0; CUTOFF = pd.Timestamp('2020-01-01')
NESTED = VAR in ('base', 'time')   # strict: thresholds from training countries (and, for 'time', pre-cutoff weeks) only
df = pd.read_parquet(f'panel_flu_fusion_{TAG}.parquet').reset_index(drop=True)
BASE = case_features(df)
CASE_COLS = [c for c in BASE if c.startswith('x_') or c.startswith('log_tests_') or c == 'n_obs8']

if VAR.startswith('delay'):
    D = int(VAR[-1])
    for c in CASE_COLS:                         # value computed at t-D for the same country (weekly grid)
        s = df.set_index(['iso3', 'date'])[c]
        shifted = df[['iso3', 'date']].assign(date=df.date - pd.Timedelta(weeks=D))
        df[c] = shifted.merge(s.rename('v').reset_index(), on=['iso3', 'date'], how='left').v.values

if VAR == 'window':
    rows = []
    for f in sorted(glob.glob('trends_raw/*.csv')):
        d = pd.read_csv(f)
        if len(d) == 0 or 'date' not in d.columns:
            continue
        c = f.split('/')[-1].split('_')[0]
        s = d.set_index(pd.to_datetime(d['date'])).iloc[:, 1].astype(float)
        s.index = s.index + pd.Timedelta(days=1)
        l = np.log(s + 1)
        w = pd.DataFrame({'trd_diff1': l - l.shift(1), 'trd_diff2': l - l.shift(2), 'trd_diff4': l - l.shift(4),
                          'trd_lag1diff1': (l - l.shift(1)).shift(1)}).iloc[4:]          # drop window-start weeks
        w['iso3'] = c; w['wstart'] = s.index.min()
        rows.append(w.rename_axis('date').reset_index())
    W = pd.concat(rows).sort_values('wstart').groupby(['iso3', 'date']).last().reset_index().drop(columns='wstart')
    keep = df.dropna(subset=['trd_diff1']).iso3.unique()          # same country set as the main analysis
    df = df.drop(columns=[c for c in df.columns if c.startswith('trd_')]).merge(W[W.iso3.isin(keep)], on=['iso3', 'date'], how='left')

TRD = [c for c in df.columns if c.startswith('trd_')]; ILI = [c for c in df.columns if c.startswith('ili_')]
if STREAM in ('trends', 'trendsonly'):
    sub = df.dropna(subset=['trd_diff1'])
elif STREAM == 'ili':
    sub = df[df.ili_lag0.notna()]
else:                                   # 'both': countries-weeks with Trends AND ILI
    sub = df.dropna(subset=['trd_diff1']); sub = sub[sub.ili_lag0.notna()]
sub = sub.dropna(subset=['x_lag0']).reset_index(drop=True)
y = sub.y.values.astype(int); g = sub.iso3.values
FS = {'case': BASE, 'fused': {'trends': BASE + TRD, 'ili': BASE + ILI, 'both': BASE + ILI + TRD,
                              'trendsonly': TRD + ['woy_sin', 'woy_cos', 'hemi_nh']}[STREAM]}
P = sub[['iso3', 'date', 'y', 'baseline', 'x_lag0']].copy()
if VAR.startswith('delay'):       # crossing/lead use the TRUE observed signal at t, not the delayed one
    P['x_lag0'] = pd.read_parquet(f'panel_flu_fusion_{TAG}.parquet').set_index(['iso3', 'date']).x_lag0.reindex(
        pd.MultiIndex.from_frame(sub[['iso3', 'date']])).values
eps_all = crossing_weeks(P, episodes(P), R, A, False)

res = {}; per_seed = []; leads = []
for seed in (0, 1, 2):
    alerts = {}; scores = {}
    folds = country_folds(g, 5, seed)
    test_mask = np.zeros(len(sub), bool)
    for name, F in FS.items():
        X = sub[F].values.astype(np.float32)
        pred = np.full(len(sub), np.nan); thr = np.full(len(sub), np.nan)
        for tr, te in folds:
            if VAR == 'time':
                tr = tr[sub.date.values[tr] < CUTOFF]; te = te[sub.date.values[te] >= CUTOFF]
            m = make_model(name, seed); m.fit(X[tr], y[tr]); pred[te] = m.predict_proba(X[te])[:, 1]
            if NESTED:
                inner = np.full(len(tr), np.nan)
                for itr, ite in country_folds(g[tr], 4, 100 + seed):
                    mi = make_model(name, seed); mi.fit(X[tr][itr], y[tr][itr]); inner[ite] = mi.predict_proba(X[tr][ite])[:, 1]
                thr[te] = q_neg(inner, y[tr])
            test_mask[te] = True
        if not NESTED:   # 'otherfold' thresholds: training countries' outer out-of-fold scores
            for tr, te in folds:
                if VAR == 'time':
                    te = te[sub.date.values[te] >= CUTOFF]; trs = tr[sub.date.values[tr] >= CUTOFF]
                else:
                    trs = tr
                thr[te] = q_neg(pred[trs], y[trs])
        scores[name] = pred; alerts[name] = pred > thr
        P[f'{name}_s{seed}'] = pred
    eps = eps_all if VAR != 'time' else eps_all[eps_all.start >= CUTOFF].reset_index(drop=True)
    fa = {n: first_alerts(P, alerts[n] & ~np.isnan(scores[n]), eps) for n in FS}
    L = {n: (eps.cross - fa[n]).dt.days / 7 for n in FS}
    dc, df_ = fa['case'].notna(), fa['fused'].notna()
    both = dc & df_
    dl = (fa['case'][both] - fa['fused'][both]).dt.days / 7
    msk = ~np.isnan(scores['case'])
    r = dict(seed=seed, episodes=len(eps), both=int(both.sum()), case_only=int((dc & ~df_).sum()),
             fused_only=int((~dc & df_).sum()), neither=int((~dc & ~df_).sum()),
             lead_mean=float(dl.mean()), lead_median=float(dl.median()), earlier=float((dl > 0).mean()), later=float((dl < 0).mean()),
             case_ge1=float((L['case'] >= 1).mean()), fused_ge1=float((L['fused'] >= 1).mean()),
             case_ge2=float((L['case'] >= 2).mean()), fused_ge2=float((L['fused'] >= 2).mean()),
             case_det=float(dc.mean()), fused_det=float(df_.mean()),
             pr_case=metrics(y[msk], scores['case'][msk])['pr_auc'], pr_fused=metrics(y[msk], scores['fused'][msk])['pr_auc'],
             spec_case=float((~alerts['case'][msk][y[msk] == 0]).mean()), spec_fused=float((~alerts['fused'][msk][y[msk] == 0]).mean()))
    leads.append(pd.DataFrame({'seed': seed, 'iso3': eps.iso3[both].values, 'start': eps.start[both].values, 'lead': dl.values}))
    per_seed.append(r); print(VAR, STREAM, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
out = pd.DataFrame(per_seed)
summ = out.mean(numeric_only=True).to_frame('mean').join(out.std(numeric_only=True).to_frame('sd'))
# paired PR-AUC difference with country bootstrap on partition-averaged scores
for n in FS:
    P[n] = P[[f'{n}_s{s}' for s in (0, 1, 2)]].mean(1)
Pm = P.dropna(subset=['case', 'fused'])
d, lo, hi, p = cluster_boot_diff(Pm, 'fused', 'case', 'pr_auc', B=1000)
summ.loc['d_pr_auc_avg', 'mean'] = d; summ.loc['d_pr_auc_lo', 'mean'] = lo; summ.loc['d_pr_auc_hi', 'mean'] = hi
summ.loc['n_countries', 'mean'] = sub.iso3.nunique(); summ.loc['n_rows_test', 'mean'] = len(Pm)
pd.concat(leads).to_csv(f'results_fusion2_{VAR}_{STREAM}_leads.csv', index=False)
summ.to_csv(f'results_fusion2_{VAR}_{STREAM}.csv'); out.to_csv(f'results_fusion2_{VAR}_{STREAM}_perseed.csv', index=False)
print(summ.round(3).to_string())
