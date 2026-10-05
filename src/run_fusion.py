"""Does adding a second stream (Google Trends 'Influenza' topic; WHO FluID ILI rate) improve surge detection
or give earlier warning?  LightGBM, country-held-out, paired against case-only on identical rows/folds."""
import glob, os, sys, numpy as np, pandas as pd, lightgbm as lgb
from common import case_features, country_folds, metrics, cluster_boot_diff, episodes, sens_at_spec

TAG = sys.argv[1] if len(sys.argv) > 1 else 'H3_R1.5'
H = int(TAG.split('_')[0][1:]); R = float(TAG.split('_')[1][1:]); A = 5.0
df = pd.read_parquet(f'panel_flu_{TAG}.parquet').reset_index(drop=True)

# ---- chain Google Trends windows ----
tr = []
for c in df.iso3.unique():
    fs = sorted(glob.glob(f'trends_raw/{c}_*.csv'))
    if len(fs) < 4:
        continue
    series = None
    for f in fs:
        d = pd.read_csv(f)
        if len(d) == 0 or 'date' not in d.columns:
            series = None; break
        d = d.set_index(pd.to_datetime(d['date'])).iloc[:, 1].astype(float)
        if len(d) == 0:
            series = None; break
        if series is None:
            series = d
        else:
            ov = series.index.intersection(d.index)
            a, b = series.loc[ov].mean(), d.loc[ov].mean()
            k = a / b if b > 0 and a > 0 else np.nan
            if np.isnan(k):
                series = None; break
            series = pd.concat([series, d[d.index > series.index.max()] * k])
    if series is None or (series > 0).mean() < 0.8:
        continue
    s = pd.DataFrame({'trend': series.values}, index=series.index + pd.Timedelta(days=1))  # Sunday -> ISO Monday
    s['iso3'] = c
    tr.append(s.rename_axis('date').reset_index())
tr = pd.concat(tr)
print('Trends usable countries', tr.iso3.nunique())
lt = []
for c, g in tr.groupby('iso3'):
    g = g.set_index('date').trend.sort_index()
    l = np.log(g + 1)
    f = pd.DataFrame({'trd_diff1': l - l.shift(1), 'trd_diff2': l - l.shift(2), 'trd_diff4': l - l.shift(4),
                      'trd_rel52': l - np.log(g.rolling(52, min_periods=26).mean() + 1),
                      'trd_lag1diff1': (l - l.shift(1)).shift(1)})
    f['iso3'] = c
    lt.append(f.rename_axis('date').reset_index())
df = df.merge(pd.concat(lt), on=['iso3', 'date'], how='left')
df.to_parquet(f'panel_flu_fusion_{TAG}.parquet')

BASE = case_features(df)
TRD = [c for c in df.columns if c.startswith('trd_')]
ILI = [c for c in df.columns if c.startswith('ili_')]


def run(sub, featsets, label):
    sub = sub.reset_index(drop=True)
    y = sub.y.values.astype(int); g = sub.iso3.values
    P = sub[['iso3', 'date', 'y', 'baseline', 'x_lag0']].copy()
    for name, F in featsets.items():
        X = sub[F].values.astype(np.float32)
        cols = []
        for seed in (0, 1, 2):
            pred = np.full(len(sub), np.nan)
            for trn, te in country_folds(g, 5, seed):
                m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                                       subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=2)
                m.fit(X[trn], y[trn]); pred[te] = m.predict_proba(X[te])[:, 1]
            P[f'{name}_s{seed}'] = pred; cols.append(f'{name}_s{seed}')
        P[name] = P[cols].mean(1)
    rows = {}
    for name in featsets:
        per = [metrics(y, P[f'{name}_s{s}'].values, P[f'{name}_s{s}'].values) for s in (0, 1, 2)]
        rows[name] = {k: np.mean([d[k] for d in per]) for k in per[0]}
        rows[name]['pr_auc_sd'] = np.std([d['pr_auc'] for d in per])
    res = pd.DataFrame(rows).T
    for name in featsets:
        if name == 'case':
            continue
        for met in ('pr_auc', 'roc_auc'):
            d, lo, hi, p = cluster_boot_diff(P, name, 'case', met, B=1000)
            res.loc[name, f'd_{met}'] = d; res.loc[name, f'd_{met}_lo'] = lo; res.loc[name, f'd_{met}_hi'] = hi; res.loc[name, f'p_{met}'] = p
    # paired lead time at matched 95% specificity
    eps = episodes(P)
    by = {c: gg.sort_values('date').reset_index(drop=True) for c, gg in P.groupby('iso3')}
    fa = {}
    for name in featsets:
        _, thr = sens_at_spec(y, P[name].values, 0.95)
        out = []
        for _, e in eps.iterrows():
            gg = by[e.iso3]
            w = gg[(gg.date >= e.start - pd.Timedelta(weeks=4)) & (gg.date <= e.end)]
            hit = w[w[name] > thr]
            out.append(hit.date.iloc[0] if len(hit) else pd.NaT)
        fa[name] = pd.Series(out)
        res.loc[name, 'episodes'] = len(eps); res.loc[name, 'detected'] = fa[name].notna().mean()
    for name in featsets:
        if name == 'case':
            continue
        both = fa['case'].notna() & fa[name].notna()
        dl = (fa['case'][both] - fa[name][both]).dt.days / 7      # + = fused model earlier
        res.loc[name, 'paired_eps'] = both.sum(); res.loc[name, 'lead_mean'] = dl.mean(); res.loc[name, 'lead_median'] = dl.median()
        res.loc[name, 'frac_earlier'] = (dl > 0).mean(); res.loc[name, 'frac_later'] = (dl < 0).mean()
        # bootstrap CI over countries for mean paired lead
        e2 = eps[both.values].copy(); e2['dl'] = dl.values
        rng = np.random.RandomState(0); ks = e2.iso3.unique(); bs = []
        grp = e2.groupby('iso3').dl.apply(list).to_dict()
        for _ in range(1000):
            smp = np.concatenate([grp[k] for k in rng.choice(ks, len(ks))])
            bs.append(np.mean(smp))
        res.loc[name, 'lead_lo'], res.loc[name, 'lead_hi'] = np.percentile(bs, [2.5, 97.5])
    res['subset'] = label; res['n_rows'] = len(sub); res['n_countries'] = sub.iso3.nunique(); res['prevalence'] = y.mean()
    P.to_parquet(f'preds/fusion_{label}_{TAG}.parquet')
    return res


out = []
s1 = df.dropna(subset=['trd_diff1'])
out.append(run(s1, {'case': BASE, 'case+trends': BASE + TRD, 'trends_only': TRD + ['woy_sin', 'woy_cos', 'hemi_nh']}, 'trends'))
s2 = df[df.ili_lag0.notna()]
out.append(run(s2, {'case': BASE, 'case+ili': BASE + ILI}, 'ili'))
s3 = df.dropna(subset=['trd_diff1'])
s3 = s3[s3.ili_lag0.notna()]
if s3.iso3.nunique() >= 10:
    out.append(run(s3, {'case': BASE, 'case+trends': BASE + TRD, 'case+ili': BASE + ILI, 'case+ili+trends': BASE + ILI + TRD}, 'trends_ili'))
res = pd.concat(out)
res.to_csv(f'results_fusion_{TAG}.csv')
pd.set_option('display.width', 250)
print(res.round(3).to_string())
