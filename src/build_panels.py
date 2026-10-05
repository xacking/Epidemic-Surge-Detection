"""Build weekly country panels with leakage-safe surge labels.

Influenza (WHO FluNet): primary signal = weekly test positivity (%), weeks with >=20 specimens.
Second surveillance stream (WHO FluID): ILI rate = ILI cases / outpatients (all ages).
COVID-19 (WHO weekly): cases per 100k.

Label (computed ONLY from weeks t+1..t+H, never used as a feature):
  y_t = 1 if max_{k=1..H} x_{t+k} >= max(R * b_t, b_t + A)
  with b_t = mean(x_{t-1}, x_t)  (observed baseline).
Features use weeks <= t only.
"""
import numpy as np, pandas as pd, sys, json
D = 'data/'
H = int(sys.argv[1]) if len(sys.argv) > 1 else 3
R = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
A_FLU = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0      # percentage points
A_COV = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0     # cases per 100k per week
import os
BASEW = int(os.environ.get('BASEW', '2'))
TAG = f'H{H}_R{R}' + ('' if BASEW == 2 else f'_B{BASEW}')
NLAG = 8


def grid(df, key, date, cols):
    out = []
    for c, g in df.groupby(key):
        g = g.set_index(date)[cols].sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq='7D')
        g = g.reindex(idx)
        g[key] = c
        g.index.name = 'date'
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def add_label_features(p, x, extra=(), A=5.0, log=False):
    p = p.sort_values(['iso3', 'date']).copy()
    rows = []
    for c, g in p.groupby('iso3'):
        g = g.copy()
        s = g[x]
        b = (s + s.shift(1)) / 2 if BASEW == 2 else s.rolling(BASEW, min_periods=BASEW).mean()
        fut = pd.concat([s.shift(-k) for k in range(1, H + 1)], axis=1)
        fmax = fut.max(axis=1, skipna=False)          # require full future window observed
        thr = np.maximum(R * b, b + A)
        g['y'] = np.where(fmax.notna() & b.notna(), (fmax >= thr).astype(float), np.nan)
        g['future_max'] = fmax
        g['baseline'] = b
        tr = (lambda v: np.log1p(v)) if log else (lambda v: v)
        for L in range(NLAG):
            g[f'x_lag{L}'] = tr(s.shift(L))
        for k in (1, 2, 4):
            g[f'x_diff{k}'] = tr(s) - tr(s.shift(k))
        g['x_mean4'] = tr(s.rolling(4, min_periods=2).mean())
        g['x_max8'] = tr(s.rolling(8, min_periods=2).max())
        g['x_ly'] = tr(s.shift(52))                    # same week last year (past data)
        g['x_ly_next'] = tr(s.shift(52 - H))           # last year's value H weeks ahead (still past)
        g['n_obs8'] = s.rolling(8, min_periods=1).count()
        for e in extra:
            se = g[e]
            for L in range(4):
                g[f'{e}_lag{L}'] = se.shift(L)
            g[f'{e}_diff1'] = se - se.shift(1)
            g[f'{e}_diff2'] = se - se.shift(2)
        wk = g['date'].dt.isocalendar().week.astype(float)
        g['woy_sin'] = np.sin(2 * np.pi * wk / 52.18)
        g['woy_cos'] = np.cos(2 * np.pi * wk / 52.18)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


# ---------------- Influenza ----------------
f = pd.read_csv(D + 'Dataset_WHO_FluNet/who_flunet.csv', low_memory=False)
f['date'] = pd.to_datetime(f.ISO_WEEKSTARTDATE)
meta = f.groupby('COUNTRY_CODE').agg(region=('WHOREGION', 'first'), hemi=('HEMISPHERE', 'first'),
                                    name=('COUNTRY_AREA_TERRITORY', 'first'))
g = f.groupby(['COUNTRY_CODE', 'date'])[['INF_ALL', 'SPEC_PROCESSED_NB']].sum(min_count=1).reset_index()
g = g.rename(columns={'COUNTRY_CODE': 'iso3'})
g = g[g.date >= '2010-01-04']
g['pos'] = np.where(g.SPEC_PROCESSED_NB >= 20, 100 * g.INF_ALL.fillna(0) / g.SPEC_PROCESSED_NB, np.nan)
g['log_tests'] = np.log1p(g.SPEC_PROCESSED_NB)
fl = grid(g, 'iso3', 'date', ['pos', 'log_tests'])

# FluID ILI rate (all ages)
d = pd.read_csv(D + 'Dataset_WHO_FluID/who_fluid_epi.csv', low_memory=False,
                usecols=['COUNTRY_CODE', 'ISO_WEEKSTARTDATE', 'AGEGROUP_CODE', 'ILI_CASE', 'ILI_OUTPATIENTS', 'SARI_CASE'])
d = d[d.AGEGROUP_CODE.isin(['ALL', 'All'])]
d['date'] = pd.to_datetime(d.ISO_WEEKSTARTDATE)
d = d.groupby(['COUNTRY_CODE', 'date'])[['ILI_CASE', 'ILI_OUTPATIENTS', 'SARI_CASE']].sum(min_count=1).reset_index()
d['ili'] = np.where(d.ILI_OUTPATIENTS >= 50, 100 * d.ILI_CASE / d.ILI_OUTPATIENTS, np.nan)
d = d.rename(columns={'COUNTRY_CODE': 'iso3'})[['iso3', 'date', 'ili']]
fl = fl.merge(d, on=['iso3', 'date'], how='left')

fl = add_label_features(fl, 'pos', extra=('log_tests', 'ili'), A=A_FLU)
fl = fl.merge(meta, left_on='iso3', right_index=True, how='left')
fl['hemi_nh'] = (fl.hemi == 'NH').astype(float)
fl = fl[fl.y.notna() & fl.x_lag0.notna()]
# keep countries with enough labelled weeks and at least 5 surge events
cnt = fl.groupby('iso3').agg(n=('y', 'size'), pos=('y', 'sum'))
keep = cnt[(cnt.n >= 104) & (cnt.pos >= 5)].index
fl = fl[fl.iso3.isin(keep)]
fl.to_parquet(f'panel_flu_{TAG}.parquet')
print('FLU', fl.shape, 'countries', fl.iso3.nunique(), 'prevalence', round(fl.y.mean(), 3),
      'with ILI', fl.groupby('iso3').ili_lag0.count().gt(52).sum())

# ---------------- COVID-19 ----------------
c = pd.read_csv(D + 'Datasets_covid_mpox_lookups/who_covid19_weekly.csv', keep_default_na=False, na_values=[''])
lk = pd.read_csv(D + 'Datasets_covid_mpox_lookups/jhu_lookup_population.csv')
lk = lk[lk.Province_State.isna()][['iso2', 'iso3', 'Population']].dropna()
c = c.merge(lk, left_on='Country_code', right_on='iso2', how='inner')
c['date'] = pd.to_datetime(c.Date_reported)
c = c[(c.date >= '2020-03-01') & (c.date <= '2023-06-30') & (c.Population >= 500000)]
c['inc'] = 1e5 * c.New_cases.clip(lower=0) / c.Population
cv = grid(c, 'iso3', 'date', ['inc'])
cv = add_label_features(cv, 'inc', A=A_COV, log=True)
cv = cv[cv.y.notna() & cv.x_lag0.notna()]
cnt = cv.groupby('iso3').agg(n=('y', 'size'), pos=('y', 'sum'))
cv = cv[cv.iso3.isin(cnt[(cnt.n >= 80) & (cnt.pos >= 5)].index)]
cv.to_parquet(f'panel_covid_{TAG}.parquet')
print('COVID', cv.shape, 'countries', cv.iso3.nunique(), 'prevalence', round(cv.y.mean(), 3))
