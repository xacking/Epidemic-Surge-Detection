"""Full weekly signal per country (all observed weeks, labelled or not), built exactly as in build_panels.py.
Used to locate the crossing week at onset+1..onset+H. Cached to signal_{flu,covid}.parquet."""
import os, numpy as np, pandas as pd
D = 'data/'


def grid(df, key, date, col):
    out = []
    for c, g in df.groupby(key):
        g = g.set_index(date)[[col]].sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq='7D')); g[key] = c; g.index.name = 'date'
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def load_signal(ds):
    fn = f'signal_{ds}.parquet'
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    if ds == 'flu':
        f = pd.read_csv(D + 'Dataset_WHO_FluNet/who_flunet.csv', low_memory=False, usecols=['COUNTRY_CODE', 'ISO_WEEKSTARTDATE', 'INF_ALL', 'SPEC_PROCESSED_NB'])
        f['date'] = pd.to_datetime(f.ISO_WEEKSTARTDATE)
        g = f.groupby(['COUNTRY_CODE', 'date'])[['INF_ALL', 'SPEC_PROCESSED_NB']].sum(min_count=1).reset_index().rename(columns={'COUNTRY_CODE': 'iso3'})
        g = g[g.date >= '2010-01-04']
        g['x'] = np.where(g.SPEC_PROCESSED_NB >= 20, 100 * g.INF_ALL.fillna(0) / g.SPEC_PROCESSED_NB, np.nan)
    else:
        c = pd.read_csv(D + 'Datasets_covid_mpox_lookups/who_covid19_weekly.csv', keep_default_na=False, na_values=[''])
        lk = pd.read_csv(D + 'Datasets_covid_mpox_lookups/jhu_lookup_population.csv')
        lk = lk[lk.Province_State.isna()][['iso2', 'iso3', 'Population']].dropna()
        c = c.merge(lk, left_on='Country_code', right_on='iso2', how='inner'); c['date'] = pd.to_datetime(c.Date_reported)
        c = c[(c.date >= '2020-03-01') & (c.date <= '2023-06-30') & (c.Population >= 500000)]
        c['x'] = 1e5 * c.New_cases.clip(lower=0) / c.Population; g = c
    s = grid(g, 'iso3', 'date', 'x')[['iso3', 'date', 'x']]
    s.to_parquet(fn); return s


if __name__ == '__main__':
    for ds in ('flu', 'covid'):
        s = load_signal(ds); P = pd.read_parquet(f'panel_{ds}_H3_R1.5.parquet', columns=['iso3', 'date', 'x_lag0'])
        m = P.merge(s, on=['iso3', 'date']); raw = np.expm1(m.x_lag0) if ds == 'covid' else m.x_lag0
        print(ds, len(s), 'matched', len(m), 'of', len(P), 'max abs diff', float(np.nanmax(np.abs(raw - m.x))))
