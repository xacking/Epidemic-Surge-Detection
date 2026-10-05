"""Regional PR-AUC of LightGBM (partition-averaged held-out scores) with country-bootstrap 95% CIs; flu and COVID-19."""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
names = {'AFRO': 'Africa', 'AMRO': 'Americas', 'EMRO': 'E. Mediterranean', 'EURO': 'Europe', 'SEARO': 'SE Asia', 'WPRO': 'W. Pacific', 'AFR': 'Africa', 'AMR': 'Americas', 'EMR': 'E. Mediterranean', 'EUR': 'Europe', 'SEAR': 'SE Asia', 'WPR': 'W. Pacific'}
meta = pd.read_parquet('panel_flu_H3_R1.5.parquet', columns=['iso3', 'region']).drop_duplicates('iso3').set_index('iso3').region
f = pd.read_csv('data/Dataset_WHO_FluNet/who_flunet.csv', low_memory=False, usecols=['COUNTRY_CODE', 'WHOREGION']).drop_duplicates('COUNTRY_CODE').set_index('COUNTRY_CODE').WHOREGION
c = pd.read_csv('data/Datasets_covid_mpox_lookups/who_covid19_weekly.csv', keep_default_na=False, na_values=[''])
lk = pd.read_csv('data/Datasets_covid_mpox_lookups/jhu_lookup_population.csv'); lk = lk[lk.Province_State.isna()][['iso2', 'iso3']].dropna()
cr = c.drop_duplicates('Country_code').merge(lk, left_on='Country_code', right_on='iso2').set_index('iso3').WHO_region
rows = []
for ds in ('flu', 'covid'):
    P = pd.read_parquet(f'preds/{ds}_H3_R1.5.parquet'); P['s'] = P[[c for c in P.columns if c.startswith('lgbm_s')]].mean(1)
    P['region'] = (P.iso3.map(f) if ds == 'flu' else P.iso3.map(cr)).map(names)
    for reg, g in P.groupby('region'):
        cs = g.iso3.unique(); by = {c: gg for c, gg in g.groupby('iso3')}; rng = np.random.default_rng(0); bs = []
        for _ in range(500):
            s = pd.concat([by[c] for c in rng.choice(cs, len(cs))])
            if s.y.nunique() == 2: bs.append(ap(s.y, s.s))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(disease=ds, region=reg, countries=len(cs), country_weeks=len(g), base_rate=g.y.mean(),
                         pr_auc=ap(g.y, g.s), lo=lo, hi=hi, median_weeks_per_country=g.groupby('iso3').size().median()))
R = pd.DataFrame(rows); R.to_csv('results_region_ci.csv', index=False); print(R.round(3).to_string())
