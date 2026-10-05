"""Fetch weekly Google Trends for the 'Influenza' topic (/m/0cycc) per country.
Overlapping <5-year windows chained by the ratio of overlap means.
Note: Trends normalises each window to its own max (uses within-window future);
downstream features use only log-ratios (scale-free) so the normalisation does not leak.
"""
import pandas as pd, numpy as np, time, os, json, sys
from pytrends.request import TrendReq

TOPIC = '/m/0cycc'
WINDOWS = [('2010-01-01', '2014-12-31'), ('2014-01-01', '2018-12-31'), ('2018-01-01', '2022-12-31'),
           ('2022-01-01', '2026-09-30')]
os.makedirs('trends_raw', exist_ok=True)
panel = pd.read_parquet('panel_flu_H3_R1.5.parquet')
iso3 = sorted(panel.iso3.unique())
lk = pd.read_csv('data/Datasets_covid_mpox_lookups/jhu_lookup_population.csv')
m = lk[lk.Province_State.isna()].dropna(subset=['iso2', 'iso3']).drop_duplicates('iso3').set_index('iso3').iso2
p = TrendReq(hl='en-US', tz=0, timeout=(10, 30))

for c in iso3:
    if c not in m.index:
        print('no iso2', c); continue
    geo = m[c]
    for a, b in WINDOWS:
        fn = f'trends_raw/{c}_{a[:4]}.csv'
        if os.path.exists(fn):
            continue
        for attempt in range(5):
            try:
                p.build_payload([TOPIC], timeframe=f'{a} {b}', geo=geo)
                d = p.interest_over_time()
                (d[[TOPIC]] if len(d) else pd.DataFrame(columns=[TOPIC])).to_csv(fn)
                break
            except Exception as e:
                w = 30 * (attempt + 1)
                print(c, a, 'err', str(e)[:80], 'sleep', w, flush=True)
                time.sleep(w)
        time.sleep(4)
    print('done', c, flush=True)
print('ALL DONE')
