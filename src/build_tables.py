"""Write markdown tables for the revised manuscript into word/tables/*.md."""
import os, numpy as np, pandas as pd
os.makedirs('word/tables', exist_ok=True)
NM = {'lgbm_chronos': 'LightGBM + Chronos feature', 'tabpfn': 'TabPFN v2', 'lgbm': 'LightGBM', 'rf': 'Random forest',
      'cnnlstm_rf': 'CNN-LSTM-RF hybrid', 'tft': 'Temporal Fusion Transformer', 'chronos': 'Chronos-Bolt (zero-shot)',
      'logreg': 'Logistic regression', 'expgrowth': 'Exponential-growth extrapolation', 'growth': 'Growth-rate rule'}
f3 = lambda v: f'{v:.3f}'
pct = lambda v: f'{100 * v:.0f}%'
sg = lambda v: f'{v:+.3f}'.replace('-', '−')
sg2 = lambda v: ('0.00' if abs(v) < 0.005 else f'{v:+.2f}'.replace('-', '−'))


def md(df):
    h = '| ' + ' | '.join(df.columns) + ' |\n| ' + ' | '.join(['---'] * df.shape[1]) + ' |\n'
    return h + '\n'.join('| ' + ' | '.join(str(x) for x in r) + ' |' for r in df.values) + '\n'


def lead(v):
    return f'{v:.1f}'.rstrip('0').rstrip('.') if abs(v - round(v)) > 1e-9 else f'{v:.0f}'


# ---- Table 1
R = {d: pd.read_csv(f'results_{d}_H3_R1.5.csv', index_col=0) for d in ('flu', 'covid')}
F = {d: pd.read_csv(f'results_frozen_{d}.csv', index_col=0) for d in ('flu', 'covid')}
order = R['flu'].sort_values('pr_avg', ascending=False).index
rows = []
for m in ['lgbm_chronos', 'tabpfn', 'lgbm', 'rf', 'cnnlstm_rf', 'tft', 'chronos', 'logreg', 'expgrowth', 'growth']:
    pair = lambda col, fmt, src: f"{fmt(src['flu'].loc[m, col])} / {fmt(src['covid'].loc[m, col])}"
    rows.append([NM[m], pair('roc_avg', f3, R), pair('sens', f3, F), pair('detected', pct, F), pair('median_lead', lead, F),
                 pair('ge1', pct, F), pair('ge2', pct, F)])
T1 = pd.DataFrame(rows, columns=['Model', 'ROC-AUC', 'Sensitivity at 95% specificity', 'Episodes flagged', 'Median lead, weeks', 'Flagged ≥ 1 week early', 'Flagged ≥ 2 weeks early'])
open('word/tables/TABLE1.md', 'w').write(md(T1))

# ---- Table 2
S = pd.read_csv('results_fusion2_summary.csv', index_col=0)
lab = {'base_trends': 'Google Trends', 'base_trendsonly': 'Google Trends instead of cases', 'base_ili': 'Syndromic ILI',
       'base_both': 'Trends + ILI', 'window_trends': 'Google Trends, single-window features',
       'delay1_trends': 'Google Trends, case data delayed 1 week', 'delay2_trends': 'Google Trends, case data delayed 2 weeks',
       'time_trends': 'Google Trends, countries and time held out'}
rows = []
for k, name in lab.items():
    if k not in S.index:
        continue
    r = S.loc[k]
    rows.append([name, int(r.countries), f3(r.pr_case), f'{sg(r.d_pr)} ({sg(r.d_pr_lo)}, {sg(r.d_pr_hi)})',
                 f'{r.paired:.0f} / {r.case_only:.0f} / {r.fused_only:.0f} / {r.neither:.0f}',
                 f'{sg2(r.lead_mean)} ({sg2(r.lead_lo)}, {sg2(r.lead_hi)})', f'{pct(r.earlier)} / {pct(r.same)} / {pct(r.later)}',
                 f'{pct(r.case_ge1)} → {pct(r.fused_ge1)}'])
T2 = pd.DataFrame(rows, columns=['Added stream', 'Countries', 'Case-only PR-AUC', 'ΔPR-AUC (95% CI)',
                                 'Episodes: both / case only / fused only / neither', 'Paired lead, weeks (95% CI)',
                                 'Earlier / same / later', 'Flagged ≥ 1 week early'])
open('word/tables/TABLE2.md', 'w').write(md(T2))

# ---- Table 3
rows = []
for d, dn in (('flu', 'Influenza'), ('covid', 'COVID-19')):
    fn = f'results_timesplit_{d}.csv'
    if not os.path.exists(fn):
        continue
    t = pd.read_csv(fn, index_col=0)
    for m in t.sort_values('pr_auc', ascending=False).index:
        if m in t.index:
            r = t.loc[m]
            rows.append([dn, NM[m], f3(r.pr_auc), f3(r.roc_auc), f3(r.spec), pct(r.detected), lead(r.median_lead), pct(r.ge1), pct(r.ge2)])
T3 = pd.DataFrame(rows, columns=['Disease', 'Model', 'PR-AUC', 'ROC-AUC', 'Achieved specificity', 'Episodes flagged',
                                 'Median lead, weeks', '≥ 1 week early', '≥ 2 weeks early'])
open('word/tables/TABLE3.md', 'w').write(md(T3))

# ---- Table 4 (sensitivity, incl. 4-week baseline)
s = pd.read_csv('results_sensitivity.csv')
rows = [[{'flu': 'Influenza', 'covid': 'COVID-19'}[r.disease], (r.definition.replace('H=', '*H* = ').replace(' wk', ' weeks').replace('(R=2, larger floor)', '(*R* = 2, *A* = ' + ('10 points)' if r.disease == 'flu' else '20 per 100,000)'))),
         r.countries, f3(r.prevalence), f3(r.growth), f3(r.expgrowth), f3(r.logreg), f3(r.lgbm)] for r in s.itertuples()]
for d, dn in (('flu', 'Influenza'), ('covid', 'COVID-19')):
    fn = f'results_{d}_H3_R1.5_B4.csv'
    if os.path.exists(fn):
        b = pd.read_csv(fn, index_col=0); p = pd.read_parquet(f'panel_{d}_H3_R1.5_B4.parquet', columns=['iso3', 'y'])
        col = 'pr_avg' if 'pr_avg' in b else 'pr_auc'
        rows.append([dn, '*H* = 3, four-week baseline', p.iso3.nunique(), f3(p.y.mean())] + [f3(b.loc[m, col]) for m in ('growth', 'expgrowth', 'logreg', 'lgbm')])
T4 = pd.DataFrame(rows, columns=['Disease', 'Definition', 'Countries', 'Base rate', 'Growth-rate rule', 'Exponential extrapolation',
                                 'Logistic regression', 'LightGBM'])
T4 = T4.sort_values('Disease', key=lambda c: c.map({'Influenza': 0, 'COVID-19': 1}), kind='stable')
open('word/tables/TABLE4.md', 'w').write(md(T4))

# ---- S2 representativeness
r = pd.read_csv('results_subset_representativeness.csv')
r.columns = ['Subset', 'Countries', 'Country-weeks', 'Base rate', 'Median weeks per country', 'Median weekly specimens',
             'Africa', 'Americas', 'E. Medit.', 'Europe', 'SE Asia', 'W. Pacific']
r['Country-weeks'] = r['Country-weeks'].map('{:,}'.format); r['Base rate'] = r['Base rate'].map('{:.3f}'.format)
open('word/tables/TABLES2.md', 'w').write(md(r))

# ---- S3 simulation parameters
p = pd.read_csv('results_sim_parameters.csv').fillna('')
p.columns = ['Parameter', 'Value', 'Notes']
open('word/tables/TABLES3.md', 'w').write(md(p))

# ---- S4 partitions
v = pd.read_csv('results_partition_variability.csv')
v = pd.DataFrame([[{'flu': 'Influenza', 'covid': 'COVID-19'}[x.disease], NM[x.model], f3(x.p0), f3(x.p1), f3(x.p2), f3(x.range)] for x in v.itertuples()],
                 columns=['Disease', 'Model', 'Partition 0', 'Partition 1', 'Partition 2', 'Range'])
open('word/tables/TABLES4.md', 'w').write(md(v))

# ---- S5 regions
g = pd.read_csv('results_region_ci.csv')
g = pd.DataFrame([[{'flu': 'Influenza', 'covid': 'COVID-19'}[x.disease], x.region, x.countries, f'{x.country_weeks:,}', f3(x.base_rate),
                   f'{x.pr_auc:.3f} ({x.lo:.3f}–{x.hi:.3f})'] for x in g.itertuples()],
                 columns=['Disease', 'WHO region', 'Countries', 'Country-weeks', 'Base rate', 'PR-AUC (95% CI)'])
open('word/tables/TABLES5.md', 'w').write(md(g))
print('tables ok'); print(open('word/tables/TABLE1.md').read()); print(open('word/tables/TABLE2.md').read())
