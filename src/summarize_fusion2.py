"""Collect revised second-stream results; paired-lead CI by country cluster bootstrap (episodes pooled over partitions)."""
import glob, numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob('results_fusion2_*_perseed.csv')):
    key = f.replace('results_fusion2_', '').replace('_perseed.csv', '')
    ps = pd.read_csv(f); sm = pd.read_csv(f.replace('_perseed', ''), index_col=0)
    L = pd.read_csv(f.replace('_perseed', '_leads'))
    by = {c: g.lead.values for c, g in L.groupby('iso3')}; ks = list(by)
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        smp = np.concatenate([by[k] for k in rng.choice(ks, len(ks))]); bs.append(smp.mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    m = ps.mean(numeric_only=True)
    r = dict(analysis=key, countries=int(sm.loc['n_countries', 'mean']), episodes=m.episodes, paired=m.both,
             case_only=m.case_only, fused_only=m.fused_only, neither=m.neither,
             pr_case=m.pr_case, pr_fused=m.pr_fused, d_pr=sm.loc['d_pr_auc_avg', 'mean'], d_pr_lo=sm.loc['d_pr_auc_lo', 'mean'],
             d_pr_hi=sm.loc['d_pr_auc_hi', 'mean'], lead_mean=L.lead.mean(), lead_lo=lo, lead_hi=hi,
             earlier=(L.lead > 0).mean(), same=(L.lead == 0).mean(), later=(L.lead < 0).mean(),
             case_det=m.case_det, fused_det=m.fused_det, case_ge1=m.case_ge1, fused_ge1=m.fused_ge1,
             case_ge2=m.case_ge2, fused_ge2=m.fused_ge2, spec_case=m.spec_case, spec_fused=m.spec_fused)
    rows.append(r)
R = pd.DataFrame(rows).set_index('analysis'); R.to_csv('results_fusion2_summary.csv')
pd.set_option('display.width', 300); print(R.round(3).T.to_string())
