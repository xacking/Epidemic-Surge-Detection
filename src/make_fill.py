"""Compute numeric placeholder values from result CSVs and write them into word/fill.json."""
import json, pandas as pd, numpy as np
J = json.load(open('word/fill.json')); ph = J['ph']
pct = lambda v: f'{100 * v:.0f}%'
sg = lambda v: f'{v:+.3f}'.replace('-', '−')
sg2 = lambda v: ('0.00' if abs(v) < 0.005 else f'{v:+.2f}'.replace('-', '−'))
F = {d: pd.read_csv(f'results_frozen_{d}.csv', index_col=0) for d in ('flu', 'covid')}
L = {d: F[d].loc['lgbm'] for d in F}
ph.update(LGBM_DET_FLU=pct(L['flu'].detected), LGBM_DET_COV=pct(L['covid'].detected), LGBM_GE1_FLU=pct(L['flu'].ge1),
          LGBM_GE1_COV=pct(L['covid'].ge1), LGBM_GE2_FLU=pct(L['flu'].ge2), LGBM_GE2_COV=pct(L['covid'].ge2),
          LGBM_MED_FLU=f"{L['flu'].median_lead:.0f}", LGBM_MED_COV=f"{L['covid'].median_lead:.0f}", LR_DET_FLU=pct(F['flu'].loc['logreg'].detected))
sp = pd.concat([F[d].spec for d in F]); ph['SPEC_RANGE'] = f'{sp.min():.3f}–{sp.max():.3f}'
md = {k: max((F[d][k] - F[d]['oracle_' + k]).abs().max() for d in F) for k in ('detected', 'ge1', 'ge2')}
ph['ORACLE_DIFF'] = (f"{100 * md['detected']:.0f} percentage points in the proportion of episodes flagged and "
                     f"{100 * max(md['ge1'], md['ge2']):.0f} points in the proportions flagged at least one or two weeks early")
of = max(max(abs(F[d].loc['lgbm', 'otherfold_' + k] - F[d].loc['lgbm', k]) for k in ('detected', 'ge1')) for d in F)
ph['OF_DIFF'] = f'{100 * of:.0f} percentage point' + ('' if round(100 * of) == 1 else 's')
S = pd.read_csv('results_fusion2_summary.csv', index_col=0)
ci = lambda r, a, lo, hi, f: f'{f(r[a])} ({f(r[lo])} to {f(r[hi])})'
b = S.loc['base_trends']
ph.update(TR_PRCASE=f'{b.pr_case:.3f}', TR_DPR=f"{sg(b.d_pr)} (95% CI {sg(b.d_pr_lo)} to {sg(b.d_pr_hi)})",
          TR_LEAD=ci(b, 'lead_mean', 'lead_lo', 'lead_hi', sg2), TR_SAME=pct(b.same),
          TR_2X2=f"per partition, {b.paired:,.0f} of {b.episodes:,.0f} episodes were flagged by both, {b.case_only:.0f} by the case-only model alone, {b.fused_only:.0f} by the fused model alone and {b.neither:.0f} by neither",
          TR_GE1=f'{pct(b.case_ge1)} vs {pct(b.fused_ge1)}', TR_GE2=f'{pct(b.case_ge2)} vs {pct(b.fused_ge2)}')
t = S.loc['base_trendsonly']
ph.update(TO_PR=f'{t.pr_fused:.3f}', TO_LEAD_WORD='later' if t.lead_mean < 0 else 'earlier',
          TO_LEAD=f'{abs(t.lead_mean):.2f} ({abs(t.lead_hi):.2f}–{abs(t.lead_lo):.2f})')
i = S.loc['base_ili']
ph.update(ILI_DPR=f'{sg(i.d_pr)}; {sg(i.d_pr_lo)} to {sg(i.d_pr_hi)}', ILI_LEAD=ci(i, 'lead_mean', 'lead_lo', 'lead_hi', sg2), ILI_SAME=pct(i.same))
bo = S.loc['base_both']
ph['BOTH_SENTENCE'] = (f"neither stream nor their combination changed detection or timing appreciably (ΔPR-AUC {sg(bo.d_pr)}, {sg(bo.d_pr_lo)} to {sg(bo.d_pr_hi)}; "
                       f"lead {sg2(bo.lead_mean)} weeks, {sg2(bo.lead_lo)} to {sg2(bo.lead_hi)}).")
w = S.loc['window_trends']
ph.update(WIN_DPR=ci(w, 'd_pr', 'd_pr_lo', 'd_pr_hi', sg), WIN_LEAD=ci(w, 'lead_mean', 'lead_lo', 'lead_hi', sg2))
d1, d2 = S.loc['delay1_trends'], S.loc['delay2_trends']
ph.update(D1_LEAD=ci(d1, 'lead_mean', 'lead_lo', 'lead_hi', sg2), D1_DPR=ci(d1, 'd_pr', 'd_pr_lo', 'd_pr_hi', sg),
          D2_LEAD=ci(d2, 'lead_mean', 'lead_lo', 'lead_hi', sg2), D2_DPR=ci(d2, 'd_pr', 'd_pr_lo', 'd_pr_hi', sg))
ph['DELAY_SENTENCE'] = (f"Delay lowered the case-only model's PR-AUC ({b.pr_case:.3f}, {d1.pr_case:.3f} and {d2.pr_case:.3f} for delays of 0, 1 and 2 weeks) "
                        f"and roughly doubled the detection gain from Trends, but that gain stayed near 0.01 and alert timing did not change: the proportion of episodes flagged at least one week early was "
                        f"{pct(d1.case_ge1)} vs {pct(d1.fused_ge1)} (one-week delay) and {pct(d2.case_ge1)} vs {pct(d2.fused_ge1)} (two weeks).")
tt = S.loc['time_trends']
ph.update(TT_DPR=ci(tt, 'd_pr', 'd_pr_lo', 'd_pr_hi', sg), TT_LEAD=ci(tt, 'lead_mean', 'lead_lo', 'lead_hi', sg2))
T = {d: pd.read_csv(f'results_timesplit_{d}.csv', index_col=0) for d in ('flu', 'covid')}
tf, tc = T['flu'], T['covid']
ph['TS_SENTENCE'] = (f"Trained tabular models remained best (influenza: LightGBM {tf.loc['lgbm', 'pr_auc']:.3f}, TabPFN {tf.loc['tabpfn', 'pr_auc']:.3f}; "
                     f"COVID-19: LightGBM {tc.loc['lgbm', 'pr_auc']:.3f}, TabPFN {tc.loc['tabpfn', 'pr_auc']:.3f}), while for COVID-19 the simple growth-rate rule "
                     f"({tc.loc['growth', 'pr_auc']:.3f}) came within {tc.loc['lgbm', 'pr_auc'] - tc.loc['growth', 'pr_auc']:.2f} of LightGBM. Thresholds estimated before the cutoff transferred imperfectly: "
                     f"achieved specificity ranged from {pd.concat([tf.spec, tc.spec]).min():.2f} (Chronos, COVID-19) to {pd.concat([tf.spec, tc.spec]).max():.2f} (TabPFN, COVID-19), "
                     f"and this shift changed episode detection more than the choice among the top models (TabPFN flagged {pct(tc.loc['tabpfn', 'detected'])} of COVID-19 episodes against "
                     f"{pct(tc.loc['lgbm', 'detected'])} for LightGBM, despite its higher PR-AUC). LightGBM flagged {pct(tf.loc['lgbm', 'detected'])} of influenza and {pct(tc.loc['lgbm', 'detected'])} "
                     f"of COVID-19 episodes, {pct(tf.loc['lgbm', 'ge1'])} and {pct(tc.loc['lgbm', 'ge1'])} at least one week early.")
ph['DELAY_DISCUSSION'] = f"even with case data delayed by two weeks, adding real-time search data raised PR-AUC by only {d2.d_pr:.3f} and did not move alerts earlier: the mean paired lead was {sg2(d2.lead_mean)} weeks (95% CI {sg2(d2.lead_lo)} to {sg2(d2.lead_hi)})."
ph['CNN20_SENTENCE'] = J['ph'].get('CNN20_SENTENCE', '«CNN20_SENTENCE»')
json.dump(J, open('word/fill.json', 'w'), ensure_ascii=False, indent=1)
for k in ('LGBM_DET_FLU', 'LGBM_GE1_FLU', 'LGBM_GE2_FLU', 'ORACLE_DIFF', 'OF_DIFF', 'SPEC_RANGE', 'TR_LEAD', 'ILI_LEAD', 'D2_LEAD', 'TS_SENTENCE', 'TO_LEAD'):
    print(k, '=', ph[k])
