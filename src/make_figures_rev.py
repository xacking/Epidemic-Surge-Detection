"""Revised figures: Fig 1 (alerts with thresholds fixed on training countries), Fig 3 (leakage-free second-stream
analysis incl. reporting-delay test), Supplementary Fig S3 (inclusion flow)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from frozen import row_thresholds
exec(open('make_figures.py').read().split('# ---------------- Figure 1')[0])   # shared style, save()

# ---------------- Figure 1 ----------------
P = pd.read_parquet('preds/flu_H3_R1.5.parquet'); P['date'] = pd.to_datetime(P.date)
thr = row_thresholds('otherfold', P.lgbm_s0.values, P.y.values.astype(int), P.iso3.values, 0)
P['alert'] = P.lgbm_s0.values > thr
panel = pd.read_parquet('panel_flu_H3_R1.5.parquet')
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={'width_ratios': [2.3, 1]})
ax = axes[0]
g = P[(P.iso3 == 'FRA') & (P.date >= '2015-07-01') & (P.date < '2020-01-01')].sort_values('date')
gl = g.set_index('date').x_lag0.reindex(pd.date_range(g.date.min(), g.date.max(), freq='7D'))
ax.plot(gl.index, gl.values, color=CAT[0], lw=1.6, label='Influenza test positivity (%; gaps = no data)')
for d, yy in zip(g.date, g.y):
    if yy == 1:
        ax.axvspan(d - pd.Timedelta(days=3), d + pd.Timedelta(days=4), color='#eda100', alpha=0.22, lw=0)
al = g[g.alert]
ax.scatter(al.date, np.full(len(al), -4), marker='|', s=60, color=CAT[7], label='Alerts (95% spec.; threshold set on training countries)')
ax.text(g.date.min(), -6.3, 'Alerts', fontsize=7, color=CAT[7])
ax.fill_between([], [], color='#eda100', alpha=0.3, label='Surge within next 3 weeks (label)')
ax.set_ylabel('Positivity (%)'); ax.set_ylim(-7, None); ax.grid(axis='y', color=GRID, lw=0.6)
ax.set_title('a  Task illustration: France, 2015–2019 (held-out LightGBM, one partition)', loc='left', fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=7.5, loc='upper left', ncol=1)
ax = axes[1]
reg = panel.groupby('region').iso3.nunique().sort_values()
names = {'AFR': 'Africa', 'AMR': 'Americas', 'EMR': 'E. Medit.', 'EUR': 'Europe', 'SEAR': 'SE Asia', 'WPR': 'W. Pacific'}
ax.barh([names.get(r, r) for r in reg.index], reg.values, color=CAT[0], height=0.6)
for i, v in enumerate(reg.values):
    ax.text(v + 0.5, i, str(v), va='center', fontsize=8, color=INK2)
ax.set_xlabel('Countries (influenza panel)')
ax.set_title(f'b  Coverage: {panel.iso3.nunique()} countries, {len(panel):,} country-weeks', loc='left', fontsize=10, color=INK)
save(fig, 'fig1_task_coverage')

# ---------------- Figure 3 ----------------
S = pd.read_csv('results_fusion2_summary.csv', index_col=0)
fig, axes = plt.subplots(1, 4, figsize=(17, 3.9), gridspec_kw={'width_ratios': [1.05, 1, 1, 1], 'wspace': 0.38})
ax = axes[0]
b, to = S.loc['base_trends'], S.loc['base_trendsonly']
labs = ['Case data\nonly', 'Case +\nGoogle Trends', 'Google Trends\nonly']
vals = [b.pr_case, b.pr_fused, to.pr_fused]
ax.bar(labs, vals, color=[CAT[0], CAT[1], CAT[3]], width=0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8, color=INK2)
ax.set_ylabel('PR-AUC (held-out countries)'); ax.set_ylim(0, 0.75); ax.tick_params(axis='x', labelsize=7.5)
ax.set_title(f"a  Detection, {int(b.countries)} countries\nwith usable search data", loc='left', fontsize=9, color=INK)
for ax, (key, ttl, c) in zip(axes[1:3], [('base_trends', 'b  Adding Google Trends', CAT[1]), ('base_ili', 'c  Adding syndromic ILI', CAT[2])]):
    L = pd.read_csv(f'results_fusion2_{key}_leads.csv'); d = L.lead.values; r = S.loc[key]
    w = np.full(len(d), 1 / L.seed.nunique())                   # pooled over partitions, counts per partition
    ax.hist(d, bins=np.arange(-6.5, 7.5, 1), weights=w, color=c, rwidth=0.85)
    ax.axvline(0, color=INK, lw=1)
    ax.set_title(f'{ttl}\nmean {r.lead_mean:+.2f} wk (95% CI {r.lead_lo:+.2f} to {r.lead_hi:+.2f})\n'
                 f'{r.paired:.0f} paired episodes per partition', loc='left', fontsize=9, color=INK)
    ax.set_xlabel('Extra warning from added stream (weeks)\n< 0 later   |   > 0 earlier')
    ax.set_ylabel('Surge episodes'); ax.grid(axis='y', color=GRID, lw=0.6)
    ax.text(0.98, 0.95, f'{r.earlier:.0%} earlier\n{r.same:.0%} same week\n{r.later:.0%} later',
            transform=ax.transAxes, ha='right', va='top', fontsize=8, color=INK2)
ax = axes[3]
keys = [('base_trends', '0'), ('delay1_trends', '1'), ('delay2_trends', '2')]
for i, (k, lab) in enumerate(keys):
    r = S.loc[k]
    ax.plot([i, i], [r.lead_lo, r.lead_hi], color=CAT[1], lw=2); ax.scatter(i, r.lead_mean, color=CAT[1], s=40, zorder=3)
    ax.text(i + 0.08, r.lead_mean, f'{(r.lead_mean if abs(r.lead_mean) >= 0.005 else 0):+.2f}'.replace('+0.00', '0.00'), fontsize=8, color=INK2, va='center')
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels([f'{k[1]} wk' for k in keys]); ax.set_xlim(-0.5, 2.6)
ax.set_xlabel('Simulated delay of case data\n(Google Trends kept real-time)'); ax.set_ylabel('Mean extra warning (weeks, 95% CI)')
ax.grid(axis='y', color=GRID, lw=0.6)
ax.set_title('d  Adding Trends when case\nreporting is delayed', loc='left', fontsize=9, color=INK)
save(fig, 'fig3_digital_signals')

# ---------------- Supplementary Figure S3: inclusion flow ----------------
fig, ax = plt.subplots(figsize=(10, 4.4)); ax.axis('off'); ax.set_xlim(0, 9.4); ax.set_ylim(1.1, 9.7)


def box(x, y, t, w=2.8, h=1.1, c=SURF):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.05', fc=c, ec=INK2, lw=0.8))
    ax.text(x, y, t, ha='center', va='center', fontsize=8, color=INK)


def arrow(x0, y0, x1, y1):
    ax.annotate('', (x1, y1 + 0.6), (x0, y0 - 0.6), arrowprops=dict(arrowstyle='->', color=INK2, lw=0.8))


box(2.2, 9, 'WHO FluNet, 2010–2026\n187 countries')
box(2.2, 6.9, '≥ 1 week with ≥ 20 specimens\n174 countries'); arrow(2.2, 9, 2.2, 6.9)
box(2.2, 4.8, 'Influenza panel (≥ 104 labelled weeks,\n≥ 5 surge weeks): 131 countries,\n61,062 country-weeks', h=1.3, c='#e8f0fb'); arrow(2.2, 6.9, 2.2, 4.8)
box(1.25, 2.2, 'Google Trends\ndownloaded: 125\nusable (≤ 20% zero\nweeks): 74', w=2.0, h=1.6, c='#fdeee6'); arrow(1.7, 4.65, 1.25, 2.4)
box(3.6, 2.2, 'FluID ILI observed\n(≥ 50 consultations):\n60 countries\nboth streams: 26', w=2.0, h=1.6, c='#e6f6ef'); arrow(2.7, 4.65, 3.6, 2.4)
box(7.4, 9, 'WHO COVID-19 weekly cases\n240 reporting areas')
box(7.4, 6.9, 'Matched to JHU population\n194; population ≥ 500,000: 166'); arrow(7.4, 9, 7.4, 6.9)
box(7.4, 4.8, 'COVID-19 panel, Mar 2020–Jun 2023\n(≥ 80 labelled weeks, ≥ 5 surge weeks):\n137 countries, 21,396 country-weeks', w=3.2, h=1.3, c='#e8f0fb'); arrow(7.4, 6.9, 7.4, 4.8)
save(fig, 'figS3_inclusion_flow')
print('figures ok')
