import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from common import episodes, sens_at_spec
import os
os.makedirs('figures', exist_ok=True)

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e2dc', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
ORDER = ['growth', 'expgrowth', 'logreg', 'chronos', 'tft', 'cnnlstm_rf', 'rf', 'tabpfn', 'lgbm', 'lgbm_chronos']
COL = {m: ('#2a78d6' if m in ('lgbm', 'lgbm_chronos', 'tabpfn', 'rf') else '#52514e') for m in ORDER}  # identity is on the axis; colour marks the trained-tree / TabPFN group
SHORT = {'growth': 'Growth-rate rule', 'expgrowth': 'Exponential-growth\nextrapolation', 'chronos': 'Chronos-Bolt\n(zero-shot)',
         'logreg': 'Logistic regression', 'cnnlstm_rf': 'CNN-LSTM-RF hybrid', 'rf': 'Random forest', 'lgbm': 'LightGBM',
         'lgbm_chronos': 'LightGBM +\nChronos feature', 'tft': 'Temporal Fusion Transformer', 'tabpfn': 'TabPFN v2 (cloud)'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 300})


def save(fig, name):
    fig.savefig(f'figures/{name}.png', bbox_inches='tight'); fig.savefig(f'figures/{name}.pdf', bbox_inches='tight'); plt.close(fig)


# ---------------- Figure 1: example country + coverage ----------------
P = pd.read_parquet('preds/flu_H3_R1.5.parquet'); P['date'] = pd.to_datetime(P.date)
lg = [c for c in P.columns if c.startswith('lgbm_s')]; P['lgbm'] = P[lg].mean(1)
_, thr = sens_at_spec(P.y.values, P.lgbm.values, 0.95)
panel = pd.read_parquet('panel_flu_H3_R1.5.parquet')
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={'width_ratios': [2.3, 1]})
ax = axes[0]
g = P[(P.iso3 == 'FRA') & (P.date >= '2015-07-01') & (P.date < '2020-01-01')].sort_values('date')
gl = g.set_index('date').x_lag0.reindex(pd.date_range(g.date.min(), g.date.max(), freq='7D'))
ax.plot(gl.index, gl.values, color=CAT[0], lw=1.6, label='Influenza test positivity (%; gaps = no data)')
for d, yy in zip(g.date, g.y):
    if yy == 1:
        ax.axvspan(d - pd.Timedelta(days=3), d + pd.Timedelta(days=4), color='#eda100', alpha=0.22, lw=0)
al = g[g.lgbm > thr]
ax.scatter(al.date, np.full(len(al), -4), marker='|', s=60, color=CAT[7], label='LightGBM alert (95% specificity)')
ax.fill_between([], [], color='#eda100', alpha=0.3, label='Surge within next 3 weeks (label)')
ax.set_ylabel('Positivity (%)'); ax.set_ylim(-7, None); ax.grid(axis='y', color=GRID, lw=0.6)
ax.set_title('a  Task illustration: France, 2015–2019 (held-out predictions)', loc='left', fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=7.5, loc='upper left', ncol=1)
ax = axes[1]
reg = panel.groupby('region').iso3.nunique().sort_values()
cov = pd.read_parquet('panel_covid_H3_R1.5.parquet')
names = {'AFR': 'Africa', 'AMR': 'Americas', 'EMR': 'E. Medit.', 'EUR': 'Europe', 'SEAR': 'SE Asia', 'WPR': 'W. Pacific'}
ax.barh([names.get(r, r) for r in reg.index], reg.values, color=CAT[0], height=0.6)
for i, v in enumerate(reg.values):
    ax.text(v + 0.5, i, str(v), va='center', fontsize=8, color=INK2)
ax.set_xlabel('Countries (influenza panel)')
ax.set_title(f'b  Coverage: {panel.iso3.nunique()} countries, {len(panel):,} country-weeks', loc='left', fontsize=10, color=INK)
save(fig, 'fig1_task_coverage')

# ---------------- Figure 2: model ladder ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, ds, ttl in zip(axes, ['flu', 'covid'], ['a  Influenza (131 countries)', 'b  COVID-19 (137 countries)']):
    r = pd.read_csv(f'results_{ds}_H3_R1.5.csv', index_col=0).reindex(ORDER)
    yv = np.arange(len(ORDER))
    for i, m in enumerate(ORDER):
        ax.plot([r.loc[m, 'pr_lo'], r.loc[m, 'pr_hi']], [i, i], color=COL[m], lw=2, solid_capstyle='round')
        ax.scatter(r.loc[m, 'pr_avg'], i, s=46, color=COL[m], zorder=3, edgecolor='white', linewidth=1.2)
        ax.text(r.loc[m, 'pr_hi'] + 0.008, i, f"{r.loc[m, 'pr_avg']:.3f}", va='center', fontsize=7.5, color=INK2)
    ax.axvline(r.prevalence.iloc[0], color=INK2, ls=':', lw=1)
    ax.text(r.prevalence.iloc[0] + 0.005, -0.45, 'base rate', fontsize=7, color=INK2)
    ax.set_yticks(yv); ax.set_yticklabels([SHORT[m].replace('\n', ' ') for m in ORDER])
    ax.set_xlabel('PR-AUC on held-out countries (95% CI, country bootstrap)')
    ax.grid(axis='x', color=GRID, lw=0.6); ax.set_title(ttl, loc='left', fontsize=10, color=INK)
    ax.set_xlim(0.15, 0.8)
save(fig, 'fig2_model_ladder')

# ---------------- Figure 3: digital signals - paired lead ----------------
def paired(label, a, b):
    F = pd.read_parquet(f'preds/fusion_{label}_H3_R1.5.parquet'); F['date'] = pd.to_datetime(F.date)
    y = F.y.values; eps = episodes(F); by = {c: gg.sort_values('date') for c, gg in F.groupby('iso3')}
    fa = {}
    for name in (a, b):
        _, t = sens_at_spec(y, F[name].values, 0.95); o = []
        for _, e in eps.iterrows():
            gg = by[e.iso3]; w = gg[(gg.date >= e.start - pd.Timedelta(weeks=4)) & (gg.date <= e.end)]
            h = w[w[name] > t]; o.append(h.date.iloc[0] if len(h) else pd.NaT)
        fa[name] = pd.Series(o)
    both = fa[a].notna() & fa[b].notna()
    return ((fa[a][both] - fa[b][both]).dt.days / 7).values


fr = pd.read_csv('results_fusion_H3_R1.5.csv', index_col=0)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), gridspec_kw={'width_ratios': [1.1, 1, 1]})
ax = axes[0]
sub = fr[fr.subset == 'trends']
labs = ['Case data\nonly', 'Case +\nGoogle Trends', 'Google Trends\nonly']
vals = [sub.loc['case', 'pr_auc'], sub.loc['case+trends', 'pr_auc'], sub.loc['trends_only', 'pr_auc']]
ax.bar(labs, vals, color=[CAT[0], CAT[1], CAT[3]], width=0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8, color=INK2)
ax.axhline(sub.prevalence.iloc[0], color=INK2, ls=':', lw=1)
ax.set_ylabel('PR-AUC (held-out countries)'); ax.set_ylim(0, 0.75); ax.tick_params(axis='x', labelsize=7.5)
ax.set_title(f"a  Detection\n{int(sub.n_countries.iloc[0])} countries with search data", loc='left', fontsize=9, color=INK)
for ax, (lab, nm, ttl, c) in zip(axes[1:], [('trends', 'case+trends', 'b  Adding Google Trends', CAT[1]), ('ili', 'case+ili', 'c  Adding syndromic ILI', CAT[2])]):
    d = paired(lab, 'case', nm)
    bins = np.arange(-6.5, 7.5, 1)
    ax.hist(d, bins=bins, color=c, rwidth=0.85)
    ax.axvline(0, color=INK, lw=1)
    row = fr[(fr.subset == lab)].loc[nm]
    ax.set_title(f'{ttl}\nmean {row.lead_mean:+.2f} wk (95% CI {row.lead_lo:+.2f} to {row.lead_hi:+.2f})\n{len(d)} paired episodes', loc='left', fontsize=9, color=INK)
    ax.set_xlabel('Extra warning from added stream (weeks)\n< 0 later   |   > 0 earlier')
    ax.set_ylabel('Surge episodes'); ax.grid(axis='y', color=GRID, lw=0.6)
    ax.text(0.98, 0.95, f'{(d > 0).mean():.0%} earlier\n{(d == 0).mean():.0%} same week\n{(d < 0).mean():.0%} later',
            transform=ax.transAxes, ha='right', va='top', fontsize=8, color=INK2)
save(fig, 'fig3_digital_signals')

# ---------------- Figure 4: simulation regime map ----------------
s = pd.read_csv('results_sim_regimes.csv'); s['dpr'] = s.pr_fused - s.pr_case
g = s.groupby(['advantage_days', 'sigma']).mean(numeric_only=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, col, ttl, fmt, cmap in [(axes[0], 'lead_mean', 'a  Mean extra warning from second stream (weeks)', '{:.2f}', 'Blues'),
                                (axes[1], 'dpr', 'b  PR-AUC gain from second stream', '{:+.3f}', 'Oranges')]:
    M = g[col].unstack('sigma')
    im = ax.imshow(M.values, cmap=cmap, aspect='auto', origin='lower', vmin=0)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f'{v}' for v in M.columns])
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([f'{v}' for v in M.index])
    ax.set_xlabel('Noise in second stream (log-scale SD)'); ax.set_ylabel('Timing advantage over reported cases (days)')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8,
                    color='white' if v > 0.6 * np.nanmax(M.values) else INK)
    ax.set_title(ttl, loc='left', fontsize=10, color=INK)
    ax.spines[['left', 'bottom']].set_visible(False)
save(fig, 'fig4_simulation_regimes')

# ---------------- Supplementary: per-region LightGBM PR-AUC ----------------
P2 = P.merge(panel[['iso3', 'region']].drop_duplicates(), on='iso3')
rows = []
for r, gg in P2.groupby('region'):
    if gg.y.nunique() > 1:
        rows.append((names.get(r, r), average_precision_score(gg.y, gg.lgbm), gg.y.mean(), gg.iso3.nunique()))
rr = pd.DataFrame(rows, columns=['region', 'pr_auc', 'prevalence', 'countries']).sort_values('pr_auc')
rr.to_csv('results_flu_by_region.csv', index=False)
fig, ax = plt.subplots(figsize=(5.5, 2.8))
ax.barh(rr.region, rr.pr_auc, color=CAT[0], height=0.6, label='LightGBM PR-AUC')
ax.scatter(rr.prevalence, rr.region, marker='|', s=120, color=INK, label='Base rate', zorder=3)
ax.set_xlabel('PR-AUC (influenza, held-out countries)'); ax.legend(frameon=False, fontsize=7.5, loc='lower right')
ax.grid(axis='x', color=GRID, lw=0.6)
save(fig, 'figS1_region')
print('figures done')
