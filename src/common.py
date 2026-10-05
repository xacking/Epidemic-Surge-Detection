import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold

H_DEFAULT = 3


def case_features(df, extra_streams=()):
    base = [c for c in df.columns if c.startswith('x_')] + ['woy_sin', 'woy_cos', 'n_obs8']
    if 'log_tests_lag0' in df.columns:
        base += [c for c in df.columns if c.startswith('log_tests_')]
    if 'hemi_nh' in df.columns:
        base += ['hemi_nh']
    for s in extra_streams:
        base += [c for c in df.columns if c.startswith(s + '_')]
    return base


def country_folds(groups, n_splits=5, seed=0):
    """Country-held-out folds: shuffle country list with seed, then GroupKFold."""
    u = np.array(sorted(set(groups)))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(u))
    fold_of = {c: i % n_splits for i, c in zip(range(len(u)), u[perm])}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_splits)]


def sens_at_spec(y, s, spec=0.95):
    thr = np.quantile(s[y == 0], spec)
    return float(np.mean(s[y == 1] > thr)), thr


def metrics(y, s, p=None):
    out = dict(roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
               sens_at_95spec=sens_at_spec(y, s, 0.95)[0], sens_at_90spec=sens_at_spec(y, s, 0.90)[0],
               prevalence=float(np.mean(y)))
    if p is not None:
        out['brier'] = brier_score_loss(y, np.clip(p, 0, 1))
    return out


def cluster_boot_diff(df, a, b, metric='pr_auc', B=1000, seed=0):
    """Paired cluster bootstrap over countries for metric(a) - metric(b)."""
    rng = np.random.RandomState(seed)
    f = average_precision_score if metric == 'pr_auc' else roc_auc_score
    groups = df.groupby('iso3').indices
    keys = np.array(list(groups.keys()))
    y = df.y.values; sa = df[a].values; sb = df[b].values
    base = f(y, sa) - f(y, sb)
    diffs = []
    for _ in range(B):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(f(y[idx], sa[idx]) - f(y[idx], sb[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return base, lo, hi, max(p, 1 / len(diffs))


def episodes(df, gap=2):
    """Surge episodes: runs of y==1 per country (merging gaps <= gap weeks)."""
    eps = []
    for c, g in df.sort_values('date').groupby('iso3'):
        d = g.date.values; yy = g.y.values
        start = None; last = None
        for i in range(len(g)):
            if yy[i] == 1:
                if start is None or (d[i] - d[last]) / np.timedelta64(7, 'D') > gap + 1:
                    if start is not None:
                        eps.append((c, d[start], d[last]))
                    start = i
                last = i
        if start is not None:
            eps.append((c, d[start], d[last]))
    return pd.DataFrame(eps, columns=['iso3', 'start', 'end'])


def first_alerts(df, score_col, thr, eps, pre=4):
    """First alert week for each episode within [start - pre weeks, end]. NaT if none."""
    out = []
    by = {c: g.sort_values('date') for c, g in df.groupby('iso3')}
    for _, e in eps.iterrows():
        g = by[e.iso3]
        w = g[(g.date >= e.start - pd.Timedelta(weeks=pre)) & (g.date <= e.end)]
        hit = w[w[score_col] > thr]
        out.append(hit.date.iloc[0] if len(hit) else pd.NaT)
    return pd.Series(out, index=eps.index)
