"""Leakage-free alert thresholds and episode metrics (reviewer point: thresholds must not use test labels).

For every country partition (seed) and outer fold k, the 95%-specificity threshold applied to the held-out
countries of fold k is estimated WITHOUT their labels:
  * 'train'    : zero-shot scorers (growth, expgrowth, chronos): quantile of the scores of training-country
                 negatives (no model is fitted, so these scores are genuine out-of-sample scores).
  * 'nested'   : trained models re-fitted with an inner 4-fold country-held-out CV inside the training
                 countries; threshold = quantile of inner out-of-fold scores of training-country negatives.
  * 'otherfold': quantile of training-country negatives' outer out-of-fold scores (scores from models that
                 never saw that country). Used for expensive models; validated against 'nested' for LightGBM.
Episode metrics are computed per partition and averaged over the three partitions.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import country_folds, episodes

SPEC = 0.95


def q_neg(scores, y):
    return float(np.quantile(scores[y == 0], SPEC))


def make_model(name, seed):
    if name in ('lgbm', 'lgbm_chronos', 'case', 'fused'):
        return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=1)
    if name == 'logreg':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    raise ValueError(name)


def nested_threshold(name, X, y, groups, tr, seed):
    """Inner country-held-out CV on the training countries only."""
    inner = np.full(len(tr), np.nan)
    for itr, ite in country_folds(groups[tr], 4, 100 + seed):
        m = make_model(name, seed); m.fit(X[tr][itr], y[tr][itr])
        inner[ite] = m.predict_proba(X[tr][ite])[:, 1]
    return q_neg(inner, y[tr])


def row_thresholds(method, score, y, groups, seed, name=None, X=None):
    """Return a per-row threshold array for one partition."""
    thr = np.full(len(y), np.nan)
    for tr, te in country_folds(groups, 5, seed):
        if method in ('train', 'otherfold'):
            t = q_neg(score[tr], y[tr])
        elif method == 'nested':
            t = nested_threshold(name, X, y, groups, tr, seed)
        thr[te] = t
    return thr


def crossing_weeks(P, eps, R, A, log, H=3):
    """Crossing week: first week onset+1 .. onset+H whose OBSERVED signal (full weekly series, labelled or not)
    reaches the threshold fixed at onset, max(R*b_onset, b_onset + A). Because the onset week is labelled, at least
    one such week exists by construction, so no episode is excluded."""
    from signal_series import load_signal
    S = load_signal('covid' if log else 'flu').set_index(['iso3', 'date']).x
    base = P.set_index(['iso3', 'date']).baseline
    out = []
    for _, e in eps.iterrows():
        b = base.loc[(e.iso3, e.start)]; thr = max(R * b, b + A); hit = pd.NaT
        for k in range(1, H + 1):
            d = e.start + pd.Timedelta(weeks=k)
            v = S.get((e.iso3, d), np.nan)
            if v >= thr - 1e-9:
                hit = d; break
        out.append(hit)
    e2 = eps.copy(); e2['cross'] = out
    return e2.dropna(subset=['cross']).reset_index(drop=True)


def first_alerts(P, alert, eps, pre=4):
    """First week in [onset - pre weeks, episode end] with an alert; NaT if none."""
    P = P.assign(_a=alert)
    by = {c: g.sort_values('date') for c, g in P.groupby('iso3')}
    out = []
    for _, e in eps.iterrows():
        g = by[e.iso3]; w = g[(g.date >= e.start - pd.Timedelta(weeks=pre)) & (g.date <= e.end) & g._a]
        out.append(w.date.iloc[0] if len(w) else pd.NaT)
    return pd.Series(out, index=eps.index)


def episode_metrics(eps, fa):
    L = (eps.cross - fa).dt.days / 7
    det = L.notna()
    return dict(detected=float(det.mean()), median_lead=float(L[det].median()) if det.any() else np.nan,
                ge1=float((L >= 1).mean()), ge2=float((L >= 2).mean()))


def rowlevel(y, score, thr):
    a = score > thr
    return dict(sens=float(a[y == 1].mean()), spec=float((~a[y == 0]).mean()))
