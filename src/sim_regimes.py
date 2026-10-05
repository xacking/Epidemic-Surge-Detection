"""Mechanistic regime map: when can a second (faster, noisier) stream give earlier warning or better detection?
Seasonal stochastic SEIRS (daily, tau-leap), aggregated weekly exactly like the real benchmark.
Reported stream: onsets delayed by d_rep days, reporting fraction rho, Poisson noise.
Digital stream: onsets delayed by d_dig days, multiplicative lognormal noise sigma + media 'scare' bursts.
Label/features/evaluation identical to the real-data pipeline (surge in reported stream within H=3 weeks,
region-held-out LightGBM, paired lead at matched 95% specificity)."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from common import country_folds, metrics, episodes, sens_at_spec

H, R, A = 3, 1.5, None
YEARS = 8; NREG = 60


def simulate(rng, n, d_rep, d_dig, sigma):
    T = 365 * YEARS + 60
    N = rng.uniform(2e5, 2e6, n); R0 = rng.uniform(1.3, 2.2, n); amp = rng.uniform(0.15, 0.35, n)
    phase = rng.uniform(0, 365, n); gam = 1 / 4; sig = 1 / 2; omega = 1 / 400
    S = N * 0.6; E = np.full(n, 50.); I = np.full(n, 50.); Rr = N - S - E - I
    onset = np.zeros((T, n))
    season_mult = rng.lognormal(0, 0.12, (YEARS + 2, n))          # strain/season variability
    for t in range(T):
        beta = season_mult[t // 365] * R0 * gam * (1 + amp * np.cos(2 * np.pi * (t - phase) / 365))
        lam = 1 - np.exp(-beta * I / N)
        ne = rng.binomial(S.astype(np.int64), lam) + rng.poisson(0.5, n)  # importations
        ni = rng.binomial(E.astype(np.int64), 1 - np.exp(-sig))
        nr = rng.binomial(I.astype(np.int64), 1 - np.exp(-gam))
        ns = rng.binomial(Rr.astype(np.int64), 1 - np.exp(-omega))
        S = S - ne + ns; E = E + ne - ni; I = I + ni - nr; Rr = Rr + nr - ns
        onset[t] = ni
    rho = rng.uniform(0.02, 0.1, n)
    rho_t = rho * np.exp(rng.normal(0, 0.25, onset.shape))            # week-to-week reporting variation
    lam_rep = rho_t * np.roll(onset, d_rep, 0)
    rep = rng.poisson(rng.gamma(5.0, lam_rep / 5.0 + 1e-9))         # negative-binomial (k=5) reporting noise
    dig_mean = np.roll(onset, d_dig, 0) / N * 1e5 + 5
    scare = np.zeros_like(dig_mean)
    for j in range(n):                         # media bursts unrelated to incidence
        for _ in range(rng.poisson(YEARS * 2)):
            s0 = rng.integers(0, T - 30); scare[s0:s0 + rng.integers(7, 28), j] += rng.uniform(5, 40)
    dig = (dig_mean + scare) * np.exp(rng.normal(0, sigma, dig_mean.shape))
    sl = slice(60 + 365, T - (T - 60 - 365) % 7)       # drop burn-in, whole weeks
    W = lambda a: a[sl].reshape(-1, 7, n).sum(1)
    return W(rep), W(dig), N


def panel(rep, dig, N):
    rows = []
    for j in range(rep.shape[1]):
        x = 1e5 * rep[:, j] / N[j]                      # reported incidence per 100k
        d = np.log(dig[:, j])
        s = pd.Series(x); ls = np.log1p(s); dd = pd.Series(d)
        b = (s + s.shift(1)) / 2
        fut = pd.concat([s.shift(-k) for k in range(1, H + 1)], axis=1).max(1, skipna=False)
        thr = np.maximum(R * b, b + 2.0)
        f = pd.DataFrame({f'x_lag{L}': ls.shift(L) for L in range(8)})
        for k in (1, 2, 4):
            f[f'x_diff{k}'] = ls - ls.shift(k)
        for k in (1, 2, 4):
            f[f'dig_diff{k}'] = dd - dd.shift(k)
        f['dig_rel52'] = dd - dd.rolling(52, min_periods=26).mean()
        f['dig_lag1diff1'] = (dd - dd.shift(1)).shift(1)
        f['y'] = (fut >= thr).astype(float).where(fut.notna() & b.notna())
        f['baseline'] = b; f['x_lag0_raw'] = s
        f['iso3'] = f'r{j}'; f['date'] = pd.Timestamp('2000-01-03') + pd.to_timedelta(7 * np.arange(len(s)), 'D')
        rows.append(f.iloc[52:])
    p = pd.concat(rows, ignore_index=True).dropna(subset=['y'])
    p['x_lag0'] = p.x_lag0_raw
    return p


def evaluate(p, seed):
    case = [c for c in p.columns if c.startswith('x_') and c != 'x_lag0_raw']
    digf = [c for c in p.columns if c.startswith('dig_')]
    y = p.y.values.astype(int); g = p.iso3.values; out = {}
    P = p[['iso3', 'date', 'y', 'baseline', 'x_lag0']].copy()
    for name, F in {'case': case, 'case+dig': case + digf}.items():
        X = p[F].values.astype(np.float32); pred = np.zeros(len(p))
        for tr, te in country_folds(g, 5, seed):
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1, n_jobs=2, random_state=seed)
            m.fit(X[tr], y[tr]); pred[te] = m.predict_proba(X[te])[:, 1]
        P[name] = pred; out[name] = metrics(y, pred)
    eps = episodes(P); by = {c: gg.sort_values('date') for c, gg in P.groupby('iso3')}
    fa = {}
    for name in ('case', 'case+dig'):
        _, thr = sens_at_spec(y, P[name].values, 0.95); o = []
        for _, e in eps.iterrows():
            gg = by[e.iso3]; w = gg[(gg.date >= e.start - pd.Timedelta(weeks=4)) & (gg.date <= e.end)]
            h = w[w[name] > thr]; o.append(h.date.iloc[0] if len(h) else pd.NaT)
        fa[name] = pd.Series(o)
    both = fa['case'].notna() & fa['case+dig'].notna()
    dl = (fa['case'][both] - fa['case+dig'][both]).dt.days / 7
    return dict(pr_case=out['case']['pr_auc'], pr_fused=out['case+dig']['pr_auc'], roc_case=out['case']['roc_auc'],
                roc_fused=out['case+dig']['roc_auc'], lead_mean=dl.mean(), frac_earlier=(dl > 0).mean(), n_eps=int(both.sum()),
                det_case=fa['case'].notna().mean(), det_fused=fa['case+dig'].notna().mean())


if __name__ == '__main__':
    res = []
    grid_adv = [0, 3, 7, 14, 21]          # digital advantage = d_rep - d_dig (days)
    grid_sig = [0.1, 0.3, 0.6, 1.2]
    for seed in (0, 1):
        for adv in grid_adv:
            for sigma in grid_sig:
                rng = np.random.default_rng(1000 * seed + 10 * adv + int(10 * sigma))
                rep, dig, N = simulate(rng, NREG, d_rep=3 + adv, d_dig=3, sigma=sigma)
                r = evaluate(panel(rep, dig, N), seed)
                r.update(seed=seed, advantage_days=adv, sigma=sigma)
                res.append(r); print(r, flush=True)
                pd.DataFrame(res).to_csv('results_sim_regimes.csv', index=False)
