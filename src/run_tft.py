"""Temporal Fusion Transformer (neuralforecast) as a surge detector, country-held-out.
A global TFT forecaster with quantile loss is trained on the training countries' weekly series, then run
on held-out countries at every week (rolling, step 1). Surge probability = max_k P(x_{t+k} >= threshold),
from the predicted quantiles (same conversion as Chronos). One country partition (seed 0) x 5 folds.
usage: python run_tft.py flu|covid [max_steps]"""
import os, sys, time, logging, warnings, numpy as np, pandas as pd, torch
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import MQLoss
from common import country_folds
warnings.filterwarnings('ignore'); logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
torch.set_num_threads(2)
DS = sys.argv[1]; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
SEED = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 0
TAG = 'H3_R1.5'; H = 3; R = 1.5; A = 5.0 if DS == 'flu' else 10.0; LOG = DS == 'covid'
L = 52
VARIANT = os.environ.get('TFT_VARIANT', 'base')
HID, STEPS_V = (64, 2000) if VARIANT == 'tuned' else (32, None)
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
df = pd.read_parquet(f'panel_{DS}_{TAG}.parquet').reset_index(drop=True)


def series(sub):
    """Contiguous weekly series per country with an availability mask (missing weeks -> 0, mask 0)."""
    out = []
    for c, g in sub.groupby('iso3'):
        s = g.set_index('date').x_lag0
        s = np.expm1(s) if LOG else s
        idx = pd.date_range(s.index.min() - pd.Timedelta(weeks=L), s.index.max() + pd.Timedelta(weeks=H), freq='7D')
        s = s.reindex(idx)
        y = (np.log1p(s) if LOG else s / 10.0)          # scale for training stability
        yv = y.interpolate(limit_direction='both').fillna(0).values if VARIANT == 'tuned' else y.fillna(0).values
        out.append(pd.DataFrame({'unique_id': c, 'ds': idx, 'y': yv,
                                 'available_mask': s.notna().astype(float).values}))
    return pd.concat(out, ignore_index=True)


def inv(v):
    return np.expm1(v) if LOG else v * 10.0


pred = np.full(len(df), np.nan); t0 = time.time()
SMOKE = len(sys.argv) > 3 and sys.argv[3] == 'smoke'
for k, (tr, te) in enumerate(country_folds(df.iso3.values, 5, SEED)):
    if SMOKE:
        tr = tr[np.isin(df.iso3.values[tr], df.iso3.values[tr][:1] if False else np.unique(df.iso3.values[tr])[:8])]
        te = te[np.isin(df.iso3.values[te], np.unique(df.iso3.values[te])[:3])]
    train = series(df.iloc[tr]); test = series(df.iloc[te])
    model = TFT(h=H, input_size=L, hidden_size=HID, n_head=4, loss=MQLoss(quantiles=QL), learning_rate=1e-3,
                max_steps=STEPS_V or STEPS, batch_size=64, windows_batch_size=256, scaler_type='robust', random_seed=SEED,
                enable_progress_bar=False, enable_model_summary=False, accelerator='cpu', logger=False)
    nf = NeuralForecast(models=[model], freq='7D')
    nf.fit(train)
    # rolling one-step-ahead windows over the held-out countries, without retraining
    nf.dataset, nf.uids, nf.last_dates, nf.ds = nf._prepare_fit(test, None, 'unique_id', 'ds', 'y')
    ins = nf.predict_insample(step_size=1); ins_cols = list(ins.columns); print(ins_cols) if SMOKE else None
    qcols = [c for c in ins.columns if c.startswith('TFT') and ('-lo-' in c or '-hi-' in c or c == 'TFT-median')]
    # map quantile columns
    def qcol(q):
        if q == 0.5:
            return 'TFT-median'
        lvl = int(round(abs(1 - 2 * q) * 100))
        return f'TFT-lo-{lvl}.0' if q < 0.5 else f'TFT-hi-{lvl}.0'
    ins = ins.rename(columns={'unique_id': 'iso3'})
    # each row: forecast for date ds made at cutoff; horizon k = (ds - cutoff)/7
    ins['k'] = ((ins.ds - ins.cutoff).dt.days // 7).astype(int)
    Q = np.stack([inv(ins[qcol(q)].values) for q in QL], 1)
    Q = np.maximum.accumulate(Q, axis=1)
    ins = ins[['iso3', 'cutoff', 'k']].copy(); ins['Q'] = list(Q)
    sub = df.iloc[te][['iso3', 'date', 'baseline']].copy(); sub['row'] = te
    m = sub.merge(ins, left_on=['iso3', 'date'], right_on=['iso3', 'cutoff'], how='left')
    b = m.baseline.values; thr = np.maximum(R * b, b + A)
    pk = np.array([1 - np.interp(t, q, QL, left=0.05, right=0.95) if isinstance(q, np.ndarray) else np.nan
                   for t, q in zip(thr, m.Q)])
    m['p'] = pk
    p = m.groupby('row').p.max()
    pred[p.index.values] = p.values
    if SMOKE:
        print(ins_cols, m[['iso3','date','p']].dropna().head(), np.isfinite(p.values).mean()); sys.exit()
    print(f'fold {k} done {time.time() - t0:.0f}s  coverage {np.isfinite(p.values).mean():.3f}', flush=True)

P = pd.read_parquet(f'preds/{DS}_{TAG}.parquet')
P[f'tft{"_tuned" if VARIANT == "tuned" else ""}_s{SEED}'] = pred
P.to_parquet(f'preds/{DS}_{TAG}.parquet')
print(f'saved tft_s{SEED}', np.isnan(pred).mean())
