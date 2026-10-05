"""Zero-shot Chronos-Bolt surge probability. No training on any country.
P(surge) approx max_k P(x_{t+k} >= thr_t), from interpolated predictive quantiles."""
import sys, time, numpy as np, pandas as pd, torch
from chronos import BaseChronosPipeline
DS = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else 'H3_R1.5'
MODEL = sys.argv[3] if len(sys.argv) > 3 else 'amazon/chronos-bolt-small'
H = int(TAG.split('_')[0][1:]); R = float(TAG.split('_')[1][1:])
A = 5.0 if DS == 'flu' else 10.0
LOG = DS == 'covid'
torch.set_num_threads(2)
df = pd.read_parquet(f'panel_{DS}_{TAG}.parquet').reset_index(drop=True)
pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map='cpu', torch_dtype=torch.float32)
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
C = 104
ctx = np.full((len(df), C), np.nan, dtype=np.float32)
for c, g in df.groupby('iso3'):
    s = g.set_index('date').x_lag0
    s = np.expm1(s) if LOG else s
    full = s.reindex(pd.date_range(s.index.min() - pd.Timedelta(weeks=C), s.index.max(), freq='7D'))
    v = full.values.astype(np.float32); pos = {d: i for i, d in enumerate(full.index)}
    for r, d in zip(g.index, g.date):
        i = pos[d]; ctx[r] = v[i - C + 1:i + 1]
b = df.baseline.values           # baseline is stored on the raw (untransformed) scale
thr = np.maximum(R * b, b + A)
prob = np.zeros(len(df)); med = np.zeros(len(df))
t0 = time.time()
for i in range(0, len(df), 512):
    q, m = pipe.predict_quantiles(torch.tensor(ctx[i:i + 512]), prediction_length=H, quantile_levels=QL)
    q = q.numpy()  # (n, H, 9)
    tt = thr[i:i + 512, None]
    pk = np.zeros(q.shape[:2])
    for k in range(H):
        for j in range(len(q)):
            qs = np.maximum.accumulate(q[j, k])
            # P(X >= thr) = 1 - F(thr); F by linear interpolation, flat tails clipped at 0.05/0.95
            F = np.interp(tt[j, 0], qs, QL, left=0.05, right=0.95)
            pk[j, k] = 1 - F
    prob[i:i + 512] = pk.max(1)
    med[i:i + 512] = (q[:, :, 4].max(1) - thr[i:i + 512])
    if i % 10240 == 0:
        print(i, f'{time.time() - t0:.0f}s', flush=True)
fn = f'preds/{DS}_{TAG}.parquet'
P = pd.read_parquet(fn)
tag = 'chronos' if 'small' in MODEL else 'chronos_' + MODEL.split('-')[-1]
scale = 1.0 if not LOG else 1.0
P[tag] = prob + 1e-4 * np.tanh(med / (np.abs(thr) + 1))   # tie-break by median exceedance
P[tag + '_prob'] = prob
P.to_parquet(fn)
print('saved', tag, f'{time.time() - t0:.0f}s')
