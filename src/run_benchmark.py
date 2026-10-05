"""Model ladder on country-held-out folds.  usage: python run_benchmark.py flu|covid [TAG] [models]"""
import sys, time, os, json, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from common import case_features, country_folds
warnings.filterwarnings('ignore')

DS = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else 'H3_R1.5'
MODELS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['growth', 'expgrowth', 'logreg', 'rf', 'lgbm', 'cnnlstm_rf', 'tabpfn']
SEEDS = [int(x) for x in os.environ.get('SEEDS', '0,1,2').split(',')]
H = int(TAG.split('_')[0][1:]); R = float(TAG.split('_')[1][1:])
A = (5.0 if DS == 'flu' else 10.0) * (2 if R >= 2 else 1)   # strict definition doubles the floor (matches build_panels args)
LOG = DS == 'covid'
df = pd.read_parquet(f'panel_{DS}_{TAG}.parquet').reset_index(drop=True)
FEATS = case_features(df)
print(DS, TAG, df.shape, len(FEATS), 'features', flush=True)
os.makedirs('preds', exist_ok=True)
out_fn = f'preds/{DS}_{TAG}.parquet'
P = pd.read_parquet(out_fn) if os.path.exists(out_fn) else df[['iso3', 'date', 'y', 'baseline', 'future_max', 'x_lag0']].copy()


def inv(v):
    return np.expm1(v) if LOG else v


# ---------------- no-training baselines ----------------
def growth_score(d):
    return d.x_diff1.fillna(0).values + d.x_diff2.fillna(0).values


def expgrowth_score(d):
    """Exponential-growth extrapolation: fit log-linear slope over last 3 weeks, project max over H weeks,
    score = projected log level minus log surge threshold."""
    lv = np.log1p(inv(d[['x_lag2', 'x_lag1', 'x_lag0']].values))
    t = np.array([-2, -1, 0.])
    slope = np.nansum((lv - np.nanmean(lv, 1, keepdims=True)) * (t - t.mean()), 1) / np.sum((t - t.mean()) ** 2)
    slope = np.nan_to_num(slope)
    proj = lv[:, 2] + np.maximum(slope, 0) * H
    b = d.baseline.values
    thr = np.maximum(R * b, b + A)
    return proj - np.log1p(thr)


# ---------------- CNN-LSTM-RF (thesis hybrid) ----------------
def build_sequences(d, W=26):
    seqs = np.zeros((len(d), W, 2), dtype=np.float32)
    for c, g in d.groupby('iso3'):
        s = g.set_index('date').x_lag0
        full = s.reindex(pd.date_range(s.index.min() - pd.Timedelta(weeks=W), s.index.max(), freq='7D'))
        v = full.values.astype(np.float32)
        pos = {dt: i for i, dt in enumerate(full.index)}
        for ridx, dt in zip(g.index, g.date):
            i = pos[dt]
            w = v[i - W + 1:i + 1]
            m = ~np.isnan(w)
            seqs[ridx, :, 0] = np.nan_to_num(w) / (10.0 if LOG else 100.0)
            seqs[ridx, :, 1] = m
    return seqs


def cnnlstm_rf(Xtr_seq, ytr, Xte_seq, Ftr, Fte, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed); torch.set_num_threads(2)

    class Enc(nn.Module):
        def __init__(s):
            super().__init__()
            s.conv = nn.Sequential(nn.Conv1d(2, 16, 3, padding=1), nn.ReLU())
            s.lstm = nn.LSTM(16, 32, batch_first=True, bidirectional=True)
            s.head = nn.Linear(64, 1)

        def emb(s, x):
            h = s.conv(x.transpose(1, 2)).transpose(1, 2)
            o, _ = s.lstm(h)
            return o[:, -1]

        def forward(s, x):
            return s.head(s.emb(x)).squeeze(-1)

    net = Enc(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    X = torch.tensor(Xtr_seq); Y = torch.tensor(ytr, dtype=torch.float32)
    pw = torch.tensor((1 - ytr.mean()) / ytr.mean())
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(X); rng = np.random.RandomState(seed)
    for ep in range(int(os.environ.get('CNN_EPOCHS', '6'))):
        perm = rng.permutation(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            opt.zero_grad(); l = lossf(net(X[b]), Y[b]); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        Etr = torch.cat([net.emb(X[i:i + 4096]) for i in range(0, n, 4096)]).numpy()
        Xt = torch.tensor(Xte_seq)
        Ete = torch.cat([net.emb(Xt[i:i + 4096]) for i in range(0, len(Xt), 4096)]).numpy()
    imp = SimpleImputer(strategy='median').fit(Ftr)
    rf = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=2, random_state=seed)
    rf.fit(np.hstack([Etr, imp.transform(Ftr)]), ytr)
    return rf.predict_proba(np.hstack([Ete, imp.transform(Fte)]))[:, 1]


SEQ = build_sequences(df) if 'cnnlstm_rf' in MODELS else None
X = df[FEATS].values.astype(np.float32); y = df.y.values.astype(int); groups = df.iso3.values

for m in MODELS:
    if m in ('growth', 'expgrowth'):
        P[m] = growth_score(df) if m == 'growth' else expgrowth_score(df)
        print(m, 'done', flush=True); continue
    for seed in SEEDS:
        col = f'{m}_s{seed}' if os.environ.get('CNN_EPOCHS', '6') == '6' or m != 'cnnlstm_rf' else f'{m}_e{os.environ["CNN_EPOCHS"]}_s{seed}'
        if col in P.columns:
            continue
        t0 = time.time(); pred = np.full(len(df), np.nan)
        for tr, te in country_folds(groups, 5, seed):
            if m == 'logreg':
                mdl = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
                mdl.fit(X[tr], y[tr]); pred[te] = mdl.predict_proba(X[te])[:, 1]
            elif m == 'rf':
                mdl = make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(300, min_samples_leaf=5, n_jobs=2, random_state=seed))
                mdl.fit(X[tr], y[tr]); pred[te] = mdl.predict_proba(X[te])[:, 1]
            elif m == 'lgbm':
                mdl = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=2)
                mdl.fit(X[tr], y[tr]); pred[te] = mdl.predict_proba(X[te])[:, 1]
            elif m == 'cnnlstm_rf':
                pred[te] = cnnlstm_rf(SEQ[tr], y[tr], SEQ[te], X[tr], X[te], seed)
            elif m == 'tabpfn':
                from tabpfn import TabPFNClassifier
                rng = np.random.RandomState(seed)
                sub = rng.choice(tr, min(3000, len(tr)), replace=False)
                imp = SimpleImputer(strategy='median').fit(X[sub])
                mdl = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True, n_estimators=2, random_state=seed)
                mdl.fit(imp.transform(X[sub]), y[sub])
                pr = [mdl.predict_proba(imp.transform(X[te[i:i + 2000]]))[:, 1] for i in range(0, len(te), 2000)]
                pred[te] = np.concatenate(pr)
        P[col] = pred
        if os.path.exists(out_fn):          # re-read so that columns written meanwhile by other jobs are kept
            Q = pd.read_parquet(out_fn); Q[col] = pred; Q.to_parquet(out_fn); P = Q
        else:
            P.to_parquet(out_fn)
        print(col, f'{time.time() - t0:.0f}s', flush=True)
if os.path.exists(out_fn):
    Q = pd.read_parquet(out_fn)
    for c in P.columns:
        if c not in Q.columns:
            Q[c] = P[c].values
    Q.to_parquet(out_fn)
else:
    P.to_parquet(out_fn)
print('saved', out_fn)
