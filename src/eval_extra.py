"""PR-AUC (with country-bootstrap CI and paired difference vs the 6-epoch / baseline version) for extra model variants."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from common import cluster_boot_diff
ds, new, ref = sys.argv[1], sys.argv[2], sys.argv[3]
P = pd.read_parquet(f'preds/{ds}_H3_R1.5.parquet')
P = P.dropna(subset=[new, ref])
d, lo, hi, p = cluster_boot_diff(P, new, ref, 'pr_auc', B=1000)
P['lgbm_avg'] = P['lgbm_s0']
d2, lo2, hi2, p2 = cluster_boot_diff(P, new, 'lgbm_avg', 'pr_auc', B=1000)
print(f'{ds} {new}: PR {ap(P.y, P[new]):.3f}  ref {ref} {ap(P.y, P[ref]):.3f}  diff {d:+.3f} ({lo:+.3f},{hi:+.3f})  vs lgbm {d2:+.3f} ({lo2:+.3f},{hi2:+.3f})')
