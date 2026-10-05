# Epidemic surge-detection benchmark

Code for the paper **"Epidemic surge detection in unseen countries: evaluating tabular, foundation, and sequence models across global surveillance data"** (Muhammed, Ogar, Obansa & Ishaq, 2026, revised manuscript).

The benchmark asks one question in a leakage-safe way: *given surveillance data up to week t, will the signal surge within the next three weeks?* It is applied to influenza in 131 countries (WHO FluNet, 2010–2026) and COVID-19 in 137 countries (WHO, 2020–2023), with every model evaluated on countries it never saw in training.

## Main results

| | Influenza PR-AUC | COVID-19 PR-AUC |
|---|---|---|
| TabPFN v2 (cloud API) | 0.614 | **0.733** |
| LightGBM + Chronos forecast feature | **0.619** | 0.716 |
| LightGBM | 0.615 | 0.709 |
| CNN-LSTM-RF hybrid | 0.598 | 0.686 |
| Temporal Fusion Transformer | 0.523 | 0.613 |
| Chronos-Bolt (zero-shot) | 0.429 | 0.426 |
| Base rate | 0.229 | 0.209 |

Adding Google Trends (74 countries) changed PR-AUC by +0.006 and moved alerts by +0.02 weeks (95% CI −0.05 to +0.09) at matched 95% specificity, with thresholds fixed on training countries only.

## Repository layout

```
src/
  build_panels.py    weekly country panels + leakage-safe surge labels
  run_benchmark.py   model ladder, country-held-out CV (5 folds x 3 partitions)
  run_chronos.py     zero-shot Chronos-Bolt surge probabilities
  run_stack.py       LightGBM + Chronos feature
  run_tft.py         Temporal Fusion Transformer (NeuralForecast), quantiles -> surge probability
  run_tabpfn_cloud.py TabPFN v2 via the Prior Labs cloud API (needs your own API key)
  evaluate.py        metrics, country-bootstrap CIs, episode lead time
  fetch_trends.py    Google Trends 'Influenza' topic per country
  run_fusion.py      second-stream analysis (Trends, ILI) with paired lead time
  sim_regimes.py     seasonal stochastic SEIRS simulation
  make_figures.py    paper figures
  common.py          shared helpers
  pipeline.sh        sensitivity runs
results/             summary CSVs reported in the paper
```

## Reproduce

```bash
pip install -r requirements.txt
cd src
# place the WHO FluNet, FluID, WHO COVID-19 and JHU lookup CSVs in src/data/ (paths at the top of build_panels.py)
python build_panels.py                 # defaults: H=3, R=1.5, floors 5 pp / 10 per 100k
python run_benchmark.py flu H3_R1.5
python run_benchmark.py covid H3_R1.5
python run_chronos.py flu H3_R1.5 && python run_chronos.py covid H3_R1.5
python run_stack.py flu H3_R1.5 && python run_stack.py covid H3_R1.5
for s in 0 1 2; do python run_tft.py flu 800 $s; python run_tft.py covid 800 $s; done
# TabPFN: put your Prior Labs API key (https://ux.priorlabs.ai) in ~/.tabpfn_token — never commit it
python run_tabpfn_cloud.py flu && python run_tabpfn_cloud.py covid   # writes preds/*_tabpfn.parquet; merge its columns into preds/<disease>_H3_R1.5.parquet
python evaluate.py flu H3_R1.5 && python evaluate.py covid H3_R1.5
python fetch_trends.py                 # slow and rate-limited
python run_fusion.py H3_R1.5
python sim_regimes.py
python make_figures.py
```

Run time on a 2-core CPU is a few hours, mostly the CNN-LSTM-RF, the Temporal Fusion Transformer and the Trends download.

## Additional analyses (run after the main pipeline, from `src/`)

```
src/
  frozen.py           alert thresholds fixed without test labels (nested / training-country scores), crossing week, episode metrics
  signal_series.py    full weekly signal per country (crossing week = first of onset+1..onset+3 reaching the threshold)
  run_frozen.py       episode metrics for all ten models with leakage-free thresholds  -> results/revision/results_frozen_*.csv
  run_fusion2.py      second streams: base | delay1 | delay2 | window | time, streams trends | ili | both | trendsonly
  summarize_fusion2.py  2x2 detection table, paired lead with country bootstrap -> results_fusion2_summary.csv
  run_timesplit.py    countries AND time held out (flu cutoff 2020-01-01, COVID-19 2021-07-01)
  region_ci.py        regional PR-AUC with country-bootstrap CIs
  eval_extra.py       tuned TFT / 20-epoch CNN-LSTM-RF comparisons
  make_figures_rev.py revised Figures 1 and 3, Supplementary Figure S3
```

```bash
python signal_series.py
python run_frozen.py flu && python run_frozen.py covid
for v in base delay1 delay2 window time; do python run_fusion2.py $v trends; done
python run_fusion2.py base ili; python run_fusion2.py base both; python run_fusion2.py base trendsonly
python summarize_fusion2.py
python run_timesplit.py flu && python run_timesplit.py covid      # TabPFN rows need the API key
BASEW=4 python build_panels.py 3 1.5 5 10                         # four-week-baseline labels
TFT_VARIANT=tuned python run_tft.py flu 2000 0
SEEDS=0 CNN_EPOCHS=20 python run_benchmark.py flu H3_R1.5 cnnlstm_rf
```

Tested with Python 3.13, numpy 2.5, pandas 2.3, scikit-learn 1.6, LightGBM 4.7, PyTorch 2.14 (CPU), NeuralForecast 3.2, chronos-forecasting 2.3 (amazon/chronos-bolt-small), tabpfn-client 0.6, on a 2-core CPU. Seeds: partitions 0–2, inner folds 100–102.

## Data sources

- WHO FluNet and FluID (Global Influenza Surveillance and Response System): https://www.who.int/tools/flunet
- WHO COVID-19 dashboard: https://data.who.int/dashboards/covid19/data
- JHU CSSE population lookup (Dong, Du & Gardner, 2020)
- Google Trends, topic /m/0cycc

Raw data are not redistributed here; download them from the sources above.

## Licence

MIT. See `LICENSE`. Please cite using `CITATION.cff`.
