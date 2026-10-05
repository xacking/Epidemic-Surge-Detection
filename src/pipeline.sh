#!/bin/bash
# waits for the flu benchmark, then runs remaining jobs sequentially
while pgrep -f "run_benchmark.py flu H3_R1.5" > /dev/null; do sleep 20; done
python3 run_chronos.py flu H3_R1.5 > chronos_flu.log 2>&1
python3 run_benchmark.py covid H3_R1.5 growth,expgrowth,logreg,rf,lgbm,cnnlstm_rf > bench_covid.log 2>&1
python3 run_chronos.py covid H3_R1.5 > chronos_covid.log 2>&1
python3 sim_regimes.py > sim.log 2>&1
# sensitivity: alternative surge definitions (LightGBM, logistic, rules, chronos)
for args in "2 1.5 5 10" "4 1.5 5 10" "3 2.0 10 20"; do
  set -- $args
  python3 build_panels.py $1 $2 $3 $4 >> sens_build.log 2>&1
  python3 run_benchmark.py flu H$1_R$2 growth,expgrowth,logreg,lgbm >> sens.log 2>&1
  python3 run_benchmark.py covid H$1_R$2 growth,expgrowth,logreg,lgbm >> sens.log 2>&1
done
echo PIPELINE DONE > pipeline.done
