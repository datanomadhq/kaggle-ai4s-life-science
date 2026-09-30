#!/bin/zsh
# Sequential GPU queue used for the report (shared Apple-Silicon GPU, ~2.4 s/step). Logs: data/logs/queue_*.log
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
.venv/bin/python run_all.py --dataset hepatopac --stage train,eval   > data/logs/queue_hepatopac.log 2>&1
.venv/bin/python run_all.py --dataset axiom --stage preprocess,train,eval > data/logs/queue_axiom.log 2>&1
.venv/bin/python run_all.py --dataset hepatopac --stage eval          >> data/logs/queue_hepatopac.log 2>&1   # transfer from axiom
