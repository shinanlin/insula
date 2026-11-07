#!/bin/bash
#SBATCH --job-name=fig4_formal_publish
#SBATCH --account=coganlab
#SBATCH --partition=common,scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig4_formal_publish_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig4_formal_publish_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/cwork/ns458/insula-paper MPLBACKEND=Agg
TASK_MODE=${1:?stage or resolved}
TASK_PY=/hpc/home/ns458/miniconda3/envs/ieeg/bin/python
"$TASK_PY" -u -m src.decoding.figure4_collect "$TASK_MODE"
if [[ "$TASK_MODE" == stage ]]; then
    flock /cwork/ns458/insula-paper/results/fig4/.notebook_execution.lock "$TASK_PY" -u scripts/execute_paper_notebook.py 4 --phonetic-only
    "$TASK_PY" -u scripts/preview_completed_figure4.py
else
    flock /cwork/ns458/insula-paper/results/fig4/.notebook_execution.lock "$TASK_PY" -u scripts/execute_paper_notebook.py 4 --phonetic-resolved
fi
