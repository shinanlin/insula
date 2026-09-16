#!/bin/bash
#SBATCH --job-name=insula_tfr_fig
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/insula_tfr_fig_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/insula_tfr_fig_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
# One SVG per insula electrode after numerical TFR products exist.
set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/insula_tfr_fig_mpl_${SLURM_JOB_ID}"

python -m pytest -q tests/test_tfr.py
python -m src.tfr.run figures
