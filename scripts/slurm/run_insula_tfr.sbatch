#!/bin/bash
#SBATCH --job-name=insula_tfr
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/insula_tfr_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/insula_tfr_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
# Hammers AIC/PIC bipolar TFR (AF recipe). Chain: bash scripts/slurm/submit_insula_tfr.sh
set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/insula_tfr_mpl_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/work/ns458/numba_cache}"
export MNE_CONFIG_PATH=/work/ns458/mne_config
mkdir -p logs/slurm /work/ns458/numba_cache /work/ns458/mne_config

MODE=${1:?Use test, smoke, or run}
case "$MODE" in
  test)
    python -m pytest -q tests/test_tfr.py
    ;;
  smoke)
    python -m src.tfr.run run \
      --task PhonemeSequence \
      --subject D0088 \
      --n-perm 10 \
      --n-jobs "${SLURM_CPUS_PER_TASK}" \
      --output-root results/tfr/_smoke
    ;;
  run)
    python -m src.tfr.run run --n-perm 1000 --n-jobs "${SLURM_CPUS_PER_TASK}"
    ;;
  *)
    exit 2
    ;;
esac
