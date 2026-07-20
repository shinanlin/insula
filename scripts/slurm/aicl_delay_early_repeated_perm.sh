#!/bin/bash
#SBATCH --job-name=aicl_early_repperm
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/aicl_early_repperm_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/aicl_early_repperm_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
set -u
export MNE_DONTWRITE_HOME=true
export NUMBA_CACHE_DIR=/tmp/ns458-numba-${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/ns458-matplotlib-${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python scripts/aicl_delay_early_repeated_perm.py \
  --n-repeats 10 \
  --n-permutations 100 \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --seed 42 \
  --output results/aicl_delay_early_repeated_perm/aicl_delay_linearsvc_var0p95_rep10_perm100_early0_0p5.h5
