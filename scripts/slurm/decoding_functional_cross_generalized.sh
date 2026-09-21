#!/bin/bash
#SBATCH --job-name=func_cross_gen
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/func_cross_gen_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/func_cross_gen_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=40
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-2%3

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

SUBJECTS=(Sensory Sustain Motor)
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

python -u src/decoding/run_cross_condition_generalized.py \
  --bids_root "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS" \
  --roi "${SUBJECT}" \
  --phase Delay \
  --train_on Repeat \
  --test_on Decision \
  --ref bipolar \
  --band highgamma \
  --datatype lexicality \
  --variance 0.80 \
  --window 0.30 \
  --step 0.03 \
  --n_perm 100 \
  --n_folds 10 \
  --n_jobs "${SLURM_CPUS_PER_TASK:-40}"
