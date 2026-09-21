#!/bin/bash
#SBATCH --job-name=smoke_decode_func
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/smoke_decode_func_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/smoke_decode_func_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-2%3

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

COMMON=(
  --bids_root /cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS
  --subject Sensoryl --ref bipolar --description Repeat --phase Delay
  --band highgamma --datatype lexicality --variance 0.90
  --n_perm 2 --n_folds 2 --n_repeats 1 --n_jobs 8
)

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -u src/decoding/run_decoding.py "${COMMON[@]}"
    ;;
  1)
    python -u src/decoding/run_decoding_resolved.py "${COMMON[@]}" --window 0.30 --step 0.30
    ;;
  2)
    python -u src/decoding/run_cross_condition_generalized.py \
      --bids_root /cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS \
      --roi Sensoryl --phase Delay --train_on Repeat --test_on Decision \
      --ref bipolar --band highgamma --datatype lexicality --variance 0.80 \
      --window 0.30 --step 0.30 --n_perm 2 --n_folds 2 --n_jobs 8
    ;;
esac
