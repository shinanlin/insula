#!/bin/bash
#SBATCH --job-name=item_leak_stg
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/item_leak_stg_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/item_leak_stg_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-3

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TQDM_DISABLE=1

ROI="${ROI:-STGl}"
DATATYPE="${DATATYPE:-lexicality}"
BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"

PHASES=(Stimulus Stimulus Delay Delay)
DESCS=(Repeat Decision Repeat Decision)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
PHASE=${PHASES[$IDX]}
DESC=${DESCS[$IDX]}

CPUS="${SLURM_CPUS_PER_TASK:-16}"

echo "array=${IDX} roi=${ROI} datatype=${DATATYPE} phase=${PHASE} desc=${DESC}"
echo "bids_root=${BIDS_ROOT} n_jobs=${CPUS}"
echo "Python: $(which python) ($(python --version 2>&1))"

python -u src/decoding/diagnose_item_leakage.py \
  --bids_root "${BIDS_ROOT}" \
  --roi "${ROI}" \
  --datatype "${DATATYPE}" \
  --phase "${PHASE}" \
  --description "${DESC}" \
  --n_jobs "${CPUS}"

echo "Exit code: $?"
