#!/bin/bash
# Temporary rerun: PhonemeSequence articulator decoding with RBF SVM.
# AICl + AICr × 4 phases = 8 array tasks (left-batch hyperparameters).
#SBATCH --job-name=decode_aic_art_rbf
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/decoding_resolved_phoneme_aic_articulator_rbf_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/decoding_resolved_phoneme_aic_articulator_rbf_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-7

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/"
PHASES=(Stimulus Delay Go Response)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${TASK_ID}" -lt 4 ]; then
  SUBJECT=AICl
  PHASE_IDX="${TASK_ID}"
else
  SUBJECT=AICr
  PHASE_IDX=$((TASK_ID - 4))
fi
PHASE="${PHASES[$PHASE_IDX]}"

N_JOBS="${SLURM_CPUS_PER_TASK:-16}"

echo "array=${TASK_ID}/7"
echo "Combination: subject=${SUBJECT} band=highgamma desc=Repeat datatype=articulator phase=${PHASE}"
echo "bids_root=${BIDS_ROOT}"
echo "n_jobs=${N_JOBS} cpus=${SLURM_CPUS_PER_TASK}"
echo "Current working directory: $(pwd)"
echo "Python: $(which python) ($(python --version 2>&1))"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

python -u src/decoding/run_decoding_resolved.py \
  --bids_root "${BIDS_ROOT}" \
  --subject "${SUBJECT}" \
  --ref bipolar \
  --description Repeat \
  --phase "${PHASE}" \
  --band highgamma \
  --datatype articulator \
  --variance 0.95 \
  --window 0.3 \
  --step 0.03 \
  --n_perm 200 \
  --n_folds 5 \
  --n_repeats 30 \
  --n_jobs "${N_JOBS}"

echo "Exit code: $?"
