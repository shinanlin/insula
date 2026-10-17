#!/bin/bash
# Fixed-window IFGl articulator decoding for PhonemeSequence Repeat.
#SBATCH --job-name=decode_ifgl_art_ps
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/decode_ifgl_art_ps_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/decode_ifgl_art_ps_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-2

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
set -u

mkdir -p logs/slurm

PHASES=(Stimulus Delay Response)
PHASE="${PHASES[${SLURM_ARRAY_TASK_ID}]}"
BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"
INPUT_H5="${BIDS_ROOT}/derivatives/decoding(bipolar)/sub-IFGl/articulator/sub-IFGl_task-PhonemeSequence_proc-${PHASE}_recording-1_desc-Repeat_highgamma.h5"
OUTPUT_H5="results/decoding/PhonemeSequence/sub-IFGl/(decode)articulator/sub-IFGl_proc-${PHASE}_recording-1_desc-Repeat_highgamma.h5"

if [[ ! -f "${INPUT_H5}" ]]; then
  echo "Missing prepared input: ${INPUT_H5}" >&2
  exit 2
fi
if [[ -f "${OUTPUT_H5}" ]]; then
  echo "SKIP: existing output ${OUTPUT_H5}"
  exit 0
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u src/decoding/run_decoding.py \
  --bids_root "${BIDS_ROOT}" \
  --subject IFGl \
  --ref bipolar \
  --description Repeat \
  --phase "${PHASE}" \
  --band highgamma \
  --datatype articulator \
  --variance 0.95 \
  --n_perm 200 \
  --n_folds 5 \
  --n_repeats 30 \
  --n_jobs "${SLURM_CPUS_PER_TASK}"
