#!/bin/bash

#SBATCH --job-name=cross_window
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/cross_window_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/cross_window_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-15

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

# =====================================================================
# Window-level cross-condition decoding (16 jobs)
# ROIs: AICl, PICl, SMCl, STGl
# Phases: Delay, Response
# Directions: Train=Repeat & Test=Decision, Train=Decision & Test=Repeat
# Params: perm=100, folds=10, jobs=40
# =====================================================================

BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"
REF=bipolar
BAND=highgamma
DATATYPE=lexicality
VARIANCE=0.8

N_PERM=200
N_FOLDS=12
N_JOBS=8

# Define arrays
ROIS=(AICl PICl SMCl STGl)
PHASES=(Delay Response)
# Define condition pairs (Train Test)
COND_PAIRS=("Repeat Decision" "Decision Repeat")
# Build job parameters dynamically (4 x 2 x 2 = 16 combos)
PARAMS=()
for roi in "${ROIS[@]}"; do
    for phase in "${PHASES[@]}"; do
        for pair in "${COND_PAIRS[@]}"; do
            PARAMS+=("${roi} ${phase} ${pair}")
        done
    done
done

# Get current job config
IDX=$SLURM_ARRAY_TASK_ID
IFS=' ' read -r ROI PHASE TRAIN TEST <<< "${PARAMS[$IDX]}"

echo "=== Window decoding ==="
echo "ROI=${ROI}, Phase=${PHASE}, Train=${TRAIN}, Test=${TEST}"
echo "Jobs=${N_JOBS}, Perm=${N_PERM}, Folds=${N_FOLDS}"

python src/run_cross_condition_window.py \
    --bids_root "${BIDS_ROOT}" \
    --roi "${ROI}" \
    --phase "${PHASE}" \
    --train_on "${TRAIN}" \
    --test_on "${TEST}" \
    --ref "${REF}" \
    --band "${BAND}" \
    --datatype "${DATATYPE}" \
    --variance ${VARIANCE} \
    --n_perm ${N_PERM} \
    --n_folds ${N_FOLDS} \
    --n_jobs ${N_JOBS}

echo "Exit code: $?"
echo "=== Done ==="
