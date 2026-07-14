#!/bin/bash

#SBATCH --job-name=cross_cond_gen
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/cross_cond_gen_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/cross_cond_gen_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=40
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-8

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

# =====================================================================
# 2D Cross-condition generalized decoding (8 jobs)
# ROIs: AICl, PICl, SMCl, STGl
# Phases: Delay, Response
# Directions: Train=Repeat & Test=Decision, Train=Decision & Test=Repeat
# Params: window=0.3, step=0.03, perm=100, folds=10, jobs=40
# Note: Maximize CPUs to speed up inner temporal loops
# =====================================================================

BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"
REF=bipolar
BAND=highgamma
DATATYPE=lexicality
VARIANCE=0.80

WINDOW=0.3
STEP=0.03
N_PERM=100
N_FOLDS=10
N_JOBS=40

# Define arrays
ROIS=(dACCl)
PHASES=(Stimulus Delay Go Response)
# CROSS CONDITION DIRECTIONS
DIRECTIONS=("Decision Repeat" "Repeat Decision")

# Build job parameters dynamically (1 ROIs x 4 Phases x 2 Directions = 8 combos)
PARAMS=()
for roi in "${ROIS[@]}"; do
    for phase in "${PHASES[@]}"; do
        for dir in "${DIRECTIONS[@]}"; do
            PARAMS+=("${roi} ${phase} ${dir}")
        done
    done
done

# Get current job config
IDX=$SLURM_ARRAY_TASK_ID
IFS=' ' read -r ROI PHASE TRAIN TEST <<< "${PARAMS[$IDX]}"

echo "=== 2D Generalized Decoding ==="
echo "ROI=${ROI}, Phase=${PHASE}, Train=${TRAIN}, Test=${TEST}"
echo "Window=${WINDOW}s, Step=${STEP}s"
echo "Jobs=${N_JOBS}, Perm=${N_PERM}, Folds=${N_FOLDS}"

python src/decoding/run_cross_condition_generalized.py \
    --bids_root "${BIDS_ROOT}" \
    --roi "${ROI}" \
    --phase "${PHASE}" \
    --train_on "${TRAIN}" \
    --test_on "${TEST}" \
    --ref "${REF}" \
    --band "${BAND}" \
    --datatype "${DATATYPE}" \
    --variance ${VARIANCE} \
    --window ${WINDOW} \
    --step ${STEP} \
    --n_perm ${N_PERM} \
    --n_folds ${N_FOLDS} \
    --n_jobs ${N_JOBS}

echo "Exit code: $?"
echo "=== Done ==="
