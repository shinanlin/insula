#!/bin/bash

#SBATCH --job-name=cross_roi_res
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/cross_roi_res_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/cross_roi_res_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=40
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-23

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

# =====================================================================
# Cross-ROI resolved decoding with CCA alignment
# Train ROI -> Test ROI within the same phase and same condition
# Params: window=0.2, step=0.02, perm=100, folds=10, jobs=40
# =====================================================================

BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"
REF=bipolar
BAND=highgamma
DATATYPE=lexicality
VARIANCE=0.80
N_COMPONENTS=5

WINDOW=0.2
STEP=0.02
N_PERM=100
N_FOLDS=10
N_JOBS=40

# Define arrays
ROI_PAIRS=("AICl SMCl" "AICl STGl" "AICl IFGl")
PHASES=(Stimulus Delay Go Response)
CONDITIONS=(Repeat Decision)

# Build job parameters dynamically (ROI pairs x phases x conditions)
PARAMS=()
for pair in "${ROI_PAIRS[@]}"; do
    for phase in "${PHASES[@]}"; do
        for condition in "${CONDITIONS[@]}"; do
            PARAMS+=("${pair} ${phase} ${condition}")
        done
    done
done

# Get current job config
IDX=$SLURM_ARRAY_TASK_ID
IFS=' ' read -r TRAIN_ROI TEST_ROI PHASE DESCRIPTION <<< "${PARAMS[$IDX]}"

echo "=== Cross-ROI Resolved Decoding ==="
echo "Train ROI=${TRAIN_ROI}, Test ROI=${TEST_ROI}"
echo "Phase=${PHASE}, Description=${DESCRIPTION}"
echo "Window=${WINDOW}s, Step=${STEP}s"
echo "Jobs=${N_JOBS}, Perm=${N_PERM}, Folds=${N_FOLDS}"

python src/run_cross_roi_resolved.py \
    --bids_root "${BIDS_ROOT}" \
    --train_roi "${TRAIN_ROI}" \
    --test_roi "${TEST_ROI}" \
    --phase "${PHASE}" \
    --description "${DESCRIPTION}" \
    --ref "${REF}" \
    --band "${BAND}" \
    --datatype "${DATATYPE}" \
    --variance ${VARIANCE} \
    --n_components ${N_COMPONENTS} \
    --window ${WINDOW} \
    --step ${STEP} \
    --n_perm ${N_PERM} \
    --n_folds ${N_FOLDS} \
    --n_jobs ${N_JOBS}

echo "Exit code: $?"
echo "=== Done ==="
