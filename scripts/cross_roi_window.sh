#!/bin/bash

#SBATCH --job-name=cross_roi_win
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/cross_roi_win_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/cross_roi_win_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=40
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-31

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

# Cross-ROI window decoding with CCA (Aim 3.3)
# 32 jobs = 2 tasks x 4 test ROIs x 4 phases (Repeat only)

REF=bipolar
BAND=highgamma
TRAIN_ROI=AICl
TEST_ROIS=(IFGl STGl MFGl SMCl)
PHASES=(Stimulus Delay Go Response)
DESCRIPTION=Repeat

VARIANCE=0.80
N_COMPONENTS=5
N_PERM=100
N_FOLDS=10
N_JOBS=40

CONFIGS=(
    "LexicalDelay lexicality /cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"
    "PhonemeSequence articulator /cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"
)
PARAMS=()
for cfg in "${CONFIGS[@]}"; do
    read -r TASK DATATYPE BIDS_ROOT <<< "$cfg"
    for TEST_ROI in "${TEST_ROIS[@]}"; do
        for PHASE in "${PHASES[@]}"; do
            PARAMS+=("${TASK} ${DATATYPE} ${BIDS_ROOT} ${TEST_ROI} ${PHASE}")
        done
    done
done

IDX=$SLURM_ARRAY_TASK_ID
IFS=' ' read -r TASK DATATYPE BIDS_ROOT TEST_ROI PHASE <<< "${PARAMS[$IDX]}"

echo "=== Cross-ROI Window Decoding ==="
echo "Task=${TASK} datatype=${DATATYPE}"
echo "Train=${TRAIN_ROI} Test=${TEST_ROI} Phase=${PHASE} Desc=${DESCRIPTION}"
echo "Perm=${N_PERM} Folds=${N_FOLDS} Jobs=${N_JOBS}"

python src/run_cross_roi_window.py \
    --task "${TASK}" \
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
    --n_perm ${N_PERM} \
    --n_folds ${N_FOLDS} \
    --n_jobs ${N_JOBS}

echo "Exit code: $?"
