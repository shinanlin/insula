#!/bin/bash

#SBATCH --job-name=gen_within
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/gen_within_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/gen_within_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=40
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-8

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"
REF=bipolar
BAND=highgamma
DATATYPE=lexicality
VARIANCE=0.90

WINDOW=0.2
STEP=0.02
N_PERM=100
N_FOLDS=10
N_JOBS=40

# Define arrays
ROIS=(MFGl)
PHASES=(Stimulus Delay Go Response)
# WITHIN CONDITION PAIRS
COND_PAIRS=("Repeat Repeat" "Decision Decision")

# Build job parameters dynamically (4 x 2 x 1 = 8 combos)
PARAMS=()
for roi in "${ROIS[@]}"; do
    for phase in "${PHASES[@]}"; do
        for pair in "${COND_PAIRS[@]}"; do
            PARAMS+=("${roi} ${phase} ${pair}")
        done
    done
done

IDX=$SLURM_ARRAY_TASK_ID
IFS=' ' read -r ROI PHASE TRAIN TEST <<< "${PARAMS[$IDX]}"

echo "=== Within-Condition 2D Decoding ==="
echo "ROI=${ROI}, Phase=${PHASE}, Train=${TRAIN}, Test=${TEST}"

python src/run_cross_condition_generalized.py \
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