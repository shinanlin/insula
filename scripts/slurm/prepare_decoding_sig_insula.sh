#!/bin/bash
# Repackage decoding datasets for insula ROIs using epoch(band)(sig).
# LexicalDelay: AIC+PIC, tasks phoneme/articulator/lexicality
# PhonemeSequence: AIC+PIC, tasks phoneme/articulator
#SBATCH --job-name=prep_decode_sig_insula
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/prepare_decoding_sig_insula_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/prepare_decoding_sig_insula_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-3%4

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

BAND=highgamma
REF=bipolar
ATLAS=hammers
INPUT_DATATYPE="epoch(band)(sig)"

# 0-1: LexicalDelay AIC/PIC; 2-3: PhonemeSequence AIC/PIC
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"
    TASK_DIR="/hpc/group/coganlab/nanlinshi/seeg-preprocessing-worktrees/lexical_delay/task"
    ROI=AIC
    TASKS=(phoneme articulator lexicality)
    ;;
  1)
    BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"
    TASK_DIR="/hpc/group/coganlab/nanlinshi/seeg-preprocessing-worktrees/lexical_delay/task"
    ROI=PIC
    TASKS=(phoneme articulator lexicality)
    ;;
  2)
    BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"
    TASK_DIR="/hpc/group/coganlab/nanlinshi/seeg-preprocessing-worktrees/phoneme_seq/task"
    ROI=AIC
    TASKS=(phoneme articulator)
    ;;
  3)
    BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"
    TASK_DIR="/hpc/group/coganlab/nanlinshi/seeg-preprocessing-worktrees/phoneme_seq/task"
    ROI=PIC
    TASKS=(phoneme articulator)
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "Repackaging ROI=${ROI} tasks=${TASKS[*]} bids_root=${BIDS_ROOT}"
echo "input_datatype=${INPUT_DATATYPE}"

for TASK in "${TASKS[@]}"; do
    echo "Running task: ${TASK} for ROI: ${ROI}"
    python "$TASK_DIR/prepare_decoding_dataset.py" \
        --bids_root "$BIDS_ROOT" \
        --task "$TASK" \
        --band "$BAND" \
        --rois "$ROI" \
        --reference "$REF" \
        --atlas "$ATLAS" \
        --input_datatype "$INPUT_DATATYPE"
done

echo "Completed repackaging for ROI ${ROI}"
