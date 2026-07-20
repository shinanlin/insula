#!/bin/bash
# Univariate contrasts: LexicalNoDelay (hammers), ~26 subjects.
#SBATCH --job-name=univ_nodelay
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/univariate_nodelay_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/univariate_nodelay_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-25%5

source ~/.bashrc
conda activate ieeg

PROJECT_ROOT="/hpc/group/coganlab/nanlinshi/insula"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${PROJECT_ROOT}/logs/slurm"

BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/"
SUBJECTS=($(ls -d "${BIDS_ROOT}derivatives/epoch(bipolar)/sub-D"*/ | xargs -n1 basename | sed 's/sub-//'))

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SUBJECTS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID >= ${#SUBJECTS[@]} subjects"
    echo "Use --array=0-$((${#SUBJECTS[@]}-1))"
    exit 1
fi

SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
echo "Processing subject: $SUBJ (task $SLURM_ARRAY_TASK_ID / ${#SUBJECTS[@]})"

python src/univariate/contrasts.py \
    --bids_root "$BIDS_ROOT" \
    --band highgamma \
    --n_perm 5000 \
    --subject "$SUBJ"
