#!/bin/bash
# Phase-window mean univariate: PictureNaming sound Repeat vs Passive.
#SBATCH --job-name=univ_rvp_pn
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/univariate_mean_rvp_pn_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/univariate_mean_rvp_pn_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-36%5

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

PROJECT_ROOT="/hpc/group/coganlab/nanlinshi/insula-functional"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MNE_DONTWRITE_HOME=true
mkdir -p "${PROJECT_ROOT}/logs/slurm"

BIDS_ROOT="/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/"
mapfile -t SUBJECTS < <(
    ls -d "${BIDS_ROOT}derivatives/epoch(bipolar)/sub-D"*/ \
        | xargs -n1 basename \
        | sed 's/sub-//' \
        | grep -vx D0121 \
        | sort
)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SUBJECTS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID >= ${#SUBJECTS[@]} subjects"
    echo "Use --array=0-$((${#SUBJECTS[@]}-1))"
    exit 1
fi

SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
echo "Processing subject: $SUBJ (task $SLURM_ARRAY_TASK_ID / ${#SUBJECTS[@]})"

python src/univariate/contrasts_mean.py \
    --bids_root "$BIDS_ROOT" \
    --band highgamma \
    --n_perm 5000 \
    --contrasts RepeatVsPassive \
    --recording sound \
    --subject "$SUBJ"
