#!/bin/bash
#SBATCH --job-name=cross_contrast
#SBATCH --output=logs/cross_contrast_%A_%a.out
#SBATCH --error=logs/cross_contrast_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common,scavenger,coganlab-gpu
#SBATCH --array=0-44%5

source ~/.bashrc
conda activate ieeg

cd /hpc/home/ns458/coganlab/nanlinshi/insula

DELAY_DIR="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/epoch(bipolar)"
NODELAY_DIR="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/derivatives/epoch(bipolar)"

# Subjects who have Delay
DELAY_SUBJECTS=($(ls -d ${DELAY_DIR}/sub-D*/ | xargs -n1 basename | sed 's/sub-//' | sort))

# Subjects who have NoDelay
NODELAY_SUBJECTS=($(ls -d ${NODELAY_DIR}/sub-D*/ | xargs -n1 basename | sed 's/sub-//' | sort))

# Common subjects between Delay and NoDelay
COMMON_SUBJECTS=($(comm -12 \
    <(printf "%s\n" "${DELAY_SUBJECTS[@]}" | sort) \
    <(printf "%s\n" "${NODELAY_SUBJECTS[@]}" | sort)
))

if [ "${#COMMON_SUBJECTS[@]}" -eq 0 ]; then
    echo "ERROR: no common subjects found between Delay and NoDelay"
    exit 1
fi

# Safety check
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#COMMON_SUBJECTS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID >= ${#COMMON_SUBJECTS[@]} common subjects"
    echo "Use --array=0-$((${#COMMON_SUBJECTS[@]}-1))"
    exit 1
fi

SUBJ=${COMMON_SUBJECTS[$SLURM_ARRAY_TASK_ID]}
echo "Processing subject: $SUBJ (task $SLURM_ARRAY_TASK_ID / ${#COMMON_SUBJECTS[@]})"

python src/cross_task_contrast.py \
    --subject "$SUBJ" \
    --band highgamma \
    --n_perm 5000
