#!/bin/bash
#SBATCH --job-name=contrasts_nd
#SBATCH --output=logs/contrasts_nd_%A_%a.out
#SBATCH --error=logs/contrasts_nd_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger,coganlab-gpu
#SBATCH --array=0-44

source ~/.bashrc
conda activate ieeg

cd /hpc/home/ns458/coganlab/nanlinshi/insula

# Get all subjects from NoDelay epoch directory
SUBJECTS=($(ls -d /cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/derivatives/epoch\(bipolar\)/sub-D*/ | xargs -n1 basename | sed 's/sub-//'))

# Safety check
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SUBJECTS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID >= ${#SUBJECTS[@]} subjects"
    echo "Use --array=0-$((${#SUBJECTS[@]}-1))"
    exit 1
fi

SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
echo "Processing subject: $SUBJ (task $SLURM_ARRAY_TASK_ID / ${#SUBJECTS[@]})"

python src/univariate_contrasts.py \
    --bids_root /cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/ \
    --band highgamma \
    --n_perm 5000 \
    --subject "$SUBJ"
