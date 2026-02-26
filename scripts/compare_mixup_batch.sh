#!/bin/bash

#SBATCH --job-name=mixup_compare
#SBATCH --output=/hpc/home/ns458/coganlab/nanlinshi/insula/logs/mixup_compare.out
#SBATCH --error=/hpc/home/ns458/coganlab/nanlinshi/insula/logs/mixup_compare.err
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenger

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
module purge

cd /hpc/home/ns458/coganlab/nanlinshi/insula

echo "Starting mixup batch comparison"
echo "Python: $(which python)"
echo "Time: $(date)"

python scripts/compare_mixup_batch.py \
    --rois STGl SMCl PICl AICl \
    --n_folds 10 \
    --n_repeats 2 \
    --n_permutations 50

echo "Finished at $(date)"
echo "Exit code: $?"
