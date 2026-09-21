#!/bin/bash
#SBATCH -p common
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 0:20:00
#SBATCH -o logs/export_X.out
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
python scripts/export_X.py
echo "DONE $?"
