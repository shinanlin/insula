#!/bin/bash
#SBATCH --job-name=xconcat
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/xconcat_%j.out
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd /hpc/group/coganlab/nanlinshi/insula-functional
python scripts/export_Xconcat.py
