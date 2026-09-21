#!/bin/bash
#SBATCH --job-name=k3_heldout
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/k3_heldout.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/k3_heldout.err
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
mkdir -p logs
python scripts/export_k3_heldout_persubject.py
echo "DONE $?"
