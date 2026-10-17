#!/bin/bash
#SBATCH --job-name=audit_d0153
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/audit_d0153_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/audit_d0153_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/audit_grant_d0153.py
