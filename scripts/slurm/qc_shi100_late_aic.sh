#!/bin/bash
#SBATCH --job-name=shi100_late_qc
#SBATCH --partition=common
#SBATCH --account=coganlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/shi100_late_qc_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/shi100_late_qc_%j.err
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -u scripts/qc_shi100_late_aic.py
