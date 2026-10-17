#!/bin/bash
#SBATCH --job-name=ccep_insula_locations
#SBATCH --partition=common
#SBATCH --account=coganlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/ccep_insula_locations_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/ccep_insula_locations_%j.err
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u scripts/qc_ccep_insula_locations.py
