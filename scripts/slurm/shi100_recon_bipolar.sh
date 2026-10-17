#!/bin/bash
#SBATCH --job-name=shi100_recon_bipolar
#SBATCH --partition=common
#SBATCH --account=coganlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/shi100_recon_bipolar_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/shi100_recon_bipolar_%j.err
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/recon_bipolar/test_anatomy.py
recon_staging="/cwork/ns458/insula_recon_screen/shi100_staging_${SLURM_JOB_ID}"
python -u scripts/recon_bipolar/run.py --output "$recon_staging"
python -u scripts/recon_bipolar/validate.py "$recon_staging"
python -u scripts/recon_bipolar/finalize.py --staging "$recon_staging" --output "$1"
