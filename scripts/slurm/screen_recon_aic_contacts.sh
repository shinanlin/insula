#!/bin/bash
# Temporary SHI-100 anatomy-only contact screen. MAPER must exist beforehand.
# Usage: sbatch scripts/slurm/screen_recon_aic_contacts.sh [script arguments]
# Exit 2 means unresolved inputs/contacts; read summary.csv, not an AIC-negative result.
#SBATCH --job-name=aic_contacts
#SBATCH --partition=common
#SBATCH --account=coganlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/aic_contacts_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/aic_contacts_%j.err

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
set -u
cd /hpc/group/coganlab/nanlinshi/insula-functional
if [[ $# -eq 0 ]]; then
    set -- --subjects D148 D149 D150 D152A D152B D153 D154 \
        --output "/cwork/ns458/insula_recon_screen/job_${SLURM_JOB_ID}"
fi
python -u scripts/screen_recon_aic_contacts.py "$@"
