#!/bin/bash
#SBATCH --job-name=nmf_inspector_build
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_inspector_build_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_inspector_build_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional/viewer/nmf_inspector

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
npm install
npm run build
echo "Built dist/ for NMF Inspector"
