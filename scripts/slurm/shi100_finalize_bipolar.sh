#!/bin/bash
#SBATCH --job-name=shi100_finalize
#SBATCH --partition=common
#SBATCH --account=coganlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/shi100_finalize_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/shi100_finalize_%j.err
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
set -eo pipefail
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/recon_bipolar/finalize.py --staging "$1" --output "$2"
