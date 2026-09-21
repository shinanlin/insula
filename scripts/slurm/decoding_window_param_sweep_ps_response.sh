#!/bin/bash
#SBATCH --job-name=ps_win_sweep
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/ps_win_sweep_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/ps_win_sweep_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-191%20

bash scripts/window_decode_param_sweep_worker.sh
