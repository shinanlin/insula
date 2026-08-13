#!/bin/bash
#SBATCH --job-name=insula_rt_smoke
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/insula_rt_smoke_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/insula_rt_smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

export TASK="${TASK:-LexicalDelay}"
export SUBJECT_OVERRIDE="${SUBJECT_OVERRIDE:-D0096}"
export N_PERM="${N_PERM:-20}"
export MAX_WINDOWS="${MAX_WINDOWS:-3}"
export N_JOBS="${SLURM_CPUS_PER_TASK:-4}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/hpc/group/coganlab/nanlinshi/insula-functional/results/rt/smoke}"
bash scripts/slurm/run_insula_rt_ridge.sh
