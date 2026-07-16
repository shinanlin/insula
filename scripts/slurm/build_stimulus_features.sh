#!/bin/bash
#SBATCH --job-name=build_stim_feat
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm/build_stim_feat_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm/build_stim_feat_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-semantic

mkdir -p /hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm
mkdir -p /hpc/group/coganlab/nanlinshi/insula-semantic/src/semantic/features

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

python src/semantic/build_stimulus_features.py
