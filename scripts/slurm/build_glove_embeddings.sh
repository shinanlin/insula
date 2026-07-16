#!/bin/bash
#SBATCH --job-name=build_glove_emb
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm/build_glove_emb_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm/build_glove_emb_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-semantic

mkdir -p /hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm
mkdir -p /hpc/group/coganlab/nanlinshi/cache/embeddings/glove
mkdir -p /hpc/group/coganlab/nanlinshi/cache/huggingface

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export HF_HOME=/hpc/group/coganlab/nanlinshi/cache/huggingface
export HF_HUB_CACHE=/hpc/group/coganlab/nanlinshi/cache/huggingface/hub

python src/semantic/build_embeddings.py
