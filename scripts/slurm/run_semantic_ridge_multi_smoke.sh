#!/bin/bash
#SBATCH --job-name=sem_multi_smoke
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm/sem_multi_smoke_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm/sem_multi_smoke_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-semantic

# Smoke: D0092 Delay×Decision, all 4 multi-block models, n_perm=50

mkdir -p /hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

SUBJECT=D0092
PHASE=Delay
DESCRIPTION=Decision
MODELS=(semantic phon acoustic full_perm_semantic)

for MODEL in "${MODELS[@]}"; do
    echo "=== ${SUBJECT} ${PHASE}/${DESCRIPTION} model=${MODEL} ==="
    python src/semantic/run_encoding_multi.py \
        --subject "${SUBJECT}" \
        --phase "${PHASE}" \
        --description "${DESCRIPTION}" \
        --model "${MODEL}" \
        --n_perm 50 \
        --n_jobs 2
done
