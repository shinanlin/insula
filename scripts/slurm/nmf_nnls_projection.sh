#!/bin/bash
#SBATCH --job-name=nmf_nnls_proj
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_nnls_projection_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_nnls_projection_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${USER}_nmf_nnls"
mkdir -p "$NUMBA_CACHE_DIR" logs/slurm results/nmf/nnls_projection img/nmf

echo "=== NNLS projection: LexicalDelay × Stimulus Delay Go Response ==="
python scripts/run_nmf_nnls_projection.py \
  --task LexicalDelay \
  --phases Stimulus Delay Go Response

echo "Done: results/nmf/nnls_projection/ and img/nmf/nnls_H_overview_LexicalDelay_*.svg"
