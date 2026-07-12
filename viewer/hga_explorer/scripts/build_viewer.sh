#!/bin/bash
#SBATCH --job-name=hga_viewer_build
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/hga_viewer_build_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/hga_viewer_build_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger

set -eo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/package.json" ]]; then
  VIEWER_ROOT="${SLURM_SUBMIT_DIR}"
else
  VIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "${VIEWER_ROOT}"
npm ci
npm run build
python scripts/qa_export.py public/data
