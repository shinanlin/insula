#!/bin/bash
#SBATCH --job-name=hga_native_insula
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/hga_native_insula_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/hga_native_insula_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger

set -eo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/export/export_native_insula_brain_mesh.py" ]]; then
  VIEWER_ROOT="${SLURM_SUBMIT_DIR}"
else
  VIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
PROJECT_ROOT="$(cd "${VIEWER_ROOT}/../.." && pwd)"

mkdir -p "${PROJECT_ROOT}/logs/slurm"

source ~/.bashrc
conda activate ieeg

python "${VIEWER_ROOT}/export/export_native_insula_brain_mesh.py" --from-index

echo "Native insula cohort export complete."
