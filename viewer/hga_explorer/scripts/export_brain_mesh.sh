#!/bin/bash
#SBATCH --job-name=hga_explorer_mesh
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/hga_explorer_mesh_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/hga_explorer_mesh_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger

set -eo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/export/export_average_brain_mesh.py" ]]; then
  VIEWER_ROOT="${SLURM_SUBMIT_DIR}"
else
  VIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
PROJECT_ROOT="$(cd "${VIEWER_ROOT}/../.." && pwd)"

mkdir -p "${PROJECT_ROOT}/logs/slurm"

source ~/.bashrc
conda activate ieeg

python "${VIEWER_ROOT}/export/export_average_brain_mesh.py"
python "${VIEWER_ROOT}/export/export_native_brain_mesh.py" \
  --subjects D0094 D0071 D0084 D0023 D0024 D0028 D0029 D0035 D0042 D0053 D0054 D0055 D0057 D0059 \
  D0063 D0066 D0068 D0069 D0070 D0077 D0079 D0086 D0096 D0100 D0102 D0103

echo "Brain mesh export complete."
