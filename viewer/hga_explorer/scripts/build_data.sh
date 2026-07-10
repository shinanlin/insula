#!/bin/bash
#SBATCH --job-name=hga_explorer_export
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/hga_explorer_export_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/hga_explorer_export_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger

set -eo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/export/compute_hga_explorer.py" ]]; then
  VIEWER_ROOT="${SLURM_SUBMIT_DIR}"
else
  VIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
PROJECT_ROOT="$(cd "${VIEWER_ROOT}/../.." && pwd)"
DATA_DIR="${VIEWER_ROOT}/public/data"
RESULTS_ROOT="${PROJECT_ROOT}/results(nw)"

FULL_COHORT_SUBJECTS=(
  D0023 D0024 D0028 D0029 D0035 D0042 D0053 D0054 D0055 D0057 D0059
  D0063 D0066 D0068 D0069 D0070 D0071 D0077 D0079 D0084 D0086 D0094
  D0096 D0100 D0102 D0103
)

if [[ "${HGA_EXPLORER_COHORT:-validation}" == "full" ]]; then
  SUBJECTS=("${FULL_COHORT_SUBJECTS[@]}")
else
  SUBJECTS=(D0094 D0071 D0084)
fi

mkdir -p "${PROJECT_ROOT}/logs/slurm"

source ~/.bashrc
conda activate ieeg

python "${VIEWER_ROOT}/export/compute_hga_explorer.py" \
  --input_root "${RESULTS_ROOT}" \
  --reference bipolar \
  --tasks PhonemeSequencing LexicalDelay \
  --subjects "${SUBJECTS[@]}" \
  --output_dir "${DATA_DIR}"

python "${VIEWER_ROOT}/scripts/qa_export.py" "${DATA_DIR}"

echo "Export complete: ${DATA_DIR}/manifest.json (${#SUBJECTS[@]} subjects)"
