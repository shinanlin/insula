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

SUBJECT_ARGS=(--subjects D0094 D0071 D0084)
if [[ "${HGA_EXPLORER_COHORT:-validation}" == "full" ]]; then
  SUBJECT_ARGS=(--all-subjects)
fi

mkdir -p "${PROJECT_ROOT}/logs/slurm"

source ~/.bashrc
conda activate ieeg

python "${VIEWER_ROOT}/export/compute_hga_explorer.py" \
  --input_root "${RESULTS_ROOT}" \
  --reference bipolar \
  --atlas all \
  --default_atlas hammers \
  --tasks PhonemeSequencing LexicalDelay \
  "${SUBJECT_ARGS[@]}" \
  --output_dir "${DATA_DIR}"

python "${VIEWER_ROOT}/scripts/qa_export.py" "${DATA_DIR}"

SUBJECT_COUNT="$(python -c "import json; print(len(json.load(open('${DATA_DIR}/manifest.json'))['metadata']['subjects']))")"
echo "Export complete: ${DATA_DIR}/manifest.json (${SUBJECT_COUNT} subjects)"
