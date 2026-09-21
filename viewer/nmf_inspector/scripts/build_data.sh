#!/bin/bash
#SBATCH --job-name=nmf_inspector_export
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_inspector_export_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_inspector_export_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional/viewer/nmf_inspector

set -eo pipefail

# Prefer --chdir / known absolute path; BASH_SOURCE is unreliable under Slurm spool copies.
if [[ -f "./export/export_nmf_inspector.py" ]]; then
  VIEWER_ROOT="$(pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/export/export_nmf_inspector.py" ]]; then
  VIEWER_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/viewer/nmf_inspector/export/export_nmf_inspector.py" ]]; then
  VIEWER_ROOT="${SLURM_SUBMIT_DIR}/viewer/nmf_inspector"
else
  VIEWER_ROOT="/hpc/group/coganlab/nanlinshi/insula-functional/viewer/nmf_inspector"
fi
PROJECT_ROOT="$(cd "${VIEWER_ROOT}/../.." && pwd)"
DATA_DIR="${VIEWER_ROOT}/public/data"
ASSETS_DIR="${VIEWER_ROOT}/public/assets"

mkdir -p "${PROJECT_ROOT}/logs/slurm" "${ASSETS_DIR}" "${DATA_DIR}"

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Reuse insula mesh from sibling hga_explorer or main insula checkout
for SRC in \
  "${VIEWER_ROOT}/../hga_explorer/public/assets" \
  "/hpc/group/coganlab/nanlinshi/insula/viewer/hga_explorer/public/assets"; do
  if [[ -f "${SRC}/cvs_avg35_insula_pial.glb" ]]; then
    ASSET_SRC="${SRC}"
    break
  fi
done
if [[ -z "${ASSET_SRC:-}" ]]; then
  echo "ERROR: insula mesh assets not found (cvs_avg35_insula_pial.glb)" >&2
  exit 1
fi

for f in cvs_avg35_insula_pial.glb cvs_avg35_insula.meta.json cvs_avg35_pial_insula_mask.json; do
  ln -sf "${ASSET_SRC}/${f}" "${ASSETS_DIR}/${f}"
done

python "${VIEWER_ROOT}/export/export_nmf_inspector.py" \
  --output_dir "${DATA_DIR}"

echo "Export complete: ${DATA_DIR}/manifest.json"
