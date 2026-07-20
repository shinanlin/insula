#!/bin/bash
#SBATCH --job-name=hga_explorer_full
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/hga_explorer_full_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/hga_explorer_full_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula/viewer/hga_explorer

export HGA_EXPLORER_COHORT=full

if [[ -f "${SLURM_SUBMIT_DIR}/viewer/hga_explorer/scripts/build_data.sh" ]]; then
  VIEWER_ROOT="${SLURM_SUBMIT_DIR}/viewer/hga_explorer"
elif [[ -f "${SLURM_SUBMIT_DIR}/scripts/build_data.sh" ]]; then
  VIEWER_ROOT="${SLURM_SUBMIT_DIR}"
else
  VIEWER_ROOT="/hpc/group/coganlab/nanlinshi/insula/viewer/hga_explorer"
fi

bash "${VIEWER_ROOT}/scripts/build_data.sh"

