#!/bin/bash
#SBATCH --job-name=D0125_maper
#SBATCH --partition=common
#SBATCH --account=coganlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=1-30%15
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/maper_D0125_%A_%a.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/maper_D0125_%A_%a.err
set -euo pipefail
RUN=/cwork/ns458/maper_run/restore_20261017/D0125
SIF=/hpc/group/coganlab/nanlinshi/maper_tool/maper.sif
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUN/launchlist_D0125.sh")
[[ "$CMD" == '/opt/maper/maper '* ]]
export APPTAINER_TMPDIR
APPTAINER_TMPDIR=$(mktemp -d "/tmp/D0125_maper_${SLURM_JOB_ID}_XXXXXX")
mkdir -p "$RUN/scratch"
MAPER_TMP=$(mktemp -d "$RUN/scratch/atlas_${SLURM_ARRAY_TASK_ID}_XXXXXX")
trap 'rm -rf "$APPTAINER_TMPDIR" "$MAPER_TMP"' EXIT
export OPENBLAS_NUM_THREADS=1
echo "D0125 atlas=${SLURM_ARRAY_TASK_ID} host=$(hostname) start=$(date -u)"
apptainer exec -B /cwork/ns458 --env "TMPDIR=$MAPER_TMP" "$SIF" bash -c "$CMD"
echo "D0125 atlas=${SLURM_ARRAY_TASK_ID} finished=$(date -u)"
