#!/bin/bash
#SBATCH --job-name=shi100_maper
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=1-210%20
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/shi100_maper_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/shi100_maper_%A_%a.err
set -euo pipefail
MANIFEST=${1:-/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/shi100_maper_20261005.tsv}
IFS=$'\t' read -r RECON MAPER_ID LAUNCH INDEX < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
INDEX=${INDEX//$'\r'/}
test -n "$RECON" && test -f "$LAUNCH"
CMD=$(sed -n "${INDEX}p" "$LAUNCH")
[[ "$CMD" == '/opt/maper/maper '* ]]
SIF=/hpc/group/coganlab/nanlinshi/maper_tool/maper.sif
export APPTAINER_TMPDIR
APPTAINER_TMPDIR=$(mktemp -d "/tmp/shi100_${SLURM_JOB_ID}_XXXXXX")
mkdir -p /cwork/ns458/maper_run/shi100_scratch
MAPER_TMP=$(mktemp -d "/cwork/ns458/maper_run/shi100_scratch/task_${SLURM_JOB_ID}_XXXXXX")
trap 'rm -rf "$APPTAINER_TMPDIR" "$MAPER_TMP"' EXIT
echo "recon=$RECON maper=$MAPER_ID atlas=$INDEX host=$(hostname) start=$(date -u)"
apptainer exec -B /cwork/ns458 --env "TMPDIR=$MAPER_TMP" "$SIF" bash -c "$CMD"
echo "completed recon=$RECON atlas=$INDEX at $(date -u)"
