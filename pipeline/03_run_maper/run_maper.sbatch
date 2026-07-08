#!/bin/bash
#SBATCH --job-name=maper
#SBATCH --partition=common
#SBATCH --array=1-30%15
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/maper_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/maper_%A_%a.err
#
# Run 30 atlas->target label-propagation tasks (one per Hammersmith n30r95
# atlas), then MAPER auto-triggers label fusion once all 30 have written
# their output. Per-task wall time ~4 min, ~1.6GB peak RSS; whole run for
# one subject ~10 min end-to-end including fusion.
#
# Set SUBJECT before submitting, e.g.:
#   sbatch --export=SUBJECT=D0044,RUN=/cwork/ns458/maper_run scripts/run_maper.sbatch
# or edit the two variables below directly.

set -uo pipefail
SUBJECT=${SUBJECT:-D0044}
RUN=${RUN:-/cwork/ns458/maper_run}
SIF=/hpc/group/coganlab/nanlinshi/maper_tool/maper.sif
LL="$RUN/launchlist_${SUBJECT}.sh"

# node-local scratch for MAPER temp working dirs
export APPTAINER_TMPDIR=/tmp/${USER}_maper_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$APPTAINER_TMPDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LL")
echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} start=$(date -u)"
echo "CMD: $CMD" | cut -c1-300

# run inside container; TMPDIR inside points to node-local /tmp
apptainer exec -B /cwork/ns458 --env TMPDIR=/tmp "$SIF" bash -c "$CMD"
rc=$?
echo "task=${SLURM_ARRAY_TASK_ID} exit=$rc end=$(date -u)"
exit $rc
