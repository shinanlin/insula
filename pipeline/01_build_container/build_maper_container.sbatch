#!/bin/bash
#SBATCH --job-name=maper_build
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/maper_build_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/maper_build_%j.err

# One-time step. Only re-run this if maper.sif is missing/corrupted or the
# container definition changes; the built .sif is reused across all subjects.

set -euo pipefail
echo "host: $(hostname)  start: $(date -u)"

export APPTAINER_TMPDIR=/tmp/${USER}_aptmp_$SLURM_JOB_ID
export APPTAINER_CACHEDIR=/tmp/${USER}_apcache_$SLURM_JOB_ID
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
trap 'rm -rf "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"' EXIT

DEF=/hpc/group/coganlab/nanlinshi/insula/scripts/maper_container.def
SIF=/hpc/group/coganlab/nanlinshi/maper_tool/maper.sif

echo "building $SIF from $DEF (built-in fakeroot shim, glibc base) ..."
apptainer build --force "$SIF" "$DEF"

echo "=== running %test ==="
apptainer test "$SIF" || true
echo "done: $(date -u)  sif: $(ls -lh $SIF)"
