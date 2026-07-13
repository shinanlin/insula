#!/bin/bash
#SBATCH --job-name=parc_slices_all
#SBATCH --partition=common
#SBATCH --account=coganlab
#SBATCH --array=1-89%20
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/parc_slices_all_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/parc_slices_all_%A_%a.err
#
# Stage 3 parcellation slice QC for all subjects (deduplicated manifest).
# Manifest: scripts/slurm/parcellation_qc_manifest.tsv
#   columns: <bids_root> <TAB> <subject>

set -o pipefail

INSULA=/hpc/group/coganlab/nanlinshi/insula
MANI="$INSULA/scripts/slurm/parcellation_qc_manifest.tsv"
ATLAS="${ATLAS:-hammers}"

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANI")
BIDS=$(printf '%s' "$line" | cut -f1)
SUBJ=$(printf '%s' "$line" | cut -f2)
CSV="${BIDS}/derivatives/parcellation/sub-${SUBJ}/bipolar/sub-${SUBJ}_${ATLAS}.csv"
FUSED="/cwork/ns458/maper_run/${SUBJ}/output/f30-seg95-${SUBJ}.nii.gz"

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd "$INSULA"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-parc_slices_all}"
mkdir -p "$MPLCONFIGDIR" "$INSULA/logs/slurm"

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} subj=${SUBJ} bids=${BIDS} start=$(date -u)"

if [[ ! -f "$CSV" ]]; then
  echo "missing parcellation CSV: $CSV" >&2
  exit 2
fi
if [[ "$ATLAS" == "hammers" && ! -f "$FUSED" ]]; then
  echo "missing MAPER fused volume: $FUSED" >&2
  exit 3
fi

if [[ "$ATLAS" == "hammers" ]]; then
  python pipeline/05_visual_qc/plot_parcellation_slices.py \
    --parcellation-csv "$CSV" \
    --atlas "$ATLAS" \
    --recon-dir /cwork/ns458/ECoG_Recon \
    --fused "$FUSED"
else
  python pipeline/05_visual_qc/plot_parcellation_slices.py \
    --parcellation-csv "$CSV" \
    --atlas "$ATLAS" \
    --recon-dir /cwork/ns458/ECoG_Recon
fi
rc=$?
echo "task=${SLURM_ARRAY_TASK_ID} subj=${SUBJ} exit=$rc end=$(date -u)"
exit $rc
