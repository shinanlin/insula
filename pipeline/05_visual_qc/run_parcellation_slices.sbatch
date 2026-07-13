#!/bin/bash
#SBATCH --job-name=parc_slices
#SBATCH --partition=common
#SBATCH --account=coganlab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/parc_slices_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/parc_slices_%j.err
#
# Stage 3 parcellation slice QC for one subject.
# Example:
#   sbatch --export=SUBJ=D0094,BIDS=/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS \
#     pipeline/05_visual_qc/run_parcellation_slices.sbatch

set -o pipefail

SUBJ="${SUBJ:?set SUBJ, e.g. D0094}"
BIDS="${BIDS:?set BIDS root}"
ATLAS="${ATLAS:-hammers}"

INSULA=/hpc/group/coganlab/nanlinshi/insula
CSV="${BIDS}/derivatives/parcellation/sub-${SUBJ}/bipolar/sub-${SUBJ}_${ATLAS}.csv"
FUSED="/cwork/ns458/maper_run/${SUBJ}/output/f30-seg95-${SUBJ}.nii.gz"

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd "$INSULA"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-parc_slices}"
mkdir -p "$MPLCONFIGDIR" /hpc/group/coganlab/nanlinshi/insula/logs/slurm

echo "host=$(hostname) subj=${SUBJ} atlas=${ATLAS} csv=${CSV} start=$(date -u)"

python pipeline/05_visual_qc/plot_parcellation_slices.py \
  --parcellation-csv "$CSV" \
  --atlas "$ATLAS" \
  --recon-dir /cwork/ns458/ECoG_Recon \
  --fused "$FUSED"
rc=$?
echo "subj=${SUBJ} exit=$rc end=$(date -u)"
exit $rc
