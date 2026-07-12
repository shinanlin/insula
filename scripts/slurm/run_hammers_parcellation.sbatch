#!/bin/bash
#SBATCH --job-name=ham_parc
#SBATCH --partition=common
#SBATCH --account=coganlab
#SBATCH --array=1-180%40
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/ham_parc_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/ham_parc_%A_%a.err
#
# Generate Hammersmith (MAPER seg95, GM-gated) bipolar parcellation for every
# runnable (task, subject) pair. One array task per manifest line.
# Manifest columns: <bids_root> <TAB> <subject>.
# Recalibration subjects (D0040/D0106/D0121) are intentionally absent from the
# manifest; run them separately once their MAPER recalibration completes.

set -o pipefail
# NB: no `set -u` — conda activate scripts reference unbound vars (ADDR2LINE).
REPO=/hpc/group/coganlab/nanlinshi/seeg-preprocessing
MANI=/hpc/group/coganlab/nanlinshi/insula/scripts/slurm/hammers_parcellation_manifest.tsv

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANI")
BIDS=$(printf '%s' "$line" | cut -f1)
SUBJ=$(printf '%s' "$line" | cut -f2)
echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} subj=${SUBJ} bids=${BIDS} start=$(date -u)"

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd "$REPO"

python common/parcellation.py \
  --bids_root "$BIDS" \
  --recon_dir /cwork/ns458/ECoG_Recon \
  --subject "$SUBJ" \
  --atlas hammers \
  --ref bipolar
rc=$?
echo "task=${SLURM_ARRAY_TASK_ID} subj=${SUBJ} exit=$rc end=$(date -u)"
exit $rc
