#!/bin/bash
#SBATCH --job-name=repack_phoneme_hga
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/repack_phoneme_hga_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/repack_phoneme_hga_%j.err
#SBATCH --time=14:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

PHONEME_BIDS="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/"
VIEWER_ROOT="/hpc/group/coganlab/nanlinshi/insula/viewer/hga_explorer"

echo "===== Repackage PhonemeSequencing HGA with stats alias fix ====="
python src/hga/package_highgamma.py \
  --bids_root "${PHONEME_BIDS}" \
  --band highgamma \
  --ref bipolar

echo "===== Rebuild HGA Explorer (full cohort) ====="
export HGA_EXPLORER_COHORT=full
bash "${VIEWER_ROOT}/scripts/build_data.sh"

echo "All done."
