#!/bin/bash
#SBATCH --job-name=pkg_hga_pn
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/package_hga_picture_naming_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/package_hga_picture_naming_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs/slurm

PICTURE_BIDS="/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/"
ROOT="results/hga/PictureNaming"

echo "===== PictureNaming (hammers, sig-union) ====="
echo "bids_root=${PICTURE_BIDS} start=$(date -u)"
python src/hga/package_highgamma.py \
  --bids_root "${PICTURE_BIDS}" \
  --band highgamma \
  --ref bipolar \
  --atlas hammers
echo "===== PictureNaming done exit=$? end=$(date -u) ====="

echo "===== Verification ====="
echo "subjects=$(find "${ROOT}" -maxdepth 1 -type d -name 'sub-*' 2>/dev/null | wc -l)"
echo "Passive_Response_csv=$(find "${ROOT}" -name '*proc-Response*desc-Passive_time.csv' 2>/dev/null | wc -l)"
echo "Passive_sound_Response_csv=$(find "${ROOT}" -name '*proc-Response_recording-sound_desc-Passive_time.csv' 2>/dev/null | wc -l)"
echo "Repeat_Response_csv=$(find "${ROOT}" -name '*proc-Response*desc-Repeat_time.csv' 2>/dev/null | wc -l)"
