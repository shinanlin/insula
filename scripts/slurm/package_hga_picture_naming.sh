#!/bin/bash
#SBATCH --job-name=pkg_picture_naming
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/package_picture_naming_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/package_picture_naming_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenger,common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula

set -eo pipefail
source ~/.bashrc
conda activate ieeg

PICTURE_BIDS="/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/"

echo "===== Package PictureNaming HGA (hammers) ====="
python -m src.hga.package_highgamma \
  --bids_root "${PICTURE_BIDS}" \
  --band highgamma \
  --ref bipolar \
  --atlas hammers

echo "===== Package PictureNaming HGA (aparc2009s) ====="
python -m src.hga.package_highgamma \
  --bids_root "${PICTURE_BIDS}" \
  --band highgamma \
  --ref bipolar \
  --atlas aparc2009s

echo "===== Done ====="
find results/PictureNaming\(bipolar\)\(hammers\) -name '*_time.csv' | wc -l
find results/PictureNaming\(bipolar\)\(aparc2009s\) -name '*_time.csv' | wc -l
