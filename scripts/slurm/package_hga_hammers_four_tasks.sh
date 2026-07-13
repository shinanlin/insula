#!/bin/bash
#SBATCH --job-name=pkg_hga_ham4
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/package_hga_hammers_four_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/package_hga_hammers_four_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

run_one() {
  local bids_root="$1"
  local name="$2"
  echo "===== ${name} (hammers) ====="
  echo "bids_root=${bids_root} start=$(date -u)"
  python src/hga/package_highgamma.py \
    --bids_root "${bids_root}" \
    --band highgamma \
    --ref bipolar \
    --atlas hammers
  echo "===== ${name} done exit=$? end=$(date -u) ====="
}

run_one "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/" "LexicalDelay"
run_one "/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/" "LexicalNoDelay"
run_one "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/" "PhonemeSequence"
run_one "/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/" "PictureNaming"

echo "All four tasks (hammers) done."
