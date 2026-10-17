#!/bin/bash
#SBATCH --job-name=pkg_coord_ham4
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/package_electrode_coords_hammers_four_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/package_electrode_coords_hammers_four_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=coganlab-gpu
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs/slurm

run_one() {
  local bids_root="$1"
  local name="$2"
  echo "===== ${name} electrode coords (hammers, all zscore channels) ====="
  echo "bids_root=${bids_root} start=$(date -u)"
  python -m src.hga.package_ave_cord \
    --bids_root "${bids_root}" \
    --band highgamma \
    --ref bipolar \
    --atlas hammers
  echo "===== ${name} done exit=$? end=$(date -u) ====="
}

run_one "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/" "LexicalDelay"
run_one "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/" "PhonemeSequence"
run_one "/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/" "PictureNaming"
run_one "/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/" "SentenceRep"

echo "===== Verification ====="
for TASK in LexicalDelay PhonemeSequence PictureNaming SentenceRep; do
  ROOT="results/hga/${TASK}"
  echo "${TASK}: subjects=$(find "${ROOT}" -maxdepth 1 -type d -name 'sub-*' 2>/dev/null | wc -l) coord_csv=$(find "${ROOT}" -name '*_coord.csv' 2>/dev/null | wc -l)"
done

echo "All four NMF tasks (hammers electrode coords) done."
