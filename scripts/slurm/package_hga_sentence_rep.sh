#!/bin/bash
#SBATCH --job-name=pkg_sent_sentence
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/package_hga_sentence_rep_sentence_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/package_hga_sentence_rep_sentence_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs/slurm

BIDS_ROOT="/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/"

echo "===== Package SentenceRep(sentence) HGA ====="
echo "bids_root=${BIDS_ROOT} start=$(date -u)"
python src/hga/package_highgamma.py \
  --bids_root "${BIDS_ROOT}" \
  --band highgamma \
  --ref bipolar \
  --atlas hammers \
  --family sentence

echo "===== Verification ====="
ROOT="results/hga/SentenceRep(sentence)"
echo "subjects: $(find "${ROOT}" -maxdepth 1 -type d -name 'sub-*' | wc -l)"
echo "total time CSVs: $(find "${ROOT}" -name '*_time.csv' | wc -l)"
echo "Repeat time CSVs: $(find "${ROOT}" -name '*desc-Repeat_time.csv' | wc -l)"
for PH in Stimulus Delay Go Response; do
  echo "  Repeat ${PH}: $(find "${ROOT}" -name "*proc-${PH}*desc-Repeat_time.csv" | wc -l)"
done
echo "Passive time CSVs: $(find "${ROOT}" -name '*desc-Passive_time.csv' | wc -l)"
echo "done end=$(date -u)"
