#!/bin/bash
# Prepare left/right IFG lexicality pools for LexicalDelay (SHI-59).
# Hammers group is exact "IFG" only (no IFGs, no mix labels).
# Input policy matches STGl/SMCl/MFGl: zscore trials + Decision∪Repeat sig union.
#SBATCH --job-name=prep_decode_ifg
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/prepare_decoding_ifg_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/prepare_decoding_ifg_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"
TASK_DIR="/hpc/group/coganlab/nanlinshi/seeg-preprocessing-worktrees/lexical_delay/task"
mkdir -p logs/slurm

echo "Preparing IFG lexicality (Hammers IFG only, zscore+sig_union)"
python -u "${TASK_DIR}/prepare_decoding_dataset.py" \
    --bids_root "${BIDS_ROOT}" \
    --task lexicality \
    --band highgamma \
    --rois IFG \
    --reference bipolar \
    --atlas hammers \
    --input_datatype "epoch(band)(zscore)+sig_union"

echo "Done. Decoding uses sub-IFGl only."
