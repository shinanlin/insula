#!/bin/bash
#SBATCH --job-name=xcorr_insula
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=common
#SBATCH --chdir=/hpc/home/ns458/coganlab/nanlinshi/insula
#SBATCH --output=/hpc/home/ns458/coganlab/nanlinshi/insula/logs/xcorr_%A_%a.out
#SBATCH --error=/hpc/home/ns458/coganlab/nanlinshi/insula/logs/xcorr_%A_%a.err
#SBATCH --array=0-4

# Adjust environment activation as needed
source ~/.bashrc
conda activate ieeg

BIDS_ROOTS=(
  "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
  "/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/"
  "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/"
  "/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/"
  "/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/"
)

BIDS_ROOT=${BIDS_ROOTS[$SLURM_ARRAY_TASK_ID]}

python src/xcorr/run_xcorr.py \
  --bids_root "$BIDS_ROOT" \
  --band highgamma \
  --reference bipolar \
  --recon_dir /cwork/ns458/ECoG_Recon/
