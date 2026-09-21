#!/bin/bash
#SBATCH --job-name=lda_summary
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_summary_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_summary_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#
# Run after decoding_resolved_lda_lexicality.sh completes.

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

python -u src/decoding/summarize_resolved_lda.py \
  --task LexicalDelay \
  --datatype lexicality \
  --band highgamma \
  --rois Sensory Sustain Motor STGl \
  --phases Stimulus Delay \
  --descriptions Repeat Decision
