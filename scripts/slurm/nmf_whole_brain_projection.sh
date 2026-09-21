#!/bin/bash
#SBATCH --job-name=nmf_wb_proj
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_whole_brain_projection_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_whole_brain_projection_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg

mkdir -p logs/slurm results/nmf/whole_brain_projection img/nmf/whole_brain_projection

echo "=== Whole-brain fixed-H projection: all Repeat HGA subjects × 4 NMF tasks ==="
python scripts/run_nmf_whole_brain_projection.py \
  --exclude-subject D0121 \
  --tasks PhonemeSequence LexicalDelay PictureNaming SentenceRep \
  --phases stimulus delay go response \
  --significance-mode mask-any \
  --min-coverage 0.95 \
  --min-tasks 1

echo "Done: results/nmf/whole_brain_projection/ and img/nmf/whole_brain_projection/"
