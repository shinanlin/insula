#!/bin/bash
#SBATCH --job-name=nmf_3phase_rank
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/nmf_3phase_rank_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/nmf_3phase_rank_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common
#SBATCH --chdir=/cwork/ns458/insula-paper

set -euo pipefail

WORKTREE=/cwork/ns458/insula-paper
SOURCE_RESULTS=/hpc/group/coganlab/nanlinshi/insula-functional/results
[[ "$(pwd -P)" == "$WORKTREE" ]] || { echo "Wrong working directory" >&2; exit 2; }
for output in "$WORKTREE/results" "$WORKTREE/results/nmf" "$WORKTREE/img" "$WORKTREE/img/nmf"; do
  [[ ! -L "$output" ]] || { echo "Refusing symlinked output: $output" >&2; exit 2; }
done
[[ -f "$SOURCE_RESULTS/nmf/exclude_channels.txt" ]] || { echo "Missing frozen exclusion list" >&2; exit 2; }

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
mkdir -p "$WORKTREE/logs/slurm" "$WORKTREE/results/nmf" "$WORKTREE/img/nmf"

python scripts/run_nmf_rank_selection.py \
  --results-root "$SOURCE_RESULTS" \
  --phases stimulus delay response \
  --k-min 2 --k-max 6 \
  --n-boot 200 --row-frac 0.8 \
  --max-iter 5000 --random-state 42 \
  --exclude-channels-file "$SOURCE_RESULTS/nmf/exclude_channels.txt" \
  --results-dir "$WORKTREE/results/nmf" \
  --images-dir "$WORKTREE/img/nmf"
