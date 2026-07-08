#!/bin/bash
#SBATCH --job-name=maper_extract
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/maper_extract_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/maper_extract_%A_%a.err

set -euo pipefail

: "${MANIFEST:?MANIFEST must point to a ready-only TSV}"
SCRIPT_DIR=/hpc/group/coganlab/nanlinshi/insula/pipeline/04_extract_labels
PYTHON=/hpc/home/ns458/miniconda3/envs/ieeg/bin/python
LUT="$SCRIPT_DIR/Hammers_mith_atlases_n30r95_label_indices_SPM12_20160111.txt"

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [[ -z "$line" ]]; then
    echo "No manifest row for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

IFS=$'\t' read -r task bids_root subject parcellation contacts fused tissue propagated orig output sensitivity status <<< "$line"
if [[ "$status" != "ready" ]]; then
    echo "Manifest row is not ready: $subject $task $status" >&2
    exit 1
fi

"$PYTHON" "$SCRIPT_DIR/extract_maper_parcellation.py" \
    --task "$task" \
    --subject "$subject" \
    --fused "$fused" \
    --tissue "$tissue" \
    --orig "$orig" \
    --propagated-dir "$propagated" \
    --parcellation-csv "$parcellation" \
    --contacts-tsv "$contacts" \
    --lut "$LUT" \
    --output "$output" \
    --sensitivity-output "$sensitivity"
