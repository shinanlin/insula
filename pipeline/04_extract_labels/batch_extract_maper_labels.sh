#!/bin/bash
# Build the task-specific extraction manifest and optionally submit its array.
# This step reuses existing MAPER fusion/propagation outputs; it never runs
# registration or label fusion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON=/hpc/home/ns458/miniconda3/envs/ieeg/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
MANIFEST_DIR=/cwork/ns458/maper_run/manifests
ALL_MANIFEST="$MANIFEST_DIR/maper_extract_all_${STAMP}.tsv"
READY_MANIFEST="$MANIFEST_DIR/maper_extract_ready_${STAMP}.tsv"
SUBMIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit) SUBMIT=1; shift ;;
        -n|--dry-run) SUBMIT=0; shift ;;
        -h|--help) sed -n '1,12p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$MANIFEST_DIR" "$REPO_ROOT/logs"
"$PYTHON" "$SCRIPT_DIR/build_extraction_manifest.py" \
    --output "$ALL_MANIFEST" \
    --ready-output "$READY_MANIFEST"

ready_count=$(( $(wc -l < "$READY_MANIFEST") - 1 ))
echo "Ready combinations: $ready_count"
echo "All manifest: $ALL_MANIFEST"
echo "Ready manifest: $READY_MANIFEST"

if [[ "$SUBMIT" -eq 0 ]]; then
    echo "Dry run only. Add --submit to submit the extraction array."
    exit 0
fi
if [[ "$ready_count" -lt 1 ]]; then
    echo "No ready combinations" >&2
    exit 1
fi

job_id=$(sbatch --parsable \
    --array="1-${ready_count}%20" \
    --export=ALL,MANIFEST="$READY_MANIFEST" \
    "$SCRIPT_DIR/run_extract_maper_labels.sbatch")
echo "Submitted job $job_id"
echo -e "job_id\tmanifest\tcombinations" > "$MANIFEST_DIR/maper_extract_job_${job_id}.tsv"
echo -e "${job_id}\t${READY_MANIFEST}\t${ready_count}" >> "$MANIFEST_DIR/maper_extract_job_${job_id}.tsv"
