#!/bin/bash
# Batch-prepare and submit MAPER Slurm array jobs for all subjects in the
# five-dataset union (see discover_maper_subjects.py).
#
# Each subject gets an isolated RUN directory:
#   /cwork/ns458/maper_run/<SUBJECT>/{target,output,launchlist_*.sh}
# Shared ancillaries are symlinked from /cwork/ns458/maper_run/ancillaries.
#
# Usage (login node):
#   module load FreeSurfer/7.2.0
#   bash pipeline/03_run_maper/batch_submit_maper.sh
#   bash pipeline/03_run_maper/batch_submit_maper.sh --dry-run
#   bash pipeline/03_run_maper/batch_submit_maper.sh --force
#
# After submission, monitor:
#   squeue -u $USER
#   ls /cwork/ns458/maper_run/*/output/f30-seg95-*.nii.gz | wc -l

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PIPE_STEP03="$SCRIPT_DIR"
DISCOVER="$SCRIPT_DIR/discover_maper_subjects.py"
PREPARE_TARGET="$PIPE_STEP03/prepare_target.sh"
GENERATE_LAUNCHLIST="$PIPE_STEP03/generate_launchlist.sh"
RUN_MAPER_SBATCH="$PIPE_STEP03/run_maper.sbatch"

SIF=/hpc/group/coganlab/nanlinshi/maper_tool/maper.sif
MAPER_BASE=/cwork/ns458/maper_run
SHARED_ANC="$MAPER_BASE/ancillaries"
RECON_ROOT=/cwork/ns458/ECoG_Recon
SUBJECT_LIST="$SCRIPT_DIR/maper_subjects_union.txt"
MANIFEST="$MAPER_BASE/batch_submit_manifest_$(date -u +%Y%m%dT%H%M%SZ).tsv"
LOG_DIR="$REPO_ROOT/logs"

# Duke module FreeSurfer lacks a cluster-wide license; use project copy.
if [[ -z "${FS_LICENSE:-}" ]]; then
    for candidate in \
        /cwork/ns458/ecog_recon/software/freesurfer/license.txt \
        /cwork/ns458/ecog_recon/software/freesurfer-8.1.0/license.txt; do
        if [[ -f "$candidate" ]]; then
            export FS_LICENSE="$candidate"
            break
        fi
    done
fi

DRY_RUN=0
FORCE=0

usage() {
    sed -n '2,20p' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run) DRY_RUN=1; shift ;;
        -f|--force) FORCE=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: missing container: $SIF" >&2
    exit 1
fi
if [[ ! -d "$SHARED_ANC/seg/seg95" ]]; then
    echo "ERROR: missing shared ancillaries: $SHARED_ANC" >&2
    exit 1
fi
if [[ "$DRY_RUN" -eq 0 ]] && ! command -v mri_convert >/dev/null 2>&1; then
    echo "ERROR: mri_convert not on PATH. Run: module load FreeSurfer/7.2.0" >&2
    exit 1
fi
if [[ "$DRY_RUN" -eq 0 && -z "${FS_LICENSE:-}" ]]; then
    echo "ERROR: FS_LICENSE not set and no license file found under /cwork/ns458/ecog_recon/software/" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "Discovering subject union across 5 BIDS datasets..."
python3 "$DISCOVER" --write "$SUBJECT_LIST" >/dev/null
mapfile -t SUBJECTS < <(grep -v '^[[:space:]]*$' "$SUBJECT_LIST")
echo "Subjects: ${#SUBJECTS[@]} (written to $SUBJECT_LIST)"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] Would write manifest to: $MANIFEST"
else
    printf 'subject\trun_dir\tjob_id\tstatus\n' > "$MANIFEST"
fi

submitted=0
skipped=0
failed=0

recon_mri_dir() {
    local subject=$1
    local num
    num=$(python3 -c "print(int('${subject}'.lstrip('D0')))")
    echo "$RECON_ROOT/D${num}/mri"
}

for SUBJECT in "${SUBJECTS[@]}"; do
    RUN="$MAPER_BASE/$SUBJECT"
    FUSED="$RUN/output/f30-seg95-${SUBJECT}.nii.gz"

    if [[ -f "$FUSED" && "$FORCE" -eq 0 ]]; then
        echo "SKIP $SUBJECT (already done: $FUSED; use --force to rerun)"
        skipped=$((skipped + 1))
        if [[ "$DRY_RUN" -eq 0 ]]; then
            printf '%s\t%s\t-\talready_done\n' "$SUBJECT" "$RUN" >> "$MANIFEST"
        fi
        continue
    fi

    RECON_MRI=$(recon_mri_dir "$SUBJECT")
    if [[ ! -f "$RECON_MRI/brainmask.mgz" ]]; then
        echo "ERROR $SUBJECT missing $RECON_MRI/brainmask.mgz" >&2
        failed=$((failed + 1))
        continue
    fi

    echo "PREP $SUBJECT -> $RUN"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  mkdir -p $RUN && ln -sfn $SHARED_ANC $RUN/ancillaries"
        echo "  $PREPARE_TARGET $SUBJECT $RECON_MRI $RUN/target"
        echo "  $GENERATE_LAUNCHLIST $SUBJECT $RUN $SIF"
        echo "  sbatch --export=SUBJECT=$SUBJECT,RUN=$RUN $RUN_MAPER_SBATCH"
        submitted=$((submitted + 1))
        continue
    fi

    mkdir -p "$RUN/output"
    ln -sfn "$SHARED_ANC" "$RUN/ancillaries"

    if ! bash "$PREPARE_TARGET" "$SUBJECT" "$RECON_MRI" "$RUN/target"; then
        echo "ERROR $SUBJECT prepare_target failed" >&2
        failed=$((failed + 1))
        continue
    fi
    if ! bash "$GENERATE_LAUNCHLIST" "$SUBJECT" "$RUN" "$SIF"; then
        echo "ERROR $SUBJECT generate_launchlist failed" >&2
        failed=$((failed + 1))
        continue
    fi

    job_id=$(sbatch --parsable \
        --export=ALL,SUBJECT="$SUBJECT",RUN="$RUN" \
        "$RUN_MAPER_SBATCH")
    echo "SUBMIT $SUBJECT job_id=$job_id"
    printf '%s\t%s\t%s\tsubmitted\n' "$SUBJECT" "$RUN" "$job_id" >> "$MANIFEST"
    submitted=$((submitted + 1))
done

echo ""
echo "=== batch_submit_maper summary ==="
echo "total subjects : ${#SUBJECTS[@]}"
echo "submitted      : $submitted"
echo "skipped        : $skipped"
echo "failed prep    : $failed"
if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "manifest       : $MANIFEST"
    echo "monitor        : squeue -u $USER"
    echo "progress       : ls $MAPER_BASE/*/output/f30-seg95-*.nii.gz | wc -l"
fi

if [[ "$failed" -gt 0 ]]; then
    exit 1
fi
