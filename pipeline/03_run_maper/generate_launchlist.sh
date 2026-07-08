#!/bin/bash
# Generate the per-atlas-pairing MAPER command list for one subject.
# Wraps MAPER's own `launchlist-gen` tool (inside the container).
#
# Usage: ./generate_launchlist.sh <SUBJECT_ID> <RUN_DIR> <SIF_PATH>
# Example:
#   ./generate_launchlist.sh D0044 /cwork/ns458/maper_run /hpc/group/coganlab/nanlinshi/maper_tool/maper.sif
#
# Requires under RUN_DIR:
#   target/tgt-description.csv, target/sub-<SUBJECT>_T1w_brain.nii.gz  (step 03 prepare_target.sh)
#   ancillaries/src-description.csv, ancillaries/{onepad,posnorm,seg/seg95}/  (step 02)
# Writes:
#   RUN_DIR/launchlist_<SUBJECT>.sh   (30 lines, one `maper` invocation per atlas)

set -euo pipefail
SUBJECT=$1
RUN_DIR=$2
SIF=$3

OUT="$RUN_DIR/launchlist_${SUBJECT}.sh"

apptainer exec -B /cwork/ns458 "$SIF" /opt/maper/launchlist-gen \
    -src-base "$RUN_DIR/ancillaries" \
    -src-description "$RUN_DIR/ancillaries/src-description.csv" \
    -tgt-base "$RUN_DIR/target" \
    -tgt-description "$RUN_DIR/target/tgt-description.csv" \
    -output-dir "$RUN_DIR/output" \
    -launchlist "$OUT" \
    -threads 4

N=$(wc -l < "$OUT")
echo "Wrote $OUT ($N lines; expect 30 for the full n30r95 atlas set)"
