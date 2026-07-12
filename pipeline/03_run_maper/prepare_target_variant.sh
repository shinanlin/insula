#!/bin/bash
# Prepare a per-subject MAPER target from a selected FreeSurfer MRI volume.

set -euo pipefail

SUBJECT=$1
RECON_MRI=$2
OUTDIR=$3
VARIANT=${4:-brainmask}

case "$VARIANT" in
    brainmask) SOURCE="brainmask.mgz" ;;
    brain) SOURCE="brain.mgz" ;;
    brain_finalsurfs) SOURCE="brain.finalsurfs.mgz" ;;
    T1) SOURCE="T1.mgz" ;;
    orig) SOURCE="orig.mgz" ;;
    *)
        echo "Unknown target variant: $VARIANT" >&2
        exit 1
        ;;
esac

INPUT="$RECON_MRI/$SOURCE"
if [[ ! -f "$INPUT" ]]; then
    echo "Missing target source for $SUBJECT $VARIANT: $INPUT" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
OUT_NII="$OUTDIR/sub-${SUBJECT}_T1w_brain.nii.gz"
mri_convert "$INPUT" "$OUT_NII"

cat > "$OUTDIR/tgt-description.csv" <<EOF
id, mri
${SUBJECT}, $(basename "$OUT_NII")
EOF

cat > "$OUTDIR/target_variant.tsv" <<EOF
subject	variant	source	output
${SUBJECT}	${VARIANT}	${INPUT}	${OUT_NII}
EOF

echo "Wrote $OUT_NII from $INPUT"
echo "Wrote $OUTDIR/tgt-description.csv"
