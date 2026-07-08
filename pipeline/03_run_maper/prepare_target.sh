#!/bin/bash
# Prepare the per-subject MAPER "target" T1 from a FreeSurfer recon.
#
# MAPER's -tgtmri wants a single skull-stripped T1 NIfTI. We reuse the
# FreeSurfer brainmask.mgz (already skull-stripped, same 256^3 conformed
# grid as orig.mgz) instead of re-running any extraction — this keeps the
# MAPER target voxel grid IDENTICAL to orig.mgz/brainmask.mgz/aparc+aseg,
# which step 04 (label extraction) depends on for its inv(vox2ras_tkr)
# coordinate transform.
#
# Usage: ./prepare_target.sh <SUBJECT_ID e.g. D0044> <RECON_MRI_DIR> <OUT_DIR>
# Example:
#   ./prepare_target.sh D0044 /cwork/ns458/ECoG_Recon/D44/mri /cwork/ns458/maper_run/target

set -euo pipefail
SUBJECT=$1   # e.g. D0044 (BIDS-style, used as MAPER -tgtid)
RECON_MRI=$2 # e.g. /cwork/ns458/ECoG_Recon/D44/mri  (contains orig.mgz, brainmask.mgz)
OUTDIR=$3    # e.g. /cwork/ns458/maper_run/target

mkdir -p "$OUTDIR"
OUT_NII="$OUTDIR/sub-${SUBJECT}_T1w_brain.nii.gz"

# mri_convert requires FreeSurfer on PATH (conda env `ieeg` or FreeSurfer module)
mri_convert "$RECON_MRI/brainmask.mgz" "$OUT_NII"

cat > "$OUTDIR/tgt-description.csv" <<EOF
id, mri
${SUBJECT}, $(basename "$OUT_NII")
EOF

echo "Wrote $OUT_NII"
echo "Wrote $OUTDIR/tgt-description.csv"
echo "NOTE: verify with 'python3 -c \"import nibabel as nib; print(nib.load(\\\"$OUT_NII\\\").shape)\"' that shape/affine match $RECON_MRI/orig.mgz exactly (should be 256^3, LIA)."
