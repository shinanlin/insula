#!/bin/bash
# Build the MAPER "ancillaries" tree from the geometry-corrected Hammers
# n30r95 labels (run prepare_hammers_native_pairs.py FIRST — this script
# assumes geometry_corrected_labels/ already exists and is correct).
#
# This is a ONE-TIME step shared across all subjects: it does not depend
# on the MAPER target, so the resulting ancillaries/ directory is reused
# unchanged for every subsequent subject.
#
# Ancillaries layout expected by MAPER (see src-description.csv):
#   onepad/aN.nii.gz     -- atlas T1 (padded to a common voxel grid)
#   posnorm/aN.dof.gz    -- pre-transformation (registration IC) to that grid
#   seg/seg95/aN.nii.gz  -- 95-region label volume, GEOMETRY-CORRECTED (Bug 1 fix)
#
# onepad/ and posnorm/ come from MAPER's own hammers_mith-ancillaries.sh
# (downloads a companion tarball of pre-normalized atlas T1s + registrations
# from the MAPER project) — run that INSIDE the container first, then this
# script overwrites only seg/seg95/ with the geometry-corrected labels.
#
# Usage:
#   apptainer exec --bind /cwork/ns458 maper.sif \
#       /opt/maper/hammers_mith-ancillaries.sh $ATLAS_DOWNLOAD_DIR $ANCILLARIES_DIR
#   ./setup_maper_ancillaries.sh <ATLAS_ROOT> <ANCILLARIES_DIR>
#
# Example:
#   ./setup_maper_ancillaries.sh /cwork/ns458/atlases/Hammersmith_n30r95 /cwork/ns458/maper_run/ancillaries

set -euo pipefail
ATLAS_ROOT=$1        # e.g. /cwork/ns458/atlases/Hammersmith_n30r95 (must already have derivatives/individual_native_pairs/geometry_corrected_labels/)
ANCILLARIES_DIR=$2   # e.g. /cwork/ns458/maper_run/ancillaries (must already contain onepad/ + posnorm/ from hammers_mith-ancillaries.sh)

CORRECTED="$ATLAS_ROOT/derivatives/individual_native_pairs/geometry_corrected_labels"
[[ -d "$CORRECTED" ]] || { echo "ERROR: run prepare_hammers_native_pairs.py first (missing $CORRECTED)"; exit 1; }

mkdir -p "$ANCILLARIES_DIR/seg/seg95"

for i in $(seq 1 30); do
    aa=$(printf '%02d' "$i")
    src="$CORRECTED/a${aa}_labels_r95_geometry-fixed.nii.gz"
    dst="$ANCILLARIES_DIR/seg/seg95/a${i}.nii.gz"
    [[ -f "$src" ]] || { echo "ERROR: missing $src"; exit 1; }
    cp "$src" "$dst"
done

echo "Replaced $ANCILLARIES_DIR/seg/seg95/a{1..30}.nii.gz with geometry-corrected labels."
echo "Verify affine match against onepad/ T1s before trusting a run, e.g.:"
echo "  python3 -c \"import nibabel as nib,numpy as np; a=nib.load('$ANCILLARIES_DIR/onepad/a1.nii.gz'); b=nib.load('$ANCILLARIES_DIR/seg/seg95/a1.nii.gz'); print(np.allclose(a.affine,b.affine,atol=1e-3))\""
