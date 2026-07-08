#!/usr/bin/env python3
"""Extract MAPER native-space Hammersmith n30r95 (95-region) labels at
bipolar sEEG electrode coordinates, and roll them up into the Faillenot
six-region insula scheme (ASG/MSG/PSG/pole/ALG/PLG; even ID = left,
odd ID = right).

This is the GENERALIZED, subject-agnostic version of the script that was
first validated as extract_maper_insula_D0044_v2.py. It fixes the same two
bugs documented in docs/D44_MAPER_worklog.md:

BUG 1 (atlas ancillaries, fixed upstream in step 02/prepare_hammers_native_pairs.py):
    the 30 Hammersmith seg95 label volumes used as MAPER ancillaries must be
    the geometry-corrected copies (same voxel arrays as the raw download,
    header/affine rewritten to match the paired atlas T1). If you set up
    ancillaries/seg/seg95/ via step 02's setup_maper_ancillaries.sh this is
    already handled; if you bypass that script, you WILL get silently wrong
    fused segmentations.

BUG 2 (coordinate frame, handled here): BIDS *_space-ACPC_electrodes.tsv
    (or equivalent) x/y/z are FreeSurfer tkRAS coordinates, NOT the fused
    segmentation NIfTI's own scanner-space affine. Voxel indices must be
    computed as inv(vox2ras_tkr) @ [x,y,z,1], with vox2ras_tkr read from the
    subject's own orig.mgz (or any FreeSurfer volume sharing that subject's
    conformed 256^3 grid) -- NOT from the fused MAPER output's header, whose
    own affine may not reflect tkRAS. Verify vox2ras_tkr's translation is
    [128,-128,128] (volume-center convention) before trusting a new subject.
    Confirm independently by sampling aparc.a2009s+aseg.mgz at a known
    anatomical electrode with the same inv(vox2ras_tkr) transform and
    checking the returned Destrieux label matches clinical expectation.

Usage:
    python extract_maper_insula.py \
        --subject D0044 \
        --fused /cwork/ns458/maper_run/output/f30-seg95-D0044.nii.gz \
        --orig /cwork/ns458/ECoG_Recon/D44/mri/orig.mgz \
        --electrodes-csv path/to/bipolar_channel_coords.csv \
        --lut Hammers_mith_atlases_n30r95_label_indices_SPM12_20160111.txt \
        --out sub-D0044_desc-maper_insula.csv

The --electrodes-csv must have columns: name, hemi, x, y, z (tkRAS mm), and
may optionally carry other columns (e.g. roi, center) which are passed
through unchanged into the output.
"""
import argparse
import re
import numpy as np
import pandas as pd
import nibabel as nib


# Faillenot 6-region grouping within the Hammersmith 95-label scheme.
# Even ID = left hemisphere, odd ID = right hemisphere.
INSULA_IDS = [20, 21, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
REGION6 = {86: 'ASG', 87: 'ASG', 88: 'MSG', 89: 'MSG', 90: 'PSG', 91: 'PSG',
           92: 'pole', 93: 'pole', 94: 'ALG', 95: 'ALG', 20: 'PLG', 21: 'PLG'}
AP = {86: 'Anterior', 87: 'Anterior', 88: 'Anterior', 89: 'Anterior',
      90: 'Anterior', 91: 'Anterior', 92: 'Anterior', 93: 'Anterior',
      94: 'Posterior', 95: 'Posterior', 20: 'Posterior', 21: 'Posterior'}


def load_lut(path):
    lut = {}
    for line in open(path):
        m = re.search(r'<index>(\d+)</index><name>(.*?)</name>', line)
        if m:
            lut[int(m.group(1))] = m.group(2)
    return lut


def sample_exact(data, inv_tkr, x, y, z):
    v = inv_tkr @ np.array([x, y, z, 1.0])
    i, j, k = np.round(v[:3]).astype(int)
    if not (0 <= i < data.shape[0] and 0 <= j < data.shape[1] and 0 <= k < data.shape[2]):
        return 0
    return int(data[i, j, k])


def sample_sphere_insula(data, inv_tkr, x, y, z, r=2.0):
    """2mm-radius majority-vote fallback among insula labels, for electrodes
    that land in white matter / grey-white boundary at the exact voxel."""
    v = inv_tkr @ np.array([x, y, z, 1.0])
    ci, cj, ck = np.round(v[:3]).astype(int)
    rr = int(np.ceil(r))
    ids = []
    for di in range(-rr, rr + 1):
        for dj in range(-rr, rr + 1):
            for dk in range(-rr, rr + 1):
                if di * di + dj * dj + dk * dk > r * r:
                    continue
                i, j, k = ci + di, cj + dj, ck + dk
                if not (0 <= i < data.shape[0] and 0 <= j < data.shape[1] and 0 <= k < data.shape[2]):
                    continue
                lab = int(data[i, j, k])
                if lab in INSULA_IDS:
                    ids.append(lab)
    if not ids:
        return 0, 0.0, 0
    vals, cnts = np.unique(ids, return_counts=True)
    top = vals[np.argmax(cnts)]
    return int(top), float(cnts.max() / len(ids)), len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', required=True, help='e.g. D0044')
    ap.add_argument('--fused', required=True, help='fused MAPER seg95 NIfTI, e.g. f30-seg95-<SUBJECT>.nii.gz')
    ap.add_argument('--orig', required=True, help="subject's FreeSurfer orig.mgz (or any volume on the same conformed grid) -- source of vox2ras_tkr")
    ap.add_argument('--electrodes-csv', required=True, help='CSV with name,hemi,x,y,z (tkRAS mm) [+ passthrough columns]')
    ap.add_argument('--lut', required=True, help='Hammers_mith label-index LUT (xml-in-txt), region ID -> name')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sphere-radius', type=float, default=2.0)
    args = ap.parse_args()

    lut = load_lut(args.lut)

    fused_img = nib.load(args.fused)
    data = np.asanyarray(fused_img.dataobj).squeeze().astype(int)
    orig_img = nib.load(args.orig)
    assert orig_img.shape[:3] == data.shape[:3], (
        f"fused seg grid {data.shape[:3]} != orig.mgz grid {orig_img.shape[:3]} "
        "-- target T1 must be prepared on the same conformed grid (see step 03/prepare_target.sh)")
    inv_tkr = np.linalg.inv(orig_img.header.get_vox2ras_tkr())
    translation = orig_img.header.get_vox2ras_tkr()[:3, 3]
    if not np.allclose(np.abs(translation), np.array(data.shape[:3]) / 2.0, atol=2):
        print(f"WARNING: vox2ras_tkr translation {translation} is not close to "
              f"the volume-center convention for shape {data.shape[:3]} -- "
              "double check this subject's orig.mgz is a standard FreeSurfer conformed volume.")

    df = pd.read_csv(args.electrodes_csv)
    rows = []
    for _, r in df.iterrows():
        x, y, z = float(r['x']), float(r['y']), float(r['z'])
        ex = sample_exact(data, inv_tkr, x, y, z)
        sph_id, sph_frac, sph_n = sample_sphere_insula(data, inv_tkr, x, y, z, args.sphere_radius)
        row = dict(r)
        row.update(dict(
            maper_id=ex,
            maper_name=lut.get(ex, '(non-insula)' if ex else 'background'),
            maper_region6=REGION6.get(ex, ''),
            maper_ap=AP.get(ex, ''),
            maper_is_insula=ex in INSULA_IDS,
            maper_sph_id=sph_id,
            maper_sph_name=lut.get(sph_id, '') if sph_id else '',
            maper_sph_region6=REGION6.get(sph_id, ''),
            maper_sph_ap=AP.get(sph_id, ''),
            maper_sph_frac=round(sph_frac, 3),
            maper_sph_ninsula=sph_n,
        ))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"WROTE {args.out} rows={len(out)}")

    ins = out[out['maper_is_insula'] | (out['maper_sph_id'] > 0)]
    print(f"\n=== channels with any insula involvement: {len(ins)} ===")
    cols = [c for c in ['name', 'hemi', 'x', 'y', 'z', 'maper_region6', 'maper_ap',
                         'maper_sph_region6', 'maper_sph_frac'] if c in ins.columns]
    with pd.option_context('display.width', 200, 'display.max_columns', 40):
        print(ins[cols].to_string(index=False))


if __name__ == '__main__':
    main()
