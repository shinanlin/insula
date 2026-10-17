"""Native MRI QC: closest D157 contacts to AIC and D158 AIC-positive contacts.

Distances are to gray-matter atlas voxel centers, not cortical boundaries.
This is a QC supplement; it never changes the exact-point screening labels.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from screen_recon_aic_contacts import load_pipeline, UPSTREAM

REPO = Path('/hpc/group/coganlab/nanlinshi/insula-functional')
RECON = Path('/cwork/ns458/ECoG_Recon')
SCREEN = Path('/cwork/ns458/insula_recon_screen/shi100_20261006')
OUT = REPO / 'img/misc/shi100_insula_contacts/late_native_qc'
AIC = (86, 87, 88, 89, 90, 91, 92, 93)
PIC = (20, 21, 94, 95)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (Path(sys.prefix) / 'fonts').glob('arial*.ttf'):
        font_manager.fontManager.addfont(str(path))
    font_manager.findfont('Arial', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
                         'font.size': 7, 'svg.fonttype': 'none', 'savefig.bbox': None})
    pipeline = load_pipeline(UPSTREAM)
    focus = []
    for subject, maper_id in [('D157', 'D0157'), ('D158', 'D0158')]:
        atlas, orig, lut, crosswalk, _ = pipeline._resolve_atlas(
            'hammers', str(RECON), subject, maper_id, str(UPSTREAM.parent),
            maper_run='/cwork/ns458/maper_run')
        df = pd.read_csv(SCREEN / f'{subject}_contacts.csv')
        tkr = orig.header.get_vox2ras_tkr()
        df['nearest_aic_gm_voxel_mm'] = np.nan
        for side, parity in [('L', 0), ('R', 1)]:
            mask = df.recon_hemi.eq(side)
            if not mask.any():
                continue
            ijk = np.column_stack(np.where(np.isin(atlas, [x for x in AIC if x % 2 == parity])))
            assert len(ijk), (subject, side, 'no AIC atlas voxels')
            xyz = nib.affines.apply_affine(tkr, ijk)
            distance, nearest = cKDTree(xyz).query(df.loc[mask, ['x', 'y', 'z']].to_numpy())
            df.loc[mask, 'nearest_aic_gm_voxel_mm'] = distance
            df.loc[mask, ['nearest_aic_x', 'nearest_aic_y', 'nearest_aic_z']] = xyz[nearest]
        df.sort_values('nearest_aic_gm_voxel_mm').to_csv(OUT / f'{subject}_aic_distance_qc.csv', index=False)
        # Anatomical candidates only: do not create a task montage or rereference data.
        pairs = []
        for shaft, group in df.groupby('shaft'):
            rows = list(group.sort_values('contact_number').itertuples())
            for first, second in zip(rows[:-1], rows[1:]):
                if second.contact_number != first.contact_number + 1:
                    continue
                assert first.recon_hemi == second.recon_hemi
                p1, p2 = np.array([first.x, first.y, first.z]), np.array([second.x, second.y, second.z])
                mid = (p1 + p2) / 2
                label = pipeline._label_coordinate(mid, orig, atlas, lut)
                mid_roi, _ = pipeline.consensus_roi([label], crosswalk)
                roi, mixed = pipeline.consensus_roi([str(first.atlas_label), label, str(second.atlas_label)], crosswalk)
                pairs.append(dict(subject=subject, shaft=shaft, contact1=first.contact,
                                  contact2=second.contact, midpoint_label=label,
                                  midpoint_roi=mid_roi, consensus_roi=roi, consensus_mix=mixed,
                                  endpoint1_label=first.atlas_label, endpoint2_label=second.atlas_label,
                                  x=mid[0], y=mid[1], z=mid[2], spacing_mm=np.linalg.norm(p2-p1)))
        candidates = pd.DataFrame(pairs)
        candidates.to_csv(OUT / f'{subject}_adjacent_pair_anatomy_qc.csv', index=False)
        aic_candidates = candidates.loc[candidates.midpoint_roi.eq('AIC') | candidates.consensus_roi.eq('AIC')]
        print(subject, 'adjacent-pair AIC candidates (anatomy only):', flush=True)
        print(aic_candidates[['contact1', 'contact2', 'midpoint_label', 'consensus_roi', 'consensus_mix']].to_string(index=False), flush=True)
        selected = (df.nsmallest(3, 'nearest_aic_gm_voxel_mm') if subject == 'D157'
                    else df.loc[df.roi.eq('AIC')])
        # Reorient MRI and masks to RAS+ in the native tkRAS frame, not scanner RAS.
        transform = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(tkr), nib.orientations.axcodes2ornt(('R', 'A', 'S')))
        affine = tkr @ nib.orientations.inv_ornt_aff(transform, orig.shape[:3])
        assert np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3])))
        mri = nib.orientations.apply_orientation(np.asarray(orig.dataobj), transform)
        labels = nib.orientations.apply_orientation(atlas, transform)
        for row in selected.itertuples():
            focus.append((subject, row, mri, labels, affine))
        print(subject, selected[['contact', 'roi', 'atlas_label', 'nearest_aic_gm_voxel_mm']].to_string(index=False), flush=True)

    fig, axes = plt.subplots(len(focus), 3, figsize=(16 / 2.54, (4.1 * len(focus) + 1.6) / 2.54), squeeze=False)
    for row_axes, (subject, row, mri, labels, affine) in zip(axes, focus):
        contact = np.array([row.x, row.y, row.z])
        voxel = np.rint(nib.affines.apply_affine(np.linalg.inv(affine), contact)).astype(int)
        vmax = np.percentile(mri[mri > 0], 99)
        for ax, (fixed, horizontal, vertical, view) in zip(row_axes, [(0, 1, 2, 'Sagittal'), (1, 0, 2, 'Coronal'), (2, 0, 1, 'Axial')]):
            image = np.take(mri, voxel[fixed], axis=fixed).T
            segmentation = np.take(labels, voxel[fixed], axis=fixed).T
            xs = affine[horizontal, horizontal] * np.arange(mri.shape[horizontal]) + affine[horizontal, 3]
            ys = affine[vertical, vertical] * np.arange(mri.shape[vertical]) + affine[vertical, 3]
            dx, dy = affine[horizontal, horizontal], affine[vertical, vertical]
            ax.imshow(image, origin='lower', cmap='gray', vmin=0, vmax=vmax,
                      extent=(xs[0]-dx/2, xs[-1]+dx/2, ys[0]-dy/2, ys[-1]+dy/2))
            for ids, color in [(AIC, '#E2585C'), (PIC, '#55A5F4')]:
                mask = np.isin(segmentation, ids)
                if mask.any():
                    ax.contour(xs, ys, mask.astype(float), levels=[0.5], colors=[color], linewidths=0.65)
            ax.plot(contact[horizontal], contact[vertical], '+', color='#FFE97A', ms=6, mew=0.8)
            ax.set_xlim(contact[horizontal]-14, contact[horizontal]+14)
            ax.set_ylim(contact[vertical]-14, contact[vertical]+14)
            ax.set_title(f'{subject} {row.contact} | {view}', fontsize=7, pad=3)
            ax.set_xlabel(f'{row.roi}; nearest AIC voxel {row.nearest_aic_gm_voxel_mm:.2f} mm', fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('Native MRI QC | red: AIC, blue: PIC, yellow cross: contact', fontsize=8, y=0.985)
    fig.text(0.5, 0.012, 'Nearest-voxel distances are not distances to the anatomical boundary. Labels remain exact-point assignments.',
             ha='center', fontsize=6)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.94, hspace=0.5, wspace=0.1)
    path = OUT / 'D157_D158_native_aic_qc.svg'
    fig.savefig(path, format='svg')
    plt.close(fig)
    print('Saved', path, flush=True)


if __name__ == '__main__':
    main()
