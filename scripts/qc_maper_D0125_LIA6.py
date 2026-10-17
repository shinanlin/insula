#!/usr/bin/env python3
"""Native LIA6 tissue/region QC after the complete 30-atlas D0125 MAPER run.

Only the LIA5-6 pair is examined. No cohort parcellation, selection or NMF
outputs are edited. Tissue posteriors describe the MRI model, not certainty
about the physical extent or recording volume of an implanted contact.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from audit_zero_endpoints import load_lut, read_recon_contact_positions
from prepare_maper_D0125 import BASE, RUN, RECON, OUT

TISSUE = {0: 'background', 1: 'CSF', 2: 'gray_matter', 3: 'white_matter'}
INSULA = {20, 21, *range(86, 96)}
COMMON = Path('/hpc/group/coganlab/nanlinshi/seeg-preprocessing/common')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_array(path, reference):
    image = nib.load(path)
    data = np.asarray(image.dataobj).squeeze()
    if data.shape[:3] != reference.shape or not np.allclose(
            image.affine, reference.affine, atol=1e-4):
        raise ValueError(f'Native geometry mismatch: {path}')
    return data


def neighborhood(center, shape, spacing, radius):
    width = int(np.ceil(radius / min(spacing))) + 1
    lo = np.maximum(np.floor(center).astype(int) - width, 0)
    hi = np.minimum(np.ceil(center).astype(int) + width + 1, shape)
    indices = np.stack(np.meshgrid(*(np.arange(a, b) for a, b in zip(lo, hi)),
                                  indexing='ij'), axis=-1).reshape(-1, 3)
    distance = np.linalg.norm((indices - center) * spacing, axis=1)
    return indices[distance <= radius], distance[distance <= radius]


def qc_figure(orig, arrays, xyz, tkras_to_voxel, output, fs_lut):
    # Use native tkRAS axes for the figure, avoiding scanner obliquity while
    # retaining the exact original voxel arrays (no MRI interpolation).
    tkr = orig.header.get_vox2ras_tkr()
    canonical = nib.as_closest_canonical(nib.Nifti1Image(np.asarray(orig.dataobj), tkr))
    background = np.asarray(canonical.dataobj)
    point_native = nib.affines.apply_affine(tkras_to_voxel, xyz)
    point_tkras = nib.affines.apply_affine(tkr, point_native)
    point = nib.affines.apply_affine(np.linalg.inv(canonical.affine), point_tkras)
    spacing = np.asarray(canonical.header.get_zooms()[:3])
    canonical_arrays = {}
    for name, data in arrays.items():
        image = nib.as_closest_canonical(nib.Nifti1Image(data.astype(np.int16), tkr))
        if image.shape != canonical.shape or not np.allclose(image.affine, canonical.affine):
            raise ValueError(f'Canonical overlay mismatch: {name}')
        canonical_arrays[name] = np.asarray(image.dataobj)
    planes = [(0, (1, 2), 'Sagittal', 'Anterior', 'Superior'),
              (1, (0, 2), 'Coronal', 'Right', 'Superior'),
              (2, (0, 1), 'Axial', 'Right', 'Anterior')]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    cmap = ListedColormap(['#000000', '#44c5df', '#f4a340', '#438dcc'])
    limits = np.percentile(background[background > 0], [1, 99.5])
    insula_fs = {i for i, label in fs_lut.items() if label.startswith('ctx_') and
                 any(s in label for s in ['G_Ins_', 'S_circular_insula'])}
    opercular_fs = {i for i, label in fs_lut.items()
                    if label in {'ctx_lh_G_front_inf-Opercular', 'ctx_rh_G_front_inf-Opercular'}}
    for col, (axis, dims, title, xlabel, ylabel) in enumerate(planes):
        index = int(np.rint(point[axis]))
        coord_axes = [(np.arange(canonical.shape[d]) - point[d]) * spacing[d] for d in dims]
        extent = [coord_axes[0][0] - spacing[dims[0]] / 2,
                  coord_axes[0][-1] + spacing[dims[0]] / 2,
                  coord_axes[1][0] - spacing[dims[1]] / 2,
                  coord_axes[1][-1] + spacing[dims[1]] / 2]
        for row in range(3):
            ax = axes[row, col]
            ax.imshow(np.take(background, index, axis=axis).T, cmap='gray', origin='lower',
                      extent=extent, vmin=limits[0], vmax=limits[1], interpolation='nearest')
            if row == 1:
                overlay = np.take(canonical_arrays['tissue'], index, axis=axis).T
                ax.imshow(np.ma.masked_equal(overlay, 0), origin='lower', extent=extent,
                          cmap=cmap, vmin=-.5, vmax=3.5, alpha=.48, interpolation='nearest')
            if row == 2:
                ribbon = np.take(canonical_arrays['ribbon'], index, axis=axis).T
                overlay = np.zeros(ribbon.shape, dtype=np.uint8)
                overlay[np.isin(ribbon, [2, 41])] = 3
                overlay[np.isin(ribbon, [3, 42])] = 2
                ax.imshow(np.ma.masked_equal(overlay, 0), origin='lower', extent=extent,
                          cmap=cmap, vmin=-.5, vmax=3.5, alpha=.48, interpolation='nearest')
                aparc = np.take(canonical_arrays['aparc'], index, axis=axis).T
                for labels, color in [(opercular_fs, '#ef56d5'), (insula_fs, '#5eee89')]:
                    mask = np.isin(aparc, list(labels))
                    if mask.any() and not mask.all():
                        ax.contour(coord_axes[0], coord_axes[1], mask, levels=[.5],
                                   colors=[color], linewidths=1)
            ax.plot(0, 0, '+', color='red', markersize=12, markeredgewidth=1.5)
            ax.set(xlim=(-18, 18), ylim=(-18, 18), xlabel=f'{xlabel} + (mm from LIA6)',
                   ylabel=f'{ylabel} + (mm from LIA6)')
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(f'{title}; slice offset {(index-point[axis])*spacing[axis]:+.2f} mm')
        axes[0, col].text(.03, .97, 'Native T1', transform=axes[0, col].transAxes,
                           color='white', va='top')
        axes[1, col].text(.03, .97, 'MAPER tissue', transform=axes[1, col].transAxes,
                           color='white', va='top')
        axes[2, col].text(.03, .97, 'FreeSurfer ribbon + Destrieux', transform=axes[2, col].transAxes,
                           color='white', va='top', fontsize=9)
    fig.suptitle('D0125 LIA6: native MRI and tissue boundaries\n'
                 f'tkRAS ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}) mm; red cross = saved contact centre',
                 fontsize=12, y=.995)
    legend = [Patch(color='#f4a340', label='Gray matter'), Patch(color='#438dcc', label='White matter'),
              Patch(color='#44c5df', label='CSF (MAPER)'),
              Patch(color='#ef56d5', label='Destrieux opercular boundary'),
              Patch(color='#5eee89', label='Destrieux insula/circular sulci boundary')]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(output, format='svg', metadata={'Title': 'D0125 LIA6 native tissue QC'})
    plt.close(fig)


def main():
    manifest = json.loads((OUT / 'run_inputs.json').read_text())
    for path, expected in manifest['input_sha256'].items():
        if sha(Path(path)) != expected:
            raise ValueError(f'Input changed during MAPER: {path}')
    orig = nib.load(RECON / 'mri/orig.mgz')
    inverse_tkr = np.linalg.inv(orig.header.get_vox2ras_tkr())
    spacing = np.asarray(orig.header.get_zooms()[:3])
    if not np.allclose(spacing, np.linalg.norm(orig.affine[:3, :3], axis=0)):
        raise ValueError('Unexpected MRI voxel metric')
    output = RUN / 'output'
    fused_path = output / 'f30-seg95-D0125.nii.gz'
    tissue_path = output / 'f30-seg95-D0125-tc3crisp.nii.gz'
    probability_path = output / 'D0125/a1-D0125/tgt-tc3raw.nii.gz'
    seg = native_array(fused_path, orig).astype(np.uint8)
    tissue = native_array(tissue_path, orig).astype(np.uint8)
    probability = native_array(probability_path, orig)
    if probability.shape != (*orig.shape, 3):
        raise ValueError(f'Unexpected tissue posterior dimensions: {probability.shape}')
    if not set(np.unique(tissue)) <= set(TISSUE):
        raise ValueError('Unexpected tissue class codes')
    aparc = native_array(RECON / 'mri/aparc.a2009s+aseg.mgz', orig).astype(np.int32)
    ribbon = native_array(RECON / 'mri/ribbon.mgz', orig).astype(np.uint8)
    aseg = native_array(RECON / 'mri/aseg.mgz', orig).astype(np.int16)
    fs_lut = load_lut(COMMON / 'FreeSurferColorLUT.txt')
    lut_path = COMMON / 'Hammers_mith_atlases_n30r95_label_indices_SPM12_20160111.txt'
    h_lut = {int(i): name for i, name in re.findall(r'<index>(\d+)</index><name>(.*?)</name>',
                                                   lut_path.read_text())}
    h_lut[0] = 'unlabeled'
    pairs = pd.read_csv(BASE / 'docs/evidence/zero_endpoints_2026-10-17/target_132_pairs.csv')
    pair = pairs.loc[pairs.channel.eq('D0125_LIA5-6')].iloc[0]
    points = {
        'LIA5': np.array([pair[f'{a}1_native'] for a in 'xyz']),
        'LIA6': np.array([pair[f'{a}2_native'] for a in 'xyz']),
        'LIA5-6_midpoint': np.array([pair[f'{a}_native'] for a in 'xyz']),
    }
    if not np.allclose(points['LIA5-6_midpoint'], (points['LIA5'] + points['LIA6']) / 2, atol=1e-4):
        raise ValueError('Saved bipolar midpoint differs from endpoint mean')
    contacts = read_recon_contact_positions(RECON)
    for name in ['LIA5', 'LIA6']:
        if not any(np.linalg.norm(points[name] - p[0]) < 1e-3 for p in contacts[name]):
            raise ValueError(f'Saved endpoint does not match restored RAS: {name}')
    voxel = {name: nib.affines.apply_affine(inverse_tkr, xyz) for name, xyz in points.items()}
    rounded = {name: tuple(np.rint(ijk).astype(int)) for name, ijk in voxel.items()}
    atlas_samples = []
    for i in range(1, 31):
        directory = output / f'D0125/a{i}-D0125'
        atlas = native_array(directory / 'seg/seg95.nii.gz', orig)
        crisp = native_array(directory / 'tgt-tc3crisp.nii.gz', orig)
        for name, ijk in rounded.items():
            atlas_samples.append(dict(atlas_id=f'a{i}', point=name,
                raw_hammers_code=int(atlas[ijk]), tissue_code=int(crisp[ijk])))
    samples = pd.DataFrame(atlas_samples)
    records, neighborhoods = [], []
    for name, xyz in points.items():
        ijk = rounded[name]
        p = np.array([map_coordinates(probability[..., i], voxel[name][:, None], order=1,
                                      mode='nearest')[0] for i in range(3)])
        if not np.isfinite(p).all() or (p < -1e-5).any() or (p > 1.00001).any():
            raise ValueError('Invalid interpolated tissue probabilities')
        code = int(tissue[ijk])
        region = int(seg[ijk])
        per_point = samples.loc[samples.point.eq(name)]
        votes = Counter(per_point.raw_hammers_code.astype(int))
        if votes[region] != max(votes.values()):
            raise ValueError(f'Fused point is not a majority-vote label: {name}')
        distance_grid, distances = neighborhood(voxel[name], np.array(orig.shape), spacing, 12.)
        dist_idx = tuple(distance_grid.T)
        def closest(mask):
            return float(distances[mask].min()) if mask.any() else None
        opercular = [i for i, label in fs_lut.items()
                     if label in {'ctx_lh_G_front_inf-Opercular', 'ctx_rh_G_front_inf-Opercular'}]
        row = dict(point=name, x_tkras_mm=xyz[0], y_tkras_mm=xyz[1], z_tkras_mm=xyz[2],
            voxel_i=int(ijk[0]), voxel_j=int(ijk[1]), voxel_k=int(ijk[2]),
            nearest_voxel_offset_mm=float(np.linalg.norm((np.array(ijk)-voxel[name])*spacing)),
            maper_tissue_code=code, maper_tissue=TISSUE[code],
            raw_hammers_code=region, raw_hammers_region=h_lut[region],
            gated_hammers_code=region if code == 2 else 0,
            gated_hammers_region=h_lut[region] if code == 2 else '0 (not gray matter in MAPER)',
            freesurfer_aparc_code=int(aparc[ijk]), freesurfer_aparc=fs_lut[int(aparc[ijk])],
            freesurfer_aseg=fs_lut.get(int(aseg[ijk]), str(int(aseg[ijk]))),
            freesurfer_ribbon_code=int(ribbon[ijk]),
            freesurfer_ribbon_tissue={0: 'outside_ribbon', 2: 'white_matter', 3: 'gray_matter',
                                     41: 'white_matter', 42: 'gray_matter'}.get(int(ribbon[ijk]), 'other'),
            p_csf_a1_nearest=float(probability[ijk][0]),
            p_gm_a1_nearest=float(probability[ijk][1]), p_wm_a1_nearest=float(probability[ijk][2]),
            p_csf_a1_trilinear=float(p[0]), p_gm_a1_trilinear=float(p[1]), p_wm_a1_trilinear=float(p[2]),
            n_pairings_same_tissue=int(per_point.tissue_code.eq(code).sum()),
            tissue_codes_over_30_pairings=json.dumps({int(k): int(v) for k, v in
                sorted(Counter(per_point.tissue_code.astype(int)).items())}),
            fused_region_votes=int(votes[region]), n_atlases=30,
            nearest_maper_gm_voxel_center_mm=closest(tissue[dist_idx] == 2),
            nearest_maper_wm_voxel_center_mm=closest(tissue[dist_idx] == 3),
            nearest_freesurfer_opercular_voxel_center_mm=closest(np.isin(aparc[dist_idx], opercular)),
            nearest_maper_insular_gm_voxel_center_mm=closest(
                np.isin(seg[dist_idx], list(INSULA)) & (tissue[dist_idx] == 2)))
        records.append(row)
        for radius in [1., 2.]:
            indices, _ = neighborhood(voxel[name], np.array(orig.shape), spacing, radius)
            local_tissue = tissue[tuple(indices.T)]
            neighborhoods.append(dict(point=name, radius_mm=radius, n_voxel_centers=len(indices),
                **{f'fraction_{TISSUE[c]}': float(np.mean(local_tissue == c)) for c in TISSUE}))
    pd.DataFrame(records).to_csv(OUT / 'contact_results.csv', index=False)
    pd.DataFrame(neighborhoods).to_csv(OUT / 'contact_neighborhoods.csv', index=False)
    samples.to_csv(OUT / 'atlas_contact_samples.csv', index=False)
    figure = OUT / 'LIA6_native_tissue_qc.svg'
    qc_figure(orig, {'tissue': tissue, 'aparc': aparc, 'ribbon': ribbon},
              points['LIA6'], inverse_tkr, figure, fs_lut)
    # Keep compact, final volumes on persistent group storage; all registration
    # scratch and intermediate pairings remain isolated under cwork.
    persistent = Path('/hpc/group/coganlab/nanlinshi/cache/maper/insula-paper/D0125_20261017')
    persistent.mkdir(parents=True, exist_ok=True)
    retained = {}
    for source in [fused_path, tissue_path, probability_path, OUT / 'run_inputs.json']:
        dest = persistent / source.name
        source_hash = sha(source)
        if dest.exists() and sha(dest) != source_hash:
            raise ValueError(f'Existing persistent result differs: {dest}')
        if not dest.exists():
            shutil.copyfile(source, dest)
        retained[str(dest)] = source_hash
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), subject='D0125',
        focus_contact='LIA6', run=str(RUN), n_complete_atlas_pairings=30,
        all_native_geometry_checks_passed=True, contact_results=records,
        neighborhoods=neighborhoods, figure=str(figure), persistent_outputs=retained,
        retained_probability_source='a1 target EM posteriors; independent of atlas label fusion',
        tissue_codes=TISSUE,
        nearest_distance_definition='Saved contact coordinate to closest labeled voxel centre, within 12 mm; not distance to a physical tissue surface',
        source_hashes={str(p): sha(p) for p in [fused_path, tissue_path, probability_path,
            lut_path, COMMON / 'FreeSurferColorLUT.txt', Path(manifest['container']),
            BASE / 'scripts/prepare_maper_D0125.py',
            BASE / 'pipeline/03_run_maper/generate_launchlist.sh', Path(__file__)]},
        scope='Only D0125 LIA6 and its LIA5-6 pair; no inclusion changes, no other subjects, no NMF',
        nmf_rerun=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
