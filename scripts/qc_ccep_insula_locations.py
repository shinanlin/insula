"""Targeted native-space insula localization for three CCEP bipolar pairs."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patheffects as pe
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from screen_recon_aic_contacts import read_contacts, load_pipeline, UPSTREAM

ROOT = Path('/hpc/group/coganlab/nanlinshi/insula-functional')
OUT = ROOT / 'results/ccep_insula_locations_20261008'
FIG = ROOT / 'img/misc/ccep_insula_locations_20261008.svg'
RECON = Path('/cwork/ns458/ECoG_Recon')
MAPER = Path('/cwork/ns458/maper_run')
TARGETS = {'D149': [('RTAS5', 'RTAS6')], 'D153': [('RFMI8', 'RFMI9'), ('RFMI9', 'RFMI10')]}
REGIONS = {'ASG': 87, 'MSG': 89, 'PSG': 91, 'AP': 93, 'ALG': 95, 'PLG': 21}
COLORS = {'ASG': '#f05d5e', 'MSG': '#f6bd60', 'PSG': '#84a59d',
          'AP': '#e78ac3', 'ALG': '#4d9de0', 'PLG': '#a78bfa'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(UPSTREAM)
    results, panels = [], []
    for recon, pairs in TARGETS.items():
        subject = f'D{int(recon[1:]):04d}'
        contacts, _, _ = read_contacts(RECON / recon)
        lookup = {c['contact']: np.array([c['x'], c['y'], c['z']]) for c in contacts}
        atlas, orig, lut, crosswalk, _ = pipeline._resolve_atlas(
            'hammers', str(RECON), recon, subject, str(UPSTREAM.parent), str(MAPER))
        aparc = pipeline._resolve_atlas('aparc2009s', str(RECON), recon, subject, str(UPSTREAM.parent))
        raw_seg = np.asarray(nib.load(MAPER / subject / 'output' / f'f30-seg95-{subject}.nii.gz').dataobj).squeeze()
        tkr = orig.header.get_vox2ras_tkr()
        trees = {}
        for mode, volume in [('gray_matter', atlas), ('ungated_segmentation', raw_seg)]:
            for region, label_id in REGIONS.items():
                ijk = np.column_stack(np.where(volume == label_id))
                xyz = nib.affines.apply_affine(tkr, ijk)
                if not len(xyz):
                    raise ValueError(f'Empty region {subject} {mode} {region}')
                trees[mode, region] = (cKDTree(xyz), xyz)
        orientation = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(tkr), nib.orientations.axcodes2ornt(('R', 'A', 'S')))
        affine = tkr @ nib.orientations.inv_ornt_aff(orientation, orig.shape[:3])
        mri = nib.orientations.apply_orientation(np.asarray(orig.dataobj), orientation)
        labels = nib.orientations.apply_orientation(atlas, orientation)
        for first, second in pairs:
            p1, p2 = lookup[first], lookup[second]
            midpoint = (p1 + p2) / 2
            points = []
            for name, xyz in [(first, p1), ('midpoint', midpoint), (second, p2)]:
                h = pipeline._label_coordinate(xyz, orig, atlas, lut)
                a = pipeline._label_coordinate(xyz, aparc[1], aparc[0], aparc[2])
                rankings = {}
                for mode in ('gray_matter', 'ungated_segmentation'):
                    distances = []
                    for region in REGIONS:
                        tree, coords = trees[mode, region]
                        distance, index = tree.query(xyz)
                        distances.append(dict(region=region, label=lut[REGIONS[region]],
                                              distance_mm=float(distance), nearest_voxel_tkras_mm=coords[index].tolist()))
                    rankings[mode] = sorted(distances, key=lambda r: r['distance_mm'])
                points.append(dict(point=name, tkras_mm=xyz.tolist(), aparc_label=a,
                                   hammers_gm_label=h, nearest_regions=rankings))
            entry = dict(subject=recon, pair=f'{first}-{second}', points=points)
            results.append(entry)
            panels.append((entry, mri, labels, affine))
            print(json.dumps(entry), flush=True)
    report = dict(
        method='Exact native tkRAS point labels plus Euclidean distance to nearest same-hemisphere (right) insular voxel centers. Distances are not distances to a continuous anatomical boundary.',
        primary_distance='Gray-matter-gated Hammers n30r95 (tissue==2); ungated segmentation included as sensitivity check only.',
        interpretation='Fine atlas subdivisions are reported directly; no new anterior/middle/posterior three-way atlas is imposed.',
        subjects=results)
    (OUT / 'locations.json').write_text(json.dumps(report, indent=2) + '\n')
    for path in (Path(sys.prefix) / 'fonts').glob('arial*.ttf'):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 3, figsize=(18/2.54, 19/2.54))
    for row_axes, (entry, mri, labels, affine) in zip(axes, panels):
        point = entry['points'][1]
        xyz = np.array(point['tkras_mm'])
        endpoints = [entry['points'][0], entry['points'][2]]
        np.testing.assert_allclose(xyz, np.mean([p['tkras_mm'] for p in endpoints], axis=0))
        voxel = np.rint(nib.affines.apply_affine(np.linalg.inv(affine), xyz)).astype(int)
        vmax = float(np.percentile(mri[mri > 0], 99))
        for ax, (fixed, horizontal, vertical, view) in zip(row_axes, [(0,1,2,'Sagittal'), (1,0,2,'Coronal'), (2,0,1,'Axial')]):
            img = np.take(mri, voxel[fixed], axis=fixed).T
            seg = np.take(labels, voxel[fixed], axis=fixed).T
            xs = affine[horizontal,horizontal]*np.arange(mri.shape[horizontal])+affine[horizontal,3]
            ys = affine[vertical,vertical]*np.arange(mri.shape[vertical])+affine[vertical,3]
            dx, dy = affine[horizontal,horizontal], affine[vertical,vertical]
            ax.imshow(img, origin='lower', cmap='gray', vmin=0, vmax=vmax,
                      extent=(xs[0]-dx/2,xs[-1]+dx/2,ys[0]-dy/2,ys[-1]+dy/2))
            for region, label_id in REGIONS.items():
                mask = seg == label_id
                if mask.any():
                    ax.contour(xs, ys, mask.astype(float), levels=[.5], colors=[COLORS[region]], linewidths=.7)
            # Orthogonal slices pass through the nearest voxel to the midpoint.
            # Contacts retain their true in-plane coordinates; explicitly report
            # signed out-of-plane offsets from the displayed slice center.
            slice_coordinate = affine[fixed,fixed]*voxel[fixed]+affine[fixed,3]
            offsets = []
            for endpoint, color, marker, text_offset, align in zip(
                    endpoints, ['#00e5ff', '#ff9f1c'], ['o', 's'],
                    [(-6, -11), (6, -11)], ['right', 'left']):
                location = np.array(endpoint['tkras_mm'])
                offsets.append(float(location[fixed]-slice_coordinate))
                ax.plot(location[horizontal], location[vertical], marker=marker,
                        linestyle='none', markerfacecolor='none', markeredgecolor=color,
                        markersize=5, markeredgewidth=1.1, zorder=5)
                text = ax.annotate(endpoint['point'],
                                   (location[horizontal], location[vertical]),
                                   xytext=text_offset, textcoords='offset points',
                                   color=color, fontsize=6, ha=align, va='top', zorder=7)
                text.set_path_effects([pe.withStroke(linewidth=1.8, foreground='black')])
            ax.plot(xyz[horizontal], xyz[vertical], '+', color='#ffea00', ms=7, mew=1.1, zorder=6)
            text = ax.annotate('BP center', (xyz[horizontal], xyz[vertical]),
                               xytext=(0, 9), textcoords='offset points',
                               color='#ffea00', fontsize=6, ha='center', va='bottom', zorder=7)
            text.set_path_effects([pe.withStroke(linewidth=1.8, foreground='black')])
            ax.set_xlim(xyz[horizontal]-18, xyz[horizontal]+18)
            ax.set_ylim(xyz[vertical]-18, xyz[vertical]+18)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{entry["subject"]} {entry["pair"]}\n{view}', fontsize=7)
            nearest = point['nearest_regions']['gray_matter'][0]
            ax.set_xlabel(
                f'BP nearest GM: {nearest["region"]}, {nearest["distance_mm"]:.2f} mm\n'
                f'Slice offset: {endpoints[0]["point"]} {offsets[0]:+.1f}, '
                f'{endpoints[1]["point"]} {offsets[1]:+.1f} mm', fontsize=5.5)
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0],[0],marker='o',linestyle='none',markerfacecolor='none',
               markeredgecolor='#00b8ce',label='Physical contact 1'),
        Line2D([0],[0],marker='s',linestyle='none',markerfacecolor='none',
               markeredgecolor='#e08000',label='Physical contact 2'),
        Line2D([0],[0],marker='+',linestyle='none',color='#a38f00',label='Virtual bipolar center'),
    ], loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.077),fontsize=6.5)
    fig.legend(handles=[Line2D([0],[0],color=COLORS[r],label=r) for r in REGIONS],
               loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5,.048))
    fig.suptitle('Native MRI: physical contacts and virtual bipolar centers', fontsize=8)
    fig.text(.5,.032,'Contacts are projected onto the midpoint slice; signed offsets indicate out-of-plane distance.',ha='center',fontsize=5.5)
    fig.text(.5,.016,'Offsets follow native RAS axes (right / anterior / superior). Contours: right insula gray matter.',ha='center',fontsize=5.5)
    fig.subplots_adjust(left=.03,right=.97,top=.91,bottom=.17,hspace=.65,wspace=.12)
    fig.savefig(FIG, format='svg')
    plt.close(fig)
    print('Saved', FIG, flush=True)


if __name__ == '__main__':
    main()
