#!/usr/bin/env python3
"""Render k=2 NMF insula brain maps for nmf.ipynb (Hammers or aparc)."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
from mne.viz import Brain
from scipy.spatial import cKDTree
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.decomposition import NMF
import xarray as xr
from mne_bids import BIDSPath
from tqdm import tqdm
import os
os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
import mne
mne.viz.set_3d_backend('notebook')

cm = 1 / 2.54
fontsize = 7
blue = '#2369BD'
red = '#A9373B'
recon_dir = '/cwork/ns458/ECoG_Recon/'
FS_SUBJECT = 'cvs_avg35_inMNI152'
WEIGHT_SIZE_MIN, WEIGHT_SIZE_MAX = 6, 22

TASKS = ['PhonemeSequence', 'LexicalNoDelay', 'LexicalDelay', 'PictureNaming', 'SentenceRep']
REF = 'bipolar'

PHASE_CONFIGS = {
    'stimulus': {'phase': 'stimulus', 'time_min': -0.5, 'time_max': 1.0},
    'delay': {'phase': 'delay', 'time_min': 0.0, 'time_max': 1.0},
    'go': {'phase': 'go', 'time_min': -0.5, 'time_max': 1.0},
    'response': {'phase': 'response', 'time_min': -0.5, 'time_max': 0.5},
}


def insula_mask(df, atlas):
    if atlas in ('aparc_raw',):
        return df['roi'].isin({'INS', 'Insula'})
    mask = df['roi'].isin({'AIC', 'PIC'})
    if atlas == 'hammers' and 'mix' in df.columns:
        mask = mask & ~df['mix'].fillna(False).astype(bool)
    return mask


def filter_qc_electrodes(spatial):
    spatial = spatial.copy()
    if 'mix' in spatial.columns:
        spatial = spatial[~spatial['mix'].fillna(False).astype(bool)]
    if 'label' in spatial.columns:
        label_num = pd.to_numeric(spatial['label'], errors='coerce')
        spatial = spatial[~(label_num == 0)]
    return spatial


def classify_insula_row(row, y_threshold=0):
    if row['roi'] != 'Insula':
        return row['roi']
    label = row['label']
    y_coord = row['y']
    if ('G_insular_short' in label or
        'S_circular_insula_ant' in label or
        ('S_circular_insula_sup' in label and y_coord > y_threshold) or
        ('S_circular_insula_inf' in label and y_coord > y_threshold)):
        return 'AIC'
    if ('G_Ins_lg_and_S_cent_ins' in label or
        ('S_circular_insula_sup' in label and y_coord <= y_threshold) or
        ('S_circular_insula_inf' in label and y_coord <= y_threshold)):
        return 'PIC'
    return row['roi']


def load_hga_data(atlas):
    hga_paths = []
    for t in TASKS:
        if atlas == 'hammers':
            root = f'../results/{t}({REF})({atlas})'
        else:
            root = f'../results/{t}({REF})'
        hga_paths.extend(
            BIDSPath(root=root, datatype='HGA', suffix='time', check=False).match()
        )
    if not hga_paths:
        raise FileNotFoundError(f'No time CSVs for atlas={atlas}')
    df = pd.concat([pd.read_csv(p) for p in tqdm(hga_paths, desc='load')], ignore_index=True)
    df.loc[df.phase == 'Resp', 'phase'] = 'Response'
    df.loc[df.phase == 'Audio', 'phase'] = 'Stimulus'
    df['phase'] = df['phase'].str.lower()
    if atlas == 'hammers':
        df.loc[df.roi == 'PrG', 'roi'] = 'SMC'
        df.loc[df.roi == 'PoG', 'roi'] = 'SMC'
        df.loc[df.roi == 'Subcentral', 'roi'] = 'SMC'
        df.loc[df.roi == 'STGp', 'roi'] = 'STG'
        df.loc[df.roi == 'STGa', 'roi'] = 'STG'
        df.loc[df.roi == 'HG', 'roi'] = 'STG'
    else:
        df.loc[df.roi == 'INS', 'roi'] = 'Insula'
        df.loc[df.roi == 'PrG', 'roi'] = 'SMC'
        df.loc[df.roi == 'PoG', 'roi'] = 'SMC'
        df.loc[df.roi == 'Subcentral', 'roi'] = 'SMC'
        df.loc[df.roi == 'STGp', 'roi'] = 'STG'
        df.loc[df.roi == 'STGa', 'roi'] = 'STG'
        df.loc[df.roi == 'HG', 'roi'] = 'STG'
        df['roi'] = df.apply(classify_insula_row, axis=1, y_threshold=0)
    return df


def channel_metadata(df):
    return (
        df.groupby('channel')
        .agg({'x': 'first', 'y': 'first', 'z': 'first', 'roi': 'first', 'label': 'first', 'hemi': 'first', 'subject': 'first'})
        .reset_index()
    )


def prepare_subset(df, *, atlas='hammers', description='all', sig_mode='sig_union', phase_configs=PHASE_CONFIGS):
    sub = df[(df.modality == 'sound') & insula_mask(df, atlas)].copy()
    if description == 'Repeat':
        sub = sub[sub.description == 'Repeat']
    meta = filter_qc_electrodes(channel_metadata(sub))
    sub = sub[sub.channel.isin(meta.channel)]
    if sig_mode == 'all_insula':
        keep = set(meta.channel)
    elif sig_mode == 'sig_union':
        sub['significance'] = sub.groupby(['task', 'channel'])['mask'].transform('any')
        keep = set(sub.loc[sub.significance, 'channel'].unique())
    else:
        raise ValueError(sig_mode)
    sub = sub[sub.channel.isin(keep)]
    meta = meta[meta.channel.isin(keep)]
    return sub, meta


def build_phase_xarray(df, phase, time_min, time_max):
    stage = (
        df[(df.phase == phase) & (df.time > time_min) & (df.time < time_max)]
        .groupby(['channel', 'time'])['value']
        .mean()
        .to_xarray()
    )
    return stage - stage.mean(dim='time')


def build_concat_stage(subset, phase_configs=PHASE_CONFIGS):
    stages, time_vectors, phase_names = [], [], []
    for name, cfg in phase_configs.items():
        stage = build_phase_xarray(subset, cfg['phase'], cfg['time_min'], cfg['time_max'])
        stages.append(stage)
        time_vectors.append(stage.time.values)
        phase_names.append(name)
    concat_stage = xr.concat(stages, dim='time').fillna(0)
    valid_channels = concat_stage.dropna(dim='channel', how='any').channel.values
    concat_stage = concat_stage.sel(channel=valid_channels)
    meta = channel_metadata(subset).set_index('channel').loc[valid_channels].reset_index()
    return concat_stage, time_vectors, phase_names, meta


def preprocess_for_nmf(data_matrix, clip_percentile=None, min_shift=True):
    X = data_matrix.copy()
    if clip_percentile is not None:
        lo, hi = np.percentile(X, clip_percentile)
        X = np.clip(X, lo, hi)
    if min_shift:
        X = X - X.min()
    return X


def nmf_k2_loadings(W):
    comp = W.argmax(axis=1)
    minor = 1 - comp
    dom_w = W[np.arange(len(W)), comp]
    minor_w = W[np.arange(len(W)), minor]
    total = W.sum(axis=1)
    dom_ratio = dom_w / np.maximum(total, 1e-12)
    return comp, dom_w, minor_w, dom_ratio


def weight_to_point_sizes(w, size_min=WEIGHT_SIZE_MIN, size_max=WEIGHT_SIZE_MAX):
    w = np.asarray(w, dtype=float)
    if len(w) == 0:
        return w
    w_max = float(w.max())
    if w_max <= 0:
        return np.full(len(w), 0.5 * (size_min + size_max))
    return size_min + (size_max - size_min) * (w / w_max)


def plot_insula_electrode_panel(plot_df, *, colors, sizes=None, title, legend_handles=None, save_path=None,
                                lh_tree=None, rh_tree=None, lh_pial_coords=None, rh_pial_coords=None,
                                lh_insula_center=None, rh_insula_center=None):
    if plot_df.empty:
        print(f'Skip plot (no electrodes): {title}')
        return
    cord = plot_df[['x', 'y', 'z']].values
    mask_lh = cord[:, 0] < 0
    mask_rh = cord[:, 0] > 0
    size_arr = None if sizes is None else np.asarray(sizes)

    def _make_insula_brain(hemi):
        return Brain(
            FS_SUBJECT, subjects_dir=recon_dir, surf='pial',
            hemi=hemi, background='white', show=False,
            cortex=(0.9, 0.9, 0.9), alpha=0.1, size=(800, 800),
        )

    def _project_to_pial(c, tree, pial_coords):
        _, indices = tree.query(c)
        return pial_coords[indices]

    def _add_electrodes_fixed(brain, coords_proj, cols, szs=None, point_size=12):
        if szs is None:
            szs = np.full(len(coords_proj), point_size)
        else:
            szs = np.asarray(szs, dtype=float)
        for pt, color, sz in zip(coords_proj, cols, szs):
            cloud = pv.PolyData(pt.reshape(1, 3))
            brain._renderer.plotter.add_mesh(
                cloud, render_points_as_spheres=True,
                point_size=float(sz), color=color, lighting=False,
            )

    lh_brain = _make_insula_brain('lh')
    rh_brain = _make_insula_brain('rh')

    if mask_lh.any():
        lh_proj = _project_to_pial(cord[mask_lh], lh_tree, lh_pial_coords)
        lh_sizes = None if size_arr is None else size_arr[mask_lh]
        _add_electrodes_fixed(lh_brain, lh_proj, np.asarray(colors)[mask_lh], sizes=lh_sizes)
    if mask_rh.any():
        rh_proj = _project_to_pial(cord[mask_rh], rh_tree, rh_pial_coords)
        rh_sizes = None if size_arr is None else size_arr[mask_rh]
        _add_electrodes_fixed(rh_brain, rh_proj, np.asarray(colors)[mask_rh], sizes=rh_sizes)

    lh_brain.show_view(azimuth=180, elevation=90, distance=180, focalpoint=lh_insula_center)
    rh_brain.show_view(azimuth=0, elevation=90, distance=180, focalpoint=rh_insula_center)

    fig, axes = plt.subplots(1, 2, figsize=(16 * cm, 8 * cm))
    axes[0].imshow(lh_brain.screenshot(mode='rgb'))
    axes[0].axis('off')
    axes[0].set_title('Left Insula', fontsize=fontsize)
    axes[1].imshow(rh_brain.screenshot(mode='rgb'))
    axes[1].axis('off')
    axes[1].set_title('Right Insula', fontsize=fontsize)
    if legend_handles:
        axes[0].legend(handles=legend_handles, loc='upper left', fontsize=fontsize - 1, framealpha=0.9)
    plt.suptitle(f'{title} (n={len(plot_df)})', fontsize=fontsize)
    lh_brain.close()
    rh_brain.close()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved {save_path}')
    plt.close(fig)


def render_brain_maps(atlas):
    atlas_key = 'hammers' if atlas == 'hammers' else 'aparc'
    output_dir = Path('../tmp/nmf') if atlas_key == 'hammers' else Path('../tmp/nmf/aparc')
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = '' if atlas_key == 'hammers' else 'aparc_'
    best_sig = 'sig_union'
    best_desc = 'all'
    best_k = 2
    best_tag = f'{prefix}k{best_k}_clip-none_sig-{best_sig}_desc-{best_desc}'
    atlas_label = 'Hammers' if atlas_key == 'hammers' else 'aparc'

    hgas = load_hga_data(atlas_key)
    subset_k, _ = prepare_subset(hgas, atlas=atlas_key, description=best_desc, sig_mode=best_sig)
    concat_k, _, _, meta_k = build_concat_stage(subset_k)
    data_k = preprocess_for_nmf(concat_k.values, clip_percentile=None)
    nmf_k = NMF(n_components=best_k, random_state=42, max_iter=500)
    W2 = nmf_k.fit_transform(data_k)
    comp2, dom_w2, minor_w2, dom_ratio2 = nmf_k2_loadings(W2)
    plot_meta = meta_k.copy()
    plot_meta['component'] = comp2
    plot_meta['weight_dom'] = dom_w2
    plot_meta['weight_minor'] = minor_w2
    plot_meta['dom_ratio'] = dom_ratio2
    plot_meta['size_dom'] = weight_to_point_sizes(dom_w2)
    plot_meta['size_minor'] = weight_to_point_sizes(minor_w2)

    print(f'{atlas_label} NMF k=2 hard assignment counts:', pd.Series(comp2).value_counts().sort_index().to_dict())

    labels = mne.read_labels_from_annot(
        subject=FS_SUBJECT, parc='aparc.a2009s',
        surf_name='pial', hemi='both', subjects_dir=recon_dir,
    )
    insula_patterns = [
        'G_insular_short', 'G_Ins_lg_and_S_cent_ins',
        'S_circular_insula_ant', 'S_circular_insula_inf', 'S_circular_insula_sup',
    ]
    lh_pial_coords, _ = mne.read_surface(f'{recon_dir}/{FS_SUBJECT}/surf/lh.pial')
    rh_pial_coords, _ = mne.read_surface(f'{recon_dir}/{FS_SUBJECT}/surf/rh.pial')
    lh_tree = cKDTree(lh_pial_coords)
    rh_tree = cKDTree(rh_pial_coords)

    def _insula_center(hemi, pial_coords):
        vertices = []
        for lab in labels:
            if lab.hemi == hemi and any(p in lab.name for p in insula_patterns):
                vertices.extend(lab.vertices)
        return pial_coords[vertices].mean(axis=0) if vertices else None

    lh_insula_center = _insula_center('lh', lh_pial_coords)
    rh_insula_center = _insula_center('rh', rh_pial_coords)

    nmf_colors = [{0: blue, 1: red}[c] for c in plot_meta['component']]
    roi_colors = [{ 'AIC': blue, 'PIC': red}.get(r, 'gray') for r in plot_meta['roi']]
    brain_kw = dict(
        lh_tree=lh_tree, rh_tree=rh_tree,
        lh_pial_coords=lh_pial_coords, rh_pial_coords=rh_pial_coords,
        lh_insula_center=lh_insula_center, rh_insula_center=rh_insula_center,
    )

    nmf_weighted_legend = [
        Patch(facecolor=blue, label='Component 1 (color)'),
        Patch(facecolor=red, label='Component 2 (color)'),
        Line2D([], [], linestyle='None', marker='o', color='w', markerfacecolor='0.45',
               markersize=9, label='Larger sphere = stronger loading'),
    ]
    minor_weight_legend = [
        Patch(facecolor=blue, label='Dominant comp. 1 (color)'),
        Patch(facecolor=red, label='Dominant comp. 2 (color)'),
        Line2D([], [], linestyle='None', marker='o', color='w', markerfacecolor='0.45',
               markersize=9, label='Size = non-dominant loading'),
    ]
    atlas_legend = [
        Patch(facecolor=blue, label=f'{atlas_label} AIC'),
        Patch(facecolor=red, label=f'{atlas_label} PIC'),
    ]

    plot_insula_electrode_panel(
        plot_meta, colors=nmf_colors, sizes=plot_meta['size_dom'].values,
        title=f'NMF k=2 | color=dominant, size=dominant W | {best_tag}',
        legend_handles=nmf_weighted_legend,
        save_path=output_dir / f'{best_tag}_brain_nmf_weighted.png',
        **brain_kw,
    )
    plot_insula_electrode_panel(
        plot_meta, colors=nmf_colors, sizes=plot_meta['size_minor'].values,
        title=f'NMF k=2 | color=dominant, size=minor W | {best_tag}',
        legend_handles=minor_weight_legend,
        save_path=output_dir / f'{best_tag}_brain_nmf_minor_weight.png',
        **brain_kw,
    )
    atlas_suffix = 'hammers' if atlas_key == 'hammers' else 'aparc'
    plot_insula_electrode_panel(
        plot_meta, colors=roi_colors,
        title=f'{atlas_label} AIC/PIC | {best_tag}',
        legend_handles=atlas_legend,
        save_path=output_dir / f'{best_tag}_brain_{atlas_suffix}.png',
        **brain_kw,
    )

    fig, axes = plt.subplots(1, 2, figsize=(20 * cm, 8 * cm))
    for ax, img_path, subtitle in zip(
        axes,
        [
            output_dir / f'{best_tag}_brain_nmf_weighted.png',
            output_dir / f'{best_tag}_brain_{atlas_suffix}.png',
        ],
        ['NMF k=2 (weighted)', f'{atlas_label} AIC/PIC'],
    ):
        ax.imshow(plt.imread(img_path))
        ax.axis('off')
        ax.set_title(subtitle, fontsize=fontsize)
    plt.suptitle(f'Insula clusters — {best_tag}', fontsize=fontsize)
    plt.tight_layout()
    combo_path = output_dir / f'{best_tag}_brain_sidebyside.png'
    fig.savefig(combo_path, dpi=200, bbox_inches='tight')
    print(f'Saved {combo_path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Render NMF k=2 insula brain maps')
    parser.add_argument('atlas', nargs='?', default='hammers', choices=['hammers', 'aparc'],
                        help='Atlas partition (default: hammers)')
    parser.add_argument('--both', action='store_true', help='Render Hammers and aparc')
    args = parser.parse_args()
    if args.both:
        for atlas in ('hammers', 'aparc'):
            print(f'\n=== {atlas} ===')
            render_brain_maps(atlas)
    else:
        render_brain_maps(args.atlas)


if __name__ == '__main__':
    main()
