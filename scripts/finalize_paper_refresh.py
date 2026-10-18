#!/usr/bin/env python3
"""Validate completed figures and record the genuine missing-input boundary."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.nmf.whole_brain_projection_viz import prepare_projection_frame

OUT = ROOT / 'docs/evidence/figures3_6_refresh_2026-10-17'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    baseline = json.loads((OUT / 'baseline.json').read_text())
    assert all(sha(p) == digest for p, digest in baseline['protected'].items())
    checks = json.loads((OUT / 'approved_method_checks.json').read_text())
    assert checks['passed']
    receipts = {}
    for number, name in [(3, 'available_execution_validation.json'), (6, 'execution_validation.json')]:
        receipt = json.loads((ROOT / f'results/fig{number}' / name).read_text())
        assert receipt['passed'] and receipt['svg_inline_only']
        for image in receipt['figures']:
            path = Path(image['path'])
            assert sha(path) == image['sha256']
            assert ET.parse(path).getroot().tag == '{http://www.w3.org/2000/svg}svg'
            archive = OUT / 'figures' / f'fig{number}'
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archive / path.name)
        receipts[str(number)] = receipt
    assert len(receipts['3']['figures']) == 5 and len(receipts['6']['figures']) == 3
    projection = ROOT / 'results/nmf/whole_brain_projection'
    manifest = json.loads((projection / 'manifest.json').read_text())
    assert manifest['phases'] == ['stimulus', 'delay', 'response'] and manifest['n_features'] == 399
    assert manifest['basis_sha256'] == sha(ROOT / 'results/nmf/H_by_phase.csv')
    assert manifest['n_subjects_skipped'] == 0
    table = pd.read_csv(projection / 'electrode_projection.csv')
    assert not table[['subject', 'channel']].duplicated().any()
    assert len(table) == manifest['n_electrodes'] and table.subject.nunique() == manifest['n_subjects_used']
    loadings = table[['loading_sensory', 'loading_sustain', 'loading_motor']].to_numpy()
    assert np.isfinite(loadings).all() and (loadings >= 0).all()
    heldout = prepare_projection_frame(table)
    membership_columns = ['subject', 'channel', 'roi', 'hemi', 'in_discovery', 'n_tasks_available', 'significant_phases']
    table[membership_columns].to_csv(ROOT / 'results/fig6/projection_membership.csv', index=False)
    heldout[membership_columns].to_csv(ROOT / 'results/fig6/map_membership.csv', index=False)
    table.groupby(['subject', 'roi']).size().rename('n_sites').reset_index().to_csv(ROOT / 'results/fig6/subject_roi_counts.csv', index=False)
    assignments = pd.read_csv(ROOT / 'results/nmf/channel_assignments.csv')
    absent = assignments.loc[~assignments.channel.isin(table.channel)]
    absent.to_csv(ROOT / 'results/fig6/discovery_sites_not_in_projection.csv', index=False)
    with np.load(projection / 'visualization/surface_motif_maps_focused.npz') as cache:
        for hemi in ('lh', 'rh'):
            for component in ('sensory', 'sustain', 'motor'):
                values = cache[f'{hemi}_{component}_specificity']
                mask = cache[f'{hemi}_{component}_display_mask'].astype(bool)
                assert values.shape == mask.shape and np.isfinite(values[mask]).all()
    f6 = dict(passed=True, n_subjects=manifest['n_subjects_used'], projected_sites=len(table),
        projected_discovery_sites=int(table.in_discovery.sum()), map_sites=len(heldout),
        map_subjects=int(heldout.subject.nunique()),
        other_insula_sites_in_original_map_rule=int(heldout.roi.isin(['AIC', 'PIC']).sum()),
        phases=manifest['phases'], n_features=399, basis_sha256=manifest['basis_sha256'],
        original_fixed_H_NNLS_and_surface_rules_preserved=True,
        no_claim_of_pairwise_connectivity=True,
        discovery_sites_absent_from_packaged_projection=absent.channel.tolist(),
        trial_counts='Not available in trial-averaged source CSVs; no trial count invented')
    (ROOT / 'results/fig6/method_validation.json').write_text(json.dumps(f6, indent=2) + '\n')
    changes = []
    preserved_plot_cells = {}
    plot_indices = {
        3: [8, 10, 12, 15, 17, 23, 28, 32, 34, 36, 38, 39, 40],
        4: [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 25, 27],
        5: [6, 7, 11, 13, 15],
        6: [2, 3, 6, 8],
    }
    for number in range(3, 7):
        nb = json.loads((ROOT / f'vizpub/fig{number}.ipynb').read_text())
        old = {c['id']: c for c in baseline['original_notebook_sources'][str(number)]}
        original_cells = baseline['original_notebook_sources'][str(number)]
        for index in plot_indices[number]:
            assert ''.join(original_cells[index]['source']) == ''.join(nb['cells'][index]['source']), (number, index)
        preserved_plot_cells[str(number)] = [original_cells[index]['id'] for index in plot_indices[number]]
        for cell in nb['cells']:
            prior = old.get(cell.get('id'))
            if cell['cell_type'] == 'code' and (prior is None or ''.join(prior['source']) != ''.join(cell['source'])):
                changes.append(dict(figure=number, cell_id=cell.get('id'),
                    original_source=prior['source'] if prior else None, current_source=cell['source']))
    (OUT / 'changed_notebook_sources.json').write_text(json.dumps(changes, indent=2, ensure_ascii=False) + '\n')
    result = dict(status='partially_complete_missing_original_epochs',
        completed=dict(figure3='5 original statistical/spatial SVGs; remaining waveforms/heatmaps pending', figure6='full notebook and 3 SVGs'),
        pending=dict(figure3='Original single-trial zscore/sig files required for full waveforms and heatmaps',
            figure4='Input preparation failed before any functional classifier fit',
            figure5='Approved cohort/SEM implementation verified; original epochs required before model fitting'),
        figures=receipts, figure6_validation=f6, approved_method_checks=checks,
        frozen_figure1_2_and_nmf_unchanged=True,
        original_plot_cells_unchanged=preserved_plot_cells,
        missing_inputs_csv=str(OUT / 'missing_original_inputs.csv'),
        missing_input_summary_csv=str(OUT / 'original_epoch_availability_summary.csv'),
        no_formal_decoding_or_RT_results_substituted=True,
        held_model_arrays_cancelled=['58872407', '58872408'])
    (OUT / 'delivery_status.json').write_text(json.dumps(result, indent=2) + '\n')
    for number, patterns in ((3, ['available_*.csv']), (6, ['*_membership.csv', '*_counts.csv', '*validation.json', 'discovery_sites_not_in_projection.csv'])):
        archive = OUT / f'fig{number}_tables'
        archive.mkdir(exist_ok=True)
        for pattern in patterns:
            for source in (ROOT / f'results/fig{number}').glob(pattern):
                shutil.copy2(source, archive / source.name)
    print(json.dumps({k: v for k, v in result.items() if k != 'figures'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
