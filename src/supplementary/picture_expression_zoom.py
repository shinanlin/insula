"""Display-only linear-y enlargement of saved paired PN expression results."""
import json
import os

from src.supplementary.picture_fixed_template_expression import (
    ROOT, DATA, IMG, COMPONENTS, GROUPS, _plot,
)

AUDIT = ROOT / 'results/audit/picture_expression_zoom_2026-10-27'
EXPORT_NAME = 'fig3_picture_response_sets_paired_expression_zoom.svg'
ZOOM_UPPER = 10.0


def notebook_cells():
    import nbformat as nbf
    return [nbf.v4.new_markdown_cell('''### 4a. Amplitude-preserving expression — enlarged 0–10 view

This is a **display-only enlargement** of the preceding full-range amplitude figure. The three components share a **linear 0–10 y-range** (with the same small padding below zero). All 49 Sound/Image electrode pairs, their exact saved coefficients, electrode-equal means and previously computed bootstrap 95% CIs are reused. All mean CIs fit within this enlarged range.

Individual Sound values above 10 are indicated by filled upward triangles; Image values above 10 by open upward triangles at the upper boundary. The second line of each component title counts these off-scale observations (electrode × condition values, not unique electrodes). Their true values and the complete range remain in the unchanged original figure. Lines and error bars use the same data, positions, colors and pairing; no observations are removed from any mean, CI or analysis.

The enlargement reads only the saved coefficient arrays and summary CSV. It does **not** rerun NNLS, NMF, normalization, bootstrap or any statistical test. The preceding full-range and normalized-weight SVGs remain unchanged.
''', id='picture-expression-zoom-note'), nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG, display
_pn_zoom_root = Path('/cwork/ns458/insula-paper')
if str(_pn_zoom_root) not in sys.path:
    sys.path.insert(0, str(_pn_zoom_root))
from src.supplementary.picture_expression_zoom import render_saved_zoom
expression_zoom_path = render_saved_zoom()
display(SVG(filename=str(expression_zoom_path)))
''', id='picture-expression-zoom-display')]


def render_saved_zoom():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    import numpy as np
    import pandas as pd
    from scripts.refresh_lexical_condition_sets import sha, write
    AUDIT.mkdir(parents=True, exist_ok=True)
    sources = [DATA / name for name in ['sites.csv', 'projection_arrays.npz', 'coefficients.csv', 'group_condition_summary.csv']]
    before = {str(path):sha(path) for path in sources}
    sites = pd.read_csv(DATA / 'sites.csv')
    summary = pd.read_csv(DATA / 'group_condition_summary.csv', float_precision='round_trip')
    coefficients = pd.read_csv(DATA / 'coefficients.csv', float_precision='round_trip')
    with np.load(DATA / 'projection_arrays.npz', allow_pickle=False) as arrays:
        weights = arrays['expression_coefficients'].copy()
        names = tuple(arrays['component_names'].tolist())
        np.testing.assert_array_equal(arrays['electrodes'], sites.electrode.to_numpy())
    assert weights.shape == (49,2,3) and len(sites) == 49 and sites.subject.nunique() == 22
    assert sites.response_set.value_counts().to_dict() == {'Sound-only':25,'Both':13,'Picture-only':11}
    assert np.isfinite(weights).all()
    saved = coefficients.set_index(['electrode','condition','component']).expression_coefficient
    plotted = []
    for i,site in enumerate(sites.itertuples()):
        for condition_index, condition in enumerate(['Sound','Image']):
            for k, component in enumerate(names):
                value = weights[i, condition_index, k]
                assert value == saved.loc[(site.electrode,condition,component)], 'Saved coefficient disagreement'
                plotted.append(dict(electrode=site.electrode, response_set=site.response_set,
                    condition=condition,component=component,coefficient=value,
                    above_zoom=bool(value > ZOOM_UPPER),retained_in_existing_summary=True))
    metrics = summary.loc[summary.metric.eq('amplitude')]
    assert len(metrics) == 18 and metrics.n_sites.sum() == 294
    assert metrics.ci_upper.max() < ZOOM_UPPER and metrics.ci_lower.min() >= 0
    path = _plot(sites,weights,names,summary,'amplitude',display_upper=ZOOM_UPPER,export_name=EXPORT_NAME)
    assert all(sha(p)==h for p,h in before.items()), 'A numerical source changed'
    points = pd.DataFrame(plotted)
    points.to_csv(AUDIT / 'plotted_observations.csv',index=False)
    counts = points.groupby(['component','response_set','condition'],sort=False).agg(
        n_observations=('coefficient','size'), n_above_zoom=('above_zoom','sum')).reset_index()
    counts.to_csv(AUDIT / 'off_scale_counts.csv',index=False)
    write(AUDIT / 'display_provenance.json',dict(
        display_only=True, sources=before, unchanged_coefficient_checks=294,
        n_sites=49,n_participants=22,groups={g:int(sites.response_set.eq(g).sum()) for g in GROUPS},
        linear_zoom_range=[0,ZOOM_UPPER],shared_y_across_components=True,
        all_saved_means_and_CI_reused=True,all_mean_CI_within_zoom=True,
        off_scale_observations=int(points.above_zoom.sum()),
        off_scale_by_component={c:int(points.loc[points.component.eq(c),'above_zoom'].sum()) for c in COMPONENTS},
        no_refitting_no_bootstrap_no_data_changes=True,export=str(path),export_sha256=sha(path)))
    return path
