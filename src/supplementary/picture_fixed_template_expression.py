"""Paired PN expression of the existing frozen temporal NMF dictionary."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os

ROOT = Path('/cwork/ns458/insula-paper')
BASE = ROOT / 'results/supplementary/fig3_picture_stimulus_condition_sets'
DATA = BASE / 'fixed_template_expression_2026-10-27'
AUDIT = ROOT / 'results/audit/picture_fixed_template_expression_2026-10-27'
IMG = ROOT / 'img/supplementary/fig3_picture_stimulus_condition_sets'
GROUPS = ('Picture-only', 'Sound-only', 'Both')
COMPONENTS = ('sensory', 'sustain', 'motor')
LABELS = dict(sensory='Sensory', sustain='Sustained', motor='Motor')
COLORS = dict(sensory='#2369BD', sustain='#C4A35A', motor='#A9373B')
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20261027
EXPORT_NAMES = {
    'amplitude': 'fig3_picture_response_sets_paired_expression.svg',
    'normalized': 'fig3_picture_response_sets_paired_normalized_weight.svg',
}


def notebook_cells():
    import nbformat as nbf
    note = nbf.v4.new_markdown_cell('''## 4. Paired Sound / Image expression of frozen NMF components

**Draft, using all 49 response-union sites / 22 participants.** Groups remain the native Stimulus **0–1 s** baseline-response sets: Picture-only 11 / 8 participants, Sound-only 25 / 17, Both 13 / 10. These overlapping participant counts must not be added. Memberships, previous significance masks and frozen functional labels are unchanged. The five originally Unassigned sites receive continuous projection coefficients here and retain their Unassigned labels elsewhere.

First audit all **294 site × condition × phase cells** using existing native-QC-kept Repeat Sound / Image z-scored HGA epochs. Exact-Stimulus repaired sources take precedence. Retain native trials and their original trial-wise missing samples; average finite native trials within each site, condition and time bin, without adding a significance mask, smoothing or an event/QC change. Never substitute another condition or fill a missing phase. Source hashes, native event names, finite counts and each cell's status are saved beside the extracted means.

Freeze the existing temporal dictionary **H**, with each temporal component normalized to unit L2 norm, and independently solve **nonnegative least squares, X ≈ W H**, for each site and condition (ridge = 0). Projection uses H's exact 399-feature grid: Stimulus 0–0.6 s (77 samples), Delay 0–1 s (129), Response −0.5–1 s (193). This differs deliberately from the **0–1 s group-definition window**. Each coefficient summarizes all three phases jointly; it is not a separate fit for each phase. No NMF model or label is refitted.

**Primary: amplitude-preserving expression coefficient.** Clip negative trial-mean HGA values at zero, as in the frozen discovery preprocessing, but omit the discovery input's per-waveform amplitude normalization. Thus native HGA amplitude remains in the coefficients (HGA is already z-scored; these are not physical-voltage units).

**Auxiliary: normalized waveform weight.** Divide the same concatenated nonnegative waveform by its own L2 norm, then use the identical frozen H. These weights summarize waveform shape, do not sum to one and are not variance percentages. A zero-energy waveform has zero primary expression but undefined auxiliary weight; no such record may be silently imputed or deleted.

Each thin line pairs Sound (filled circle) and Image (open circle) for one electrode. Large diamonds are **electrode-equal means**; intervals are percentile **95% subject-cluster bootstrap CIs**, using 10,000 joint resamples of the 22 participants (fixed seed 20261027), carrying all of each participant's sites and paired conditions together. Means are not participant-equal. All components share one y-scale within a metric. The paired-difference summary is descriptive; no new significance stars or condition-difference p-values are added in this first draft.

The audit, coefficient list, reconstruction quality and Sound/Image paired differences are exported as CSVs. Reconstruction quality is reported without excluding any low-fit site. These are descriptive, response-selected comparisons using Sound-trained templates; they do not establish independent validation or a statistically significant difference between response sets.
''', id='picture-fixed-template-expression-note')
    code = nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG, display
_pn_expression_root = Path('/cwork/ns458/insula-paper')
if str(_pn_expression_root) not in sys.path:
    sys.path.insert(0, str(_pn_expression_root))
from src.supplementary.picture_fixed_template_expression import project_and_render, DATA as expression_data
expression_paths = project_and_render()
display(SVG(filename=str(expression_paths['amplitude'])))
display(SVG(filename=str(expression_paths['normalized'])))
import pandas as pd
display(pd.read_csv(expression_data / 'group_counts.csv'))
display(pd.read_csv(expression_data / 'paired_difference_summary.csv'))
print('Detailed completeness audit:', expression_data / 'data_completeness.csv')
print('Fit diagnostics:', expression_data / 'fit_quality.csv')
print('Coefficient list:', expression_data / 'coefficients.csv')
''', id='picture-fixed-template-expression-pairs')
    return [note, code]


def _bootstrap_summary(sites, primary, normalized, component_names):
    import numpy as np
    import pandas as pd
    subjects = sorted(sites.subject.unique())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # One draw set is shared across components, conditions, groups and metrics.
    multiplicities = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), size=N_BOOTSTRAP)
    summaries, differences = [], []
    for metric, weights in [('amplitude', primary), ('normalized', normalized)]:
        for group in GROUPS:
            selected = sites.response_set.eq(group).to_numpy()
            n_subjects = sites.loc[selected, 'subject'].nunique()
            for k, name in enumerate(component_names):
                paired_valid = selected & np.isfinite(weights[:, :, k]).all(axis=1)
                counts = np.array([np.sum(paired_valid & sites.subject.eq(s).to_numpy()) for s in subjects])
                sums = np.stack([np.nansum(weights[paired_valid & sites.subject.eq(s).to_numpy(), :, k], axis=0) for s in subjects])
                denom = multiplicities @ counts
                good = denom > 0
                draws = (multiplicities[good] @ sums) / denom[good, None]
                observed = weights[paired_valid, :, k].mean(axis=0)
                for condition, label in enumerate(['Sound', 'Image']):
                    lower, upper = np.quantile(draws[:, condition], [.025, .975])
                    summaries.append(dict(metric=metric, response_set=group, component=name, condition=label,
                        n_sites=int(paired_valid.sum()), nominal_n_sites=int(selected.sum()), n_participants=int(n_subjects),
                        mean=float(observed[condition]), ci_lower=float(lower), ci_upper=float(upper),
                        n_bootstrap=N_BOOTSTRAP, finite_draws=int(good.sum()), bootstrap_seed=BOOTSTRAP_SEED))
                diff = draws[:, 1] - draws[:, 0]
                lower, upper = np.quantile(diff, [.025, .975])
                differences.append(dict(metric=metric, response_set=group, component=name,
                    n_sites=int(paired_valid.sum()), n_participants=int(n_subjects),
                    mean_image_minus_sound=float(observed[1]-observed[0]),
                    ci_lower=float(lower), ci_upper=float(upper), inference='Descriptive; no condition-difference p-value'))
    return pd.DataFrame(summaries), pd.DataFrame(differences)


def _plot(sites, weights, component_names, summary, metric, *, display_upper=None, export_name=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.5), sharey=True)
    metric_summary = summary.loc[summary.metric.eq(metric)]
    max_value = float(max(np.nanmax(weights), metric_summary.ci_upper.max()))
    upper = float(display_upper) if display_upper is not None else (max_value * 1.08 if max_value > 0 else 1)
    if display_upper is not None:
        assert upper > 0 and metric == 'amplitude'
        assert metric_summary.ci_upper.max() < upper, 'The zoom must retain all mean confidence intervals'
    jitter = np.random.default_rng(20261027).uniform(-.055, .055, len(sites))
    for ax, component in zip(axes, COMPONENTS):
        k = component_names.index(component)
        color = COLORS[component]
        for group_index, group in enumerate(GROUPS):
            indices = np.flatnonzero(sites.response_set.eq(group).to_numpy())
            for index in indices:
                x = np.array([group_index-.14, group_index+.14]) + jitter[index]
                y = weights[index, :, k]
                if np.isfinite(y).all():
                    ax.plot(x, y, color=color, alpha=.3, lw=.6, zorder=1)
                    ax.plot(x[0], y[0], 'o', ms=3.5, color=color, alpha=.7, mew=.6, zorder=2)
                    ax.plot(x[1], y[1], 'o', ms=3.5, color=color, mfc='white', alpha=.85, mew=.85, zorder=2)
                    if display_upper is not None:
                        for condition_index in range(2):
                            if y[condition_index] > upper:
                                ax.plot(x[condition_index], upper*.982, marker='^', ms=4.5, color=color,
                                        mfc=color if condition_index == 0 else 'white', mew=.8, zorder=3)
            for condition, label in enumerate(['Sound', 'Image']):
                row = metric_summary.loc[metric_summary.response_set.eq(group) & metric_summary.component.eq(component)
                                         & metric_summary.condition.eq(label)].iloc[0]
                x = group_index + [-.14, .14][condition]
                # Percentile intervals can be asymmetric and need not contain the point estimate.
                ax.plot([x, x], [row.ci_lower, row.ci_upper], color='black', lw=1.3, zorder=4)
                ax.plot([x-.028, x+.028], [row.ci_lower]*2, color='black', lw=1, zorder=4)
                ax.plot([x-.028, x+.028], [row.ci_upper]*2, color='black', lw=1, zorder=4)
                ax.plot(x, row['mean'], marker='D', ms=6, color='black',
                        mfc=color if condition == 0 else 'white', mew=1, zorder=5)
        ticks = []
        for group in GROUPS:
            chosen = sites.loc[sites.response_set.eq(group)]
            ticks.append(f'{group}\n{len(chosen)} sites\n{chosen.subject.nunique()} participants')
        ax.set_xticks(range(3), ticks, fontsize=8)
        ax.set_xlim(-.48, 2.48); ax.set_ylim(-upper*.025, upper)
        title = LABELS[component]
        if display_upper is not None:
            n_off_scale = int(np.sum(weights[:, :, k] > upper))
            title += f'\n{n_off_scale} observations above {upper:g}'
        ax.set_title(title, color=color, fontsize=11, fontweight='bold')
        ax.axhline(0, color='.65', lw=.6, zorder=0)
        ax.tick_params(direction='out', length=3)
    axes[0].set_ylabel('HGA expression coefficient' if metric == 'amplitude' else 'L2-normalized waveform weight')
    name = 'Amplitude-preserving expression' if metric == 'amplitude' else 'Normalized waveform weight · auxiliary'
    if display_upper is not None:
        name += f' · zoom 0–{upper:g}'
    fig.suptitle(f'Picture Naming · {name}', fontsize=12, y=.98)
    legend = [Line2D([], [], color='.4', marker='o', lw=.7, mfc='.4', ms=4, label='Sound'),
              Line2D([], [], color='.4', marker='o', lw=.7, mfc='white', ms=4, label='Image'),
              Line2D([], [], color='black', marker='D', lw=1, mfc='white', ms=5, label='Electrode mean + 95% CI')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.51,.91), ncol=3, frameon=False, fontsize=8)
    fig.text(.5, .05, 'Each line: one paired electrode. CI: participant-cluster bootstrap. Groups: Stimulus 0–1 s baseline responses.',
             ha='center', fontsize=8)
    if display_upper is not None:
        fig.text(.5, .015, f'Triangles indicate values above {upper:g}; see the unchanged full-range figure. No observations or CI recomputed.',
                 ha='center', fontsize=8)
    fig.subplots_adjust(left=.08, right=.99, bottom=.23, top=.75, wspace=.12)
    IMG.mkdir(parents=True, exist_ok=True)
    target = IMG / (export_name or EXPORT_NAMES[metric])
    fig.savefig(target, format='svg', bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return target


def project_and_render():
    assert os.environ.get('SLURM_JOB_ID'), 'Run scientific processing through Slurm'
    import numpy as np
    import pandas as pd
    from scripts.refresh_lexical_condition_sets import protected, sha, write
    from scripts.refresh_picture_condition_sets import protected as picture_protected
    from src.nmf.whole_brain_projection import load_temporal_basis, project_fixed_temporal_basis
    protected(); picture_protected()
    audit = json.loads((AUDIT / 'data_audit.json').read_text())
    assert audit['passed'] and audit['complete_pairs'] == 49 and audit['complete_cells'] == 294
    assert all(sha(DATA / p) == h for p, h in audit['outputs'].items()), 'Audited inputs changed'
    basis = load_temporal_basis(ROOT / 'results/nmf/H_by_phase.csv')
    assert basis.source_sha256 == audit['fixed_H_sha256']
    sites = pd.read_csv(DATA / 'sites.csv')
    with np.load(DATA / 'audited_native_means.npz', allow_pickle=False) as arrays:
        raw = arrays['raw_means'].copy()
        np.testing.assert_array_equal(arrays['electrodes'], sites.electrode.to_numpy())
        np.testing.assert_array_equal(arrays['conditions'], ['sound', 'image'])
        np.testing.assert_allclose(arrays['H'], basis.H, rtol=0, atol=0)
    assert raw.shape == (49, 2, 399) and np.isfinite(raw).all()
    positive = np.clip(raw, 0, None)
    X = positive.reshape(98, 399)
    norms = np.linalg.norm(X, axis=1)
    weights, reconstructed = project_fixed_temporal_basis(X, basis.H, ridge=0)
    valid = norms > np.finfo(float).eps
    normalized = np.full_like(weights, np.nan)
    shape_reconstructed = np.full_like(reconstructed, np.nan)
    normalized[valid], shape_reconstructed[valid] = project_fixed_temporal_basis(X[valid] / norms[valid,None], basis.H, ridge=0)
    # NNLS homogeneity checks that the auxiliary differs only by the approved input scale.
    np.testing.assert_allclose(normalized[valid], weights[valid] / norms[valid,None], atol=1e-9, rtol=1e-8)
    primary = weights.reshape(49, 2, 3)
    auxiliary = normalized.reshape(49, 2, 3)
    coefficient_rows, fits = [], []
    for site_index, site in enumerate(sites.itertuples()):
        for condition_index, condition in enumerate(['Sound', 'Image']):
            index = site_index * 2 + condition_index
            for component_index, component in enumerate(basis.component_names):
                coefficient_rows.append(dict(electrode=site.electrode, subject=site.subject, channel=site.channel,
                    response_set=site.response_set, frozen_functional_cluster=site.functional_cluster,
                    condition=condition, component=component,
                    expression_coefficient=float(primary[site_index, condition_index, component_index]),
                    normalized_weight=float(auxiliary[site_index, condition_index, component_index]),
                    positive_waveform_l2=float(norms[index]), auxiliary_defined=bool(valid[index])))
            sse = float(np.square(X[index]-reconstructed[index]).sum())
            energy = float(np.square(X[index]).sum())
            reconstruction_norm = float(np.linalg.norm(reconstructed[index]))
            fits.append(dict(electrode=site.electrode, subject=site.subject, condition=condition,
                response_set=site.response_set, frozen_functional_cluster=site.functional_cluster,
                positive_l2=norms[index], residual_l2=float(np.sqrt(sse)),
                explained_positive_energy=1-sse/energy if energy > 0 else np.nan,
                cosine_similarity=float(np.dot(X[index], reconstructed[index])/(norms[index]*reconstruction_norm))
                    if norms[index]*reconstruction_norm > 0 else np.nan,
                auxiliary_defined=bool(valid[index]), included_without_fit_threshold=True))
    pd.DataFrame(coefficient_rows).to_csv(DATA / 'coefficients.csv', index=False)
    fit_table = pd.DataFrame(fits)
    fit_table.to_csv(DATA / 'fit_quality.csv', index=False)
    counts = sites.groupby('response_set').agg(n_sites=('electrode','size'), n_participants=('subject','nunique'),
                                              original_unassigned=('functional_cluster',lambda x: int(x.eq('Unassigned').sum())))
    counts.reindex(GROUPS).reset_index().to_csv(DATA / 'group_counts.csv', index=False)
    summary, differences = _bootstrap_summary(sites, primary, auxiliary, basis.component_names)
    summary.to_csv(DATA / 'group_condition_summary.csv', index=False)
    differences.to_csv(DATA / 'paired_difference_summary.csv', index=False)
    quality_summary = fit_table.groupby(['response_set','condition']).explained_positive_energy.agg(['min','median','max'])
    quality_summary.to_csv(DATA / 'fit_quality_summary.csv')
    np.savez_compressed(DATA / 'projection_arrays.npz', raw_means=raw, positive_means=positive,
        expression_coefficients=primary, normalized_weights=auxiliary,
        reconstruction=reconstructed.reshape(49,2,399), component_names=np.array(basis.component_names),
        electrodes=sites.electrode.to_numpy(dtype=str), H=basis.H)
    paths = {metric: _plot(sites, array, basis.component_names, summary, metric)
             for metric, array in [('amplitude', primary), ('normalized', auxiliary)]}
    frozen = json.loads((AUDIT / 'frozen_nmf.json').read_text())
    assert all(sha(p) == h for p, h in frozen.items()), 'Frozen NMF changed'
    protected(); picture_protected()
    write(DATA / 'projection_provenance.json', dict(
        task='PictureNaming', description='Repeat', modalities=['sound','image'],
        n_sites=49, n_participants=22, n_paired_fits=98, n_coefficients=294,
        grouping_window=[0,1], projection_windows=audit['projection_windows'], projection_phase_samples=audit['projection_phase_samples'],
        fixed_H_path=str(basis.source_path), fixed_H_sha256=basis.source_sha256,
        component_order=list(basis.component_names), H_row_norm='unit L2',
        method='NNLS of each PN-only positive trial-mean waveform onto frozen temporal H; ridge=0',
        primary='No input L2 normalization; HGA zscore amplitude retained',
        auxiliary='Same concatenated positive waveform independently L2-normalized per site and condition',
        zero_energy_waveforms=int((~valid).sum()), no_imputed_or_refitted_labels=True,
        no_new_hypothesis_tests=True, no_fit_threshold_exclusions=True,
        error_intervals=dict(kind='Percentile subject-cluster bootstrap', confidence=.95,
            n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, point_estimate='electrode-equal mean',
            shared_draws_across_conditions_groups_components_and_metrics=True),
        fit_quality=quality_summary.reset_index().to_dict('records'),
        exports={str(p):sha(p) for p in paths.values()}, job_id=os.environ['SLURM_JOB_ID'],
        completed_utc=datetime.now(timezone.utc).isoformat()))
    return paths
