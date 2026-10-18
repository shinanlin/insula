#!/usr/bin/env python3
"""Fit the reviewed three-phase k=3 NMF and assess rank without changing the requested k."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.nmf.waveform_analysis import fit_one_nmf, CLUSTER_ORDER
from src.nmf.rank_selection import run_rank_selection, write_rank_selection_outputs
from src.nmf.reviewed_pca import update_pca_outputs
OUT = ROOT/'results/nmf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_response_windows(reference, data, features, assignments):
    """Verify the window-only change and compare labels on the identical cohort."""
    previous = np.load(reference/'input_matrix.npz')
    previous_features = pd.read_csv(reference/'input_features.csv')
    previous_manifest = json.loads((reference/'nmf_manifest.json').read_text())
    current_manifest = json.loads((OUT/'input_manifest.json').read_text())
    assert previous_manifest['fit_windows']['response'] == [0., 1.]
    assert current_manifest['fit_windows']['response'] == [-.5, 1.]
    assert np.array_equal(previous['channels'], data['channels'])
    previous_index = pd.MultiIndex.from_frame(previous_features[['phase','time']])
    current_index = pd.MultiIndex.from_frame(features[['phase','time']])
    positions = current_index.get_indexer(previous_index)
    assert np.all(positions >= 0), 'A previous feature was lost'
    assert np.allclose(previous['raw'], data['raw'][:,positions], atol=1e-12, rtol=0)
    added = current_index.difference(previous_index)
    assert len(added) == 64 and all(p == 'response' and -.5 <= t < 0 for p,t in added)
    assert len(features) == 399
    previous_waves = pd.read_csv(reference/'site_waveforms.csv.gz')
    current_waves = pd.read_csv(OUT/'site_waveforms.csv.gz')
    keys = ['subject','channel','phase','time']
    left = previous_waves.set_index(keys).sort_index()
    right = current_waves.set_index(keys).sort_index()
    pd.testing.assert_index_equal(left.index, right.index)
    assert np.allclose(left.value, right.value, atol=1e-12, rtol=0)
    previous_sites = pd.read_csv(reference/'channel_assignments.csv')
    comparison = assignments[['channel','subject','functional_cluster']].rename(
        columns={'functional_cluster':'new_cluster'}).merge(
            previous_sites[['channel','functional_cluster']].rename(
                columns={'functional_cluster':'previous_cluster'}),
            on='channel', validate='one_to_one')
    assert len(comparison) == len(assignments) == len(previous_sites)
    comparison['cluster_changed'] = comparison.new_cluster.ne(comparison.previous_cluster)
    comparison.to_csv(OUT/'comparison_with_response_0_to1.csv', index=False)
    pd.crosstab(comparison.previous_cluster, comparison.new_cluster).reindex(
        index=CLUSTER_ORDER, columns=CLUSTER_ORDER, fill_value=0).to_csv(
            OUT/'response_window_label_transitions.csv')
    result = dict(reference_run=str(reference.resolve()),
        previous_manifest_sha256=sha(reference/'nmf_manifest.json'),
        previous_response_window=[0.,1.], current_response_window=[-.5,1.],
        n_sites=len(comparison), n_subjects=int(comparison.subject.nunique()),
        previous_n_features=int(previous['X'].shape[1]), current_n_features=len(features),
        added_response_samples=len(added), common_raw_features_unchanged=True,
        site_waveforms_unchanged=True, cohort_and_site_order_unchanged=True,
        n_changed=int(comparison.cluster_changed.sum()),
        n_unchanged=int((~comparison.cluster_changed).sum()),
        ari=float(adjusted_rand_score(comparison.previous_cluster, comparison.new_cluster)))
    (OUT/'response_window_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank-only', action='store_true')
    parser.add_argument('--reference-run', type=Path,
                        help='Frozen previous Response 0–1 s run for a window-only comparison')
    args = parser.parse_args()
    data = np.load(OUT/'input_matrix.npz')
    X = data['X']
    features = pd.read_csv(OUT/'input_features.csv')
    manifest = json.loads((OUT/'input_manifest.json').read_text())
    if args.rank_only:
        metrics, consensus, decision = run_rank_selection(X, k_min=2, k_max=6, n_boot=200,
            row_frac=.8, max_iter=5000, random_state=42)
        write_rank_selection_outputs(metrics=metrics, consensus=consensus, decision=decision,
            meta={**manifest, 'requested_model_k':3, 'selected_k_does_not_override_requested_model':True,
                  'input_matrix_sha256':sha(OUT/'input_matrix.npz')},
            results_dir=OUT, images_dir=ROOT/'img/nmf')
        return
    rng = np.random.default_rng(42)
    fits, starts = [], []
    for run, seed in enumerate(rng.integers(0, np.iinfo(np.int32).max, size=20)):
        fit = fit_one_nmf(X, 3, int(seed), max_iter=5000)
        fits.append(fit)
        starts.append(dict(run=run, seed=int(seed), reconstruction_error=fit['error'], converged=fit['converged']))
        print(f"start {run+1}/20: error={fit['error']:.6f}, converged={fit['converged']}", flush=True)
    best_idx = int(np.argmin([f['error'] for f in fits]))
    best = fits[best_idx]
    W, H = np.asarray(best['W']), np.asarray(best['H'])
    phase_means = np.column_stack([H[:,features.phase.eq(p).to_numpy()].mean(axis=1) for p in manifest['phases']])
    shares = phase_means / phase_means.sum(axis=1, keepdims=True)
    component, phase = linear_sum_assignment(-shares)
    phase_names = np.array(['sensory','sustain','motor'])
    names = {int(c):str(phase_names[p]) for c,p in zip(component,phase)}
    profiles = pd.DataFrame(phase_means, columns=manifest['phases'])
    profiles.insert(0,'component',range(3))
    profiles['functional_cluster'] = profiles.component.map(names)
    profiles.to_csv(OUT/'component_phase_profiles.csv', index=False)
    assignments = pd.read_csv(OUT/'input_sites.csv')
    assert assignments.channel.tolist() == data['channels'].tolist()
    labels = W.argmax(axis=1)
    assignments['component'] = labels
    assignments['functional_cluster'] = [names[int(c)] for c in labels]
    for c, name in names.items():
        assignments[f'loading_{name}'] = W[:,c]
    assignments['dominance'] = W.max(axis=1)/W.sum(axis=1)
    p = W/W.sum(axis=1,keepdims=True)
    sorted_p = np.sort(p,axis=1)
    assignments['specificity_max'] = sorted_p[:,-1]-sorted_p[:,-2]
    assignments['is_mixed'] = assignments.specificity_max < .2
    assignments.to_csv(OUT/'channel_assignments.csv', index=False)
    np.savez_compressed(OUT/'factors.npz', W=W, H=H, channels=data['channels'], component_names=np.array([names[i] for i in range(3)]))
    long = []
    for c, name in names.items():
        frame = features.copy()
        frame['component'] = c
        frame['functional_cluster'] = name
        frame['normalized_H'] = H[c]
        long.append(frame)
    pd.concat(long).to_csv(OUT/'H_by_phase.csv', index=False)
    pd.DataFrame(starts).to_csv(OUT/'initialization_metrics.csv', index=False)
    aris = [adjusted_rand_score(a['labels'], b['labels']) for a,b in itertools.combinations(fits,2)]
    pca_metadata,km = update_pca_outputs(OUT)
    old = pd.read_csv('/hpc/group/coganlab/nanlinshi/insula-functional/results/nmf/channel_assignments.csv')
    comparison = assignments[['channel','functional_cluster']].merge(
        old[['channel','functional_cluster']].rename(columns={'functional_cluster':'old_four_phase_cluster'}),on='channel',how='outer',indicator=True)
    comparison['same_name'] = comparison.functional_cluster.eq(comparison.old_four_phase_cluster)
    comparison.to_csv(OUT/'comparison_with_old_four_phase.csv',index=False)
    shared = comparison.loc[comparison['_merge'].eq('both')]
    summary = {**manifest, 'k':3,'n_init':20,'random_state':42,'best_run':best_idx,
        'solver':'coordinate descent, squared Frobenius error, nndsvdar initialization, max_iter=5000, tol=1e-4',
        'reconstruction_error':best['error'],
        'explained_energy':float(1-np.square(X-W@H).sum()/np.square(X).sum()),
        'mean_initialization_ari':float(np.mean(aris)),
        'converged_starts':sum(f['converged'] for f in fits),
        'cluster_counts':assignments.functional_cluster.value_counts().to_dict(),
        'cluster_subject_counts':assignments.groupby('functional_cluster').subject.nunique().to_dict(),
        'n_mixed':int(assignments.is_mixed.sum()),
        'naming':'Hungarian assignment maximizing phase-mean shares: stimulus=sensory, delay=sustain, response=motor; no anatomy or legacy labels used',
        'assignment':'argmax W after unit L2 normalization of each full concatenated H component',
        'old_model_shared_sites':len(shared),
        'old_model_shared_ari':float(adjusted_rand_score(shared.functional_cluster,shared.old_four_phase_cluster)),
        'nmf_kmeans_ari':float(adjusted_rand_score(labels,km)),
        'pca_embedding_rule':pca_metadata['embedding_rule'],
        'pca_variance_threshold':pca_metadata['variance_threshold'],
        'pca_embedding_pcs':pca_metadata['n_embedding_pcs'],
        'factor_H_unit_norm':bool(np.allclose(np.linalg.norm(H,axis=1),1)),
        'outputs':{p.name:sha(p) for p in [OUT/'channel_assignments.csv',OUT/'factors.npz',OUT/'H_by_phase.csv']}}
    if manifest.get('modality')=='sound':
        previous=pd.read_csv(ROOT/'docs/evidence/figure2_response_window_2026-10-17/channel_assignments.csv')
        delta=assignments[['channel','subject','functional_cluster']].rename(columns={'functional_cluster':'sound_only_cluster'}).merge(
            previous[['channel','functional_cluster']].rename(columns={'functional_cluster':'previous_multimodal_cluster'}),
            on='channel',how='outer',indicator=True,validate='one_to_one')
        delta.to_csv(OUT/'comparison_with_previous_multimodal.csv',index=False)
        shared_delta=delta.loc[delta['_merge'].eq('both')]
        summary['previous_multimodal_comparison']=dict(shared_sites=len(shared_delta),
            removed_from_previous=int(delta['_merge'].eq('right_only').sum()),
            changed_labels=int(shared_delta.sound_only_cluster.ne(shared_delta.previous_multimodal_cluster).sum()),
            ari=float(adjusted_rand_score(shared_delta.sound_only_cluster,shared_delta.previous_multimodal_cluster)),
            interpretation='Selection and discovery are both now sound-only; this is not a window-only comparison')
        for name in ['response_window_comparison.json','comparison_with_response_0_to1.csv','response_window_label_transitions.csv']:
            (OUT/name).unlink(missing_ok=True)
    if args.reference_run is not None:
        summary['response_window_comparison'] = compare_response_windows(
            args.reference_run, data, features, assignments)
    (OUT/'nmf_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(profiles.to_string(index=False),flush=True)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['outputs','supplemented_epochs']},indent=2),flush=True)


if __name__ == '__main__': main()
