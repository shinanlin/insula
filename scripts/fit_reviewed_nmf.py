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
OUT = ROOT/'results/nmf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank-only', action='store_true')
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
    pca = PCA(svd_solver='full').fit(X)
    scores = pca.transform(X)
    n_embed = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .9)+1)
    score_table = assignments[['channel','subject','functional_cluster']].copy()
    for i in range(n_embed): score_table[f'PC{i+1}'] = scores[:,i]
    score_table.to_csv(OUT/'pc_scores.csv', index=False)
    km = KMeans(n_clusters=3,n_init=50,random_state=42).fit_predict(scores[:,:n_embed])
    (OUT/'pc_clustering_meta.json').write_text(json.dumps(dict(n_embedding_pcs=n_embed,
        random_state=42, explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        input='same rectified L2-normalized three-phase NMF matrix', n_sites=len(X),
        nmf_kmeans_ari=adjusted_rand_score(labels,km)), indent=2)+'\n')
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
        'factor_H_unit_norm':bool(np.allclose(np.linalg.norm(H,axis=1),1)),
        'outputs':{p.name:sha(p) for p in [OUT/'channel_assignments.csv',OUT/'factors.npz',OUT/'H_by_phase.csv']}}
    (OUT/'nmf_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(profiles.to_string(index=False),flush=True)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['outputs','supplemented_epochs']},indent=2),flush=True)


if __name__ == '__main__': main()
