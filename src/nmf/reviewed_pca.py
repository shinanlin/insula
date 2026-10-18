"""PCA/k-means check with the original per-PC variance threshold."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from src.main_analysis import PCA_VARIANCE_THRESHOLD
from src.nmf.pc_clustering import n_pcs_above_threshold


def update_pca_outputs(results_dir: Path):
    data=np.load(results_dir/'input_matrix.npz')
    assignments=pd.read_csv(results_dir/'channel_assignments.csv')
    assert assignments.channel.tolist()==data['channels'].tolist()
    pca=PCA(svd_solver='full').fit(data['X'])
    scores=pca.transform(data['X'])
    n_embed=n_pcs_above_threshold(pca.explained_variance_ratio_,threshold=PCA_VARIANCE_THRESHOLD)
    table=assignments[['channel','subject','functional_cluster']].copy()
    for i in range(n_embed):table[f'PC{i+1}']=scores[:,i]
    table.to_csv(results_dir/'pc_scores.csv',index=False)
    km=KMeans(n_clusters=3,n_init=50,random_state=42).fit_predict(scores[:,:n_embed])
    metadata=dict(n_embedding_pcs=n_embed,variance_threshold=PCA_VARIANCE_THRESHOLD,
        embedding_rule='original rule: each retained PC explains >5%; original minimum-two-PC fallback',
        random_state=42,explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        input='same rectified L2-normalized three-phase NMF matrix',n_sites=len(assignments),
        nmf_kmeans_ari=adjusted_rand_score(assignments.component,km))
    (results_dir/'pc_clustering_meta.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata,km
