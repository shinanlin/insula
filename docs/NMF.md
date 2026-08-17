# NMF

Canonical functional clustering is **concatenated multi-phase NMF with
post-onset crop windows**, after dropping channels listed in
``results/nmf/exclude_channels.txt``.

Cluster names (stimulus-segment early−late orientation):

| label | role |
|---|---|
| `sustain` | holds / ramps (lowest transient score) |
| `motor` | middle profile |
| `sensory` | brief / sensory-weighted (highest transient score) |

## Pipeline entry points

| Step | Command |
|---|---|
| Rank selection (bootstrap) | `scripts/run_nmf_rank_selection.py` / `sbatch scripts/slurm/nmf_rank_bootstrap.sh` |
| Fit + canonical SVGs | `scripts/plot_nmf_concat_phases.py` / `sbatch scripts/slurm/nmf_concat.sh` |
| Waveform PCA scatter | `scripts/plot_nmf_waveform_pca.py` / `sbatch scripts/slurm/nmf_waveform_pca.sh` |
| PC scree + PC-space clustering (tables) | `scripts/run_nmf_pc_clustering.py` / `sbatch scripts/slurm/nmf_pc_clustering.sh` |
| PC clustering figures | `notebooks/nmf_pc_clustering.ipynb` (reads tables, writes SVGs) |
| Helpers | `src/nmf/waveform_analysis.py`, `src/nmf/rank_selection.py`, `src/nmf/waveform_pca.py`, `src/nmf/pc_clustering.py`, `src.paths.nmf_assignments_path()` |

Canonical figures under ``img/nmf/``: ``waveforms.svg``, ``H_overview.svg``,
``spatial_yz.svg``, ``waveform_pca.svg``. Rank-selection figures are also flat
under ``img/nmf/`` (``rank_metrics.svg``, ``consensus_k*.svg``, …). All figures
are **SVG only**.

---

## 1. What is decomposed

- **Input matrix `X`:** electrodes × concatenated time from packaged Hammers HGA
  (``results/hga/{Task}/…/*desc-Repeat_time.csv``).
- **Tasks:** PhonemeSequence, LexicalDelay, PictureNaming, SentenceRep (no
  LexicalNoDelay).
- **Electrodes:** sig-union ∩ pure AIC/PIC (`mix=False`), minus
  ``exclude_channels.txt``.
- **Windows (post-onset):** stimulus/delay/go `(0,1)`, response `(0,0.5)`.
- **Prep:** `clip≥0`, row L2-normalize.
- **NMF:** `sklearn.decomposition.NMF(init="nndsvdar", solver="cd", tol=1e-4,
  max_iter=5000)`.

---

## 2. Choosing k (electrode bootstrap consensus)

Init-to-init ARI under `nndsvdar` is near-deterministic and is **not** used for
rank choice.

Protocol (`src/nmf/rank_selection.py`):

- For each `k ∈ {2,…,6}`: `B=200` electrode subsamples (80% of rows, without
  replacement); fit NMF; accumulate co-clustering consensus matrix `C`.
- **Primary:** cophenetic correlation of `1−C` (max wins; ties → smaller k).
- **Secondary (full matrix, each k):** reconstruction error, explained energy
  \(1-\|X-WH\|_F^2/\|X\|_F^2\), cosine silhouette; plus mean pairwise bootstrap ARI.

Near-ties: report all k within 0.02 cophenetic of the winner.

Flat outputs:

```text
results/nmf/
  rank_selection_metrics.csv
  chosen_k.json
  rank_selection_meta.json
  consensus_k{k}.npy
img/nmf/
  rank_metrics.svg
  cophenetic_vs_k.svg
  mean_ari_vs_k.svg
  reconstruction_vs_k.svg
  explained_energy_vs_k.svg
  consensus_k{k}.svg
```

---

## 3. Canonical fit outputs

```text
results/nmf/
  exclude_channels.txt
  channel_assignments.csv
  H_by_phase.csv
  nmf_manifest.json
  waveform_pca_scores.csv
  waveform_pca_meta.json
  pc_scree.csv
  pc_scores.csv
  pc_clustering_iterations.csv
  pc_clustering_metrics.csv
  pc_clustering_meta.json
  ...
img/nmf/
  waveforms.svg
  H_overview.svg
  spatial_yz.svg
  waveform_pca.svg
  pca_variance.svg
  pc_cluster_metrics.svg
  nmf_rank_k.svg
  waveform_pca_kmeans.svg
```

``waveform_pca.svg`` is a diagnostic scatter of the concat shape matrix ``X``
in PC1–PC2 / PC1–PC3, colored by the frozen NMF hard labels (not re-clustered
in PC space). Rebuilds ``X`` the same way as concat-NMF; does not refit NMF.

``waveform_pca_kmeans.svg`` is the same PC1–PC2 scatter colored by k-means
at NMF *k*=3, Hungarian-matched onto NMF names, with PC1 / PC2 marginal
histograms. Use this as a methods check, not as a replacement for
``channel_assignments.csv``.

``nmf_rank_k.svg`` is the NMF rank-selection justification (cophenetic,
bootstrap ARI, cosine silhouette, reconstruction error vs *k*), read from
``rank_selection_metrics.csv``. Canonical *k*=3 is the cophenetic maximum.
The reconstruction elbow is the kneedle point (maximum deviation from the
chord joining first and last *k*); here it coincides with *k*=3. Cosine
silhouette is slightly higher at *k*=2 and collapses at *k*≥4.

NMF rank justification is ``nmf_rank_k.svg`` (cophenetic / ARI / silhouette /
reconstruction). PC-space 1D/1E curves and the k-means scatter are methods
checks only: tables from ``sbatch scripts/slurm/nmf_pc_clustering.sh``,
figures from ``notebooks/nmf_pc_clustering.ipynb``.

---

## 4. How to re-run

```bash
# Rank selection
sbatch scripts/slurm/nmf_rank_bootstrap.sh

# Fit (default k=3; or pass --k from chosen_k.json)
sbatch scripts/slurm/nmf_concat.sh

# Waveform PCA scatter (frozen assignments; rebuilds X only)
sbatch scripts/slurm/nmf_waveform_pca.sh

# PC scree + PC-space clustering tables (no figures)
sbatch scripts/slurm/nmf_pc_clustering.sh
# Then plot 1D/1E, NMF rank curves, and the k-means scatter in
# notebooks/nmf_pc_clustering.ipynb
```

---

## 5. Downstream contract

Read only ``results/nmf/channel_assignments.csv`` via
`src.paths.nmf_assignments_path()`.
