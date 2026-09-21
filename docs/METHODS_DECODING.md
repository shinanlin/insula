# Time-resolved decoding

Two time-resolved decoding paths live in this repo. They answer the same
question (train on t, test on t) but are tuned for different regimes, and they
write to different output trees so results never mix.

| | production | shrinkage-LDA |
|---|---|---|
| driver | `src/decoding/run_decoding_resolved.py` | `src/decoding/run_decoding_resolved_lda.py` |
| estimator | `PCA(0.90)` + `LinearSVC` | `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` |
| features | every window sample, flattened | 5 time bins per channel, flattened |
| metric | balanced accuracy | AUC, macro one-vs-rest if multiclass |
| aggregation | mean of per-fold scores | pooled out-of-fold |
| CV | `MinimumNaNSplit` | `--cv_scheme`, see below |
| null | trial-level label shuffle | word-level label shuffle |
| output datatype | `(decode)(resolved){datatype}` | `(decode)(resolved)(lda){datatype}` |

Both share the sliding window (0.3 s, 0.03 s step), `cluster_correction`, and
the pooled-ROI HDF5 inputs from `prepare_*_decoding_dataset.py`.

## Why a second path

The production settings work on wide pools such as STGl (85 channels) but go
null on the k=3 insula functional clusters, which carry 14-30 channels each.
Subsampling STGl down to cluster channel counts reproduces the same messy,
sometimes below-chance curves, so the limit is statistical power in the
estimator, not the absence of a signal in the region (SHI-39).

Each of the five changes below targets a specific loss of power. They were
adopted together, so their individual contributions are not separable from the
runs described here.

### Estimator: shrinkage LDA

With a 0.3 s window at 128 Hz, a 16-channel pool yields 608 raw features
against ~130 trials. `PCA(0.90)` picks a component count from a covariance
estimated on those same ~130 trials, then `LinearSVC` fits in that subspace.
Ledoit-Wolf shrinkage LDA instead regularises the covariance analytically, with
no variance-threshold hyperparameter to be unstable across time points.

### Features: five time bins

Binning the 38 window samples to 5 bins (~60 ms) keeps the within-window time
course but cuts the feature count by 7.6x, which is what makes a covariance
estimate on ~130 trials tractable.

### Metric: AUC

Balanced accuracy thresholds the decision function and discards the margin.
AUC uses the full ranking, so a consistent but small separation still registers.

For targets with more than two classes the metric is a macro one-vs-rest
average: the estimator's per-class score columns are standardised column-wise
within each fold, pooled, and one binary AUC is taken per class against the
rest, then averaged. Chance stays at 0.5 whatever the class proportions are, so
articulator curves can be read on the same axis as lexicality ones.

The binary case is a special case of the same code, not a separate branch. A
binary `decision_function` returns one column, which is mirrored into `[-v, v]`;
AUC is invariant to flipping both the label and the sign of the score, so the
two columns score identically and their average is the plain binary AUC.
`tests/test_decoding_lda_resolved.py` pins that against the pre-multiclass
implementation to twelve decimal places.

A permutation whose training split happens to be missing a class is dropped as
degenerate, because the estimator would then return fewer score columns than
there are classes and the pooled matrix would be misaligned.

### Aggregation: pooled out-of-fold

Per-fold balanced accuracy on ~34 test trials is quantised and noisy; averaging
five such numbers keeps that noise. Instead every fold's test decision values
are collected and a single AUC is computed over all trials.

Decision values are standardised within each fold before pooling. That is
monotone within a fold, so it cannot create or destroy within-fold ranking, but
it stops one fold's offset decision function from dominating the pooled order.

### CV: `--cv_scheme`

Pooled datasets align trials across subjects by stimulus identity, each word is
presented twice, and every target here is a deterministic function of the word.
A random stratified split therefore lets the same word appear in train and test.
`--cv_scheme group` uses `StratifiedGroupKFold` on `condition` (the bare word,
two rows per word) to remove that; `--cv_scheme stratified` uses a plain
`StratifiedKFold` and allows it.

Lexicality uses `group`. On STGl this changes little: `diagnose_item_leakage.py`
measured 90.5% of test trials having a twin in training under the production
splitter, yet the corrected time course was essentially unchanged. Grouping is
kept there because it is correct, not because it rescued the result.

Articulator uses `stratified`, for comparability with the production articulator
runs, which are ungrouped. The cost is in what the result licenses: it shows
that articulator-correlated information is present, not that it generalises to
unheard words. The null still permutes at the word level under either scheme
(`StratifiedKFold.split` accepts and ignores the grouping vector), so whatever
item memorisation the folds allow appears in the null too rather than being
credited to the observed score.

### Null: word-level shuffle

Because the label is a function of the word, the exchangeable unit is the word,
not the trial. Trial-level shuffling can split a word's two presentations
across classes, which cannot happen under the real labels, and yields an
over-narrow null. `decode_permutation_auc_pooled` permutes the word-to-label
map and expands it back to trials, so both presentations always agree.

## Figures

`src/decoding/summarize_resolved_lda.py` writes one SVG per description, phases
as columns, ROIs overlaid. It follows the canonical `plot_phase_accuracy` in
`notebooks/decode_functional.ipynb` so the panels sit next to the production
time-resolved figures unchanged: `cm = 1/2.54` sizing at
`(2.5 * n_phases, 3)` cm, 7 pt type, `lw=1` traces smoothed with
`gaussian_filter1d(sigma=2)`, dashed grey chance and onset lines at `lw=0.5`,
and stacked `fill_between` significance strips above the data ceiling.

ROI colours come from the same map: Sensory `#2369BD`, Sustain `#C4A35A`,
Motor `#A9373B`, STG `#20B2AA` (see `FUNCTION_COLORS` in
`docs/PLOTTING_STYLE.md`). Pooled ROIs are relabelled the way
`roi_hemi_from_subject` does it, so `STGl` plots as `STG`.

Two deliberate departures, both because this grid has two phases rather than
four. The legend sits outside the last axis instead of inside panel `n // 2`,
which at two panels would cover the Delay trace. And y ticks are clipped to the
data ceiling before `despine(trim=True)` runs, so no tick lands in the
significance band where it would read as an AUC the traces never reach.

## Implementation notes

`decode_permutation_auc_pooled` in `src/decoding/decoder.py` takes the
classifier and the unsupervised `transformer` separately. The transformer is
fit once per training split and reused across permutations. This is exact, not
an approximation: the transformer never sees the labels, so permuting them
cannot change what it would have learnt. Do not pass supervised preprocessing
as the `transformer`, it would leak.

Imputed fold arrays are likewise built once. `sample_fold`'s NaN handling is
label-independent, and `tests/test_decoding_lda_resolved.py` pins that
invariant so the caching breaks loudly if `sample_fold` ever changes.

Permutations are dispatched to worker processes in batches. A single
permutation is ~150 ms, too short to amortise handing the fold arrays to a
worker, and the thread backend gains almost nothing because the cost sits in
GIL-bound scikit-learn code rather than in long BLAS calls.

## Scope of the first LDA run

Sensory, Sustain, Motor, plus STGl as a positive control; Stimulus and Delay;
Repeat and Decision; `lexicality` only. 16 combinations,
`scripts/slurm/decoding_resolved_lda_lexicality.sh`.

STGl gates the rest. It must reproduce its known lexicality time course before
the cluster results are interpreted. The reference is the `production` arm in
`results/decoding/diagnostics/item_leakage/STGl_lexicality_*.svg`, since STGl
has never been run through the production resolved driver.

The 50-permutation smoke run reproduced it: AUC sits at 0.45-0.52 before
0.35 s, crosses into significance at 0.46 s, and peaks at 0.798 at 0.64 s,
consistent with a ~0.5 s stimulus.

Held back until the clusters report: Go and Response phases and the full ROI
list. Deliberately out of scope: NNLS component decoding (SHI-24), and the
`sample_fold` NaN asymmetry, which shifts scores by 0.001-0.011 and is not being
changed.

## Scope of the articulator run

Same four pools and same two phases, `articulator` only, on both datasets:
LexicalDelay with Repeat and Decision (`decoding_resolved_lda_articulator_ld.sh`,
16 cells) and PhonemeSequence with Repeat only
(`decoding_resolved_lda_articulator_ps.sh`, 8 cells).

Both use `--cv_scheme stratified` with the word-level null, for the reasons
above. `phoneme` is deliberately not included: at 17 classes in LexicalDelay its
rare classes cannot fill five stratified folds, and that needs its own decision
about which classes to drop.

PhonemeSequence additionally passes `--drop_labels other`. `sequence_articulator`
in `prepare_functional_decoding_dataset.py` assigns a catch-all `"other"` to any
phoneme outside the four articulatory buckets, and `prepare_feature` only filters
`None` and the empty string, so `"other"` survives as a class with no linguistic
meaning. Filtering happens at load time in the driver rather than in the
preparation step, which keeps the on-disk inputs and the production results built
from them untouched.

Preparation inputs are not guaranteed to exist for every cell, and the worker
exits 0 when one is missing, which Slurm records as a success. Run
`PREFLIGHT=1 TASK=... DATATYPES=articulator bash scripts/decoding_resolved_lda_worker.sh`
before submitting; it walks the whole matrix and exits non-zero if anything is
missing.

## Choosing n_perm

`scripts/slurm/bench_decoding_resolved_lda.sh` measures the permutation rate per
pool and prints the implied wall time for 57 windows. Run it before changing
the estimator, the window, or the pool list, and size `N_PERM` in the array
script from its output.

Cost scales with the square of the channel count, since that drives the
covariance estimate, so the control pool dominates the wall time even though it
is the least interesting cell. Measured on 16 cores:

| pool | channels | perms/s | 5000 perms x 57 windows |
|---|---:|---:|---:|
| STGl | 85 | 19.3 | 4.1 h |
| Sustain | 30 | 204 | 0.4 h |
| Motor | 18 | 313 | 0.3 h |
| Sensory | 16 | 342 | 0.2 h |
