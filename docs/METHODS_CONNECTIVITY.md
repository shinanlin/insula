# sEEG strict-Hammers Insula-to-all connectivity

## Scope and interpretation

This pipeline estimates pairwise functional connectivity. It does not estimate
causal or effective connectivity. The primary network contains every eligible
bipolar pair with at least one strict Hammers Insula seed, including
Insula–Insula pairs. It is not an all-to-all network.

The old xcorr analysis selected channels from
`epoch(band)(sig)(effective)`. The primary analyses here do not use HGA
significance to select either seeds or targets:

- xcorr reads every QC-passing channel from `epoch(band)(zscore)`.
- OAEC and TF-dwPLI read every aligned QC-passing channel from `epoch(raw)`.
- `effective` membership is a nullable annotation in the output only.

At least 30 trials must remain after exact event/channel alignment and joint
finite-data QC. Missing raw/Hammers inputs, event mismatch, no strict seed, or
too few valid trials is an explicit manifest/failure reason.

## Strict Hammers seed

The derived `roi` values `AIC` and `PIC` do not define the primary seed.
Instead, each physical endpoint is matched to an explicit normalized Hammers
label. Hemisphere suffixes are removed before exact matching:

| Hammers structure | Abbreviation | Bilateral IDs |
|---|---:|---:|
| insula anterior short gyrus | ASG | 86/87 |
| insula middle short gyrus | MSG | 88/89 |
| insula posterior short gyrus | PSG | 90/91 |
| insula anterior pole (anterior inferior cortex) | AP | 92/93 |
| insula anterior long gyrus | ALG | 94/95 |
| insula posterior long gyrus | PLG | 20/21 |

A bipolar channel is a seed only when both `contact_1_label` and
`contact_2_label` match the set. `center`, midpoint, `roi`, and `mix` do not
affect inclusion. A channel spanning two Insula gyri remains a seed and is
annotated `seed_subregion_mix=True`.

Pairs sharing a physical contact are excluded. The Insula endpoint is always
stored as `source`; for an Insula–Insula pair the lexicographically first
channel is `source`. D0092 has 142 aligned QC bipolar channels, 9 strict seeds,
and 1,224 eligible pairs under these rules.

## Shared windows, trials, and null schedule

Phase windows are half-open:

- Stimulus, Go: `[0, 0.5) s`
- Delay: `[0, 1) s`
- Response: `[-0.5, 0.5) s`

Filtering and Hilbert/Morlet transforms use the complete stored epoch. The
phase time mask is applied to the transformed signal. Raw and z-scored epochs
must have identical `events`, `event_id`, and channel names after exact common
channel selection.

For an entity, a stable SHA-256-derived seed generates target-trial
derangements with no fixed points. The same permutation matrix is shared by
all pairs and all three metrics. Prototype runs use 1,000 permutations; formal
runs use 10,000.

## HGA amplitude xcorr

Input is all-channel HGA amplitude from `epoch(band)(zscore)`. No phase is
reconstructed from this signal. For every trial and lag, signed Pearson
correlation uses only the actually overlapping samples. Correlations are
Fisher transformed and averaged across trials.

Lags span ±0.25 seconds. With the implemented convention, a negative lag means
the Insula/source HGA amplitude pattern occurs earlier. This is a temporal
ordering convention, not a causal direction label. The correlation is not
squared.

The target-trial shuffle recomputes the mean lag curve. A two-sided,
studentized contiguous-lag cluster statistic controls lags within a pair.
Pair-level BH-FDR and a global maximum cluster-mass distribution control the
family of all eligible Insula-to-all pairs.

## HGA orthogonalized AEC

OAEC never reads HGA power, HGA z-score, or an effective-channel envelope. It
starts from bipolar voltage in `epoch(raw)` and builds a 70–200 Hz Gaussian
analytic filterbank using the full epoch.

Each narrowband complex signal is shifted by its filter center to complex
baseband before resampling to 128 Hz. The same unit-magnitude rotation is
applied to both channels, so relative phase and Hipp orthogonalization are
unchanged; this avoids anti-aliasing away a 70–200 Hz carrier.

For every subband and trial pairing:

1. compute the complex, band-specific source and target signals;
2. perform both directions of Hipp pairwise orthogonalization;
3. extract log-power envelopes;
4. correlate the uncorrected envelope of one channel with the other channel's
   orthogonalized envelope;
5. Fisher transform and average directions, trials, and subbands.

Every shuffled pairing is re-orthogonalized against its surrogate partner.
The implementation vectorizes pair blocks and permutation chunks; it does not
shuffle envelopes that were corrected against the original partner. The
signed symmetric statistic uses a two-sided null. Phase is used only to
reduce zero-lag leakage; OAEC measures HGA amplitude-envelope coupling, not HGA
phase synchronization.

## TF debiased squared wPLI

TF-dwPLI also starts only from `epoch(raw)`. Raw voltage is resampled to 256 Hz
and transformed over the full epoch with complex Morlet coefficients at
4–30 Hz. Statistics are evaluated only on band-specific complex coefficients:

- theta: 4–8 Hz
- alpha: 8–13 Hz
- beta: 13–30 Hz
- broadband: 4–30 Hz

For trialwise imaginary cross-spectrum values \(I_k\), the estimator is

\[
\mathrm{dwPLI}^2 =
\frac{(\sum_k I_k)^2-\sum_k I_k^2}
     {(\sum_k |I_k|)^2-\sum_k I_k^2}.
\]

The estimator is not clipped: finite-sample values below zero are retained.
The band statistic uses the trial dimension after averaging the
band-frequency/time cross-spectrum within each trial. Target-trial shuffle
preserves each channel's spectrum while breaking paired phase consistency.
Inference is one-sided (`observed > shuffled null`) over the combined
pair-by-four-band family.

Outputs include the Morlet full 10-sigma support
`5*n_cycles/(pi*frequency)`, source/target band power, valid-bin fraction, and
a short-window flag comparing phase duration with that support. Theta in a
0.5-second phase window also receives an explicit `exploratory_flag`.

## Inference and output schema

Every pair or pair-band row includes:

- uncorrected permutation `p_uncorrected`;
- BH-FDR `q_fdr` and `sig_fdr`;
- studentized global maximum-statistic `p_fwer_maxstat` and `sig_fwer`;
- full BIDS-like entities, source/target anatomy and strict-seed subregions;
- nullable effective-channel annotations;
- trial, channel, seed, and pair QC counts;
- effect statistic and null center/scale where applicable.

Pair tables are Parquet in formal runs. Debug/prototype runs can fall back
explicitly to compressed CSV when no Parquet engine is installed; provenance
records this warning. Detail NetCDF contains xcorr lag curves and clusters,
OAEC directional/subband statistics, and wPLI TF maps/band summaries. Full
null arrays are saved only with `--save-full-null`.

Outputs are written under `results/connectivity/` using `mne_bids.BIDSPath`.
Each dataset has its own task folder, for example
`results/connectivity/LexicalDelay/sub-D0092/xcorr/`. Filenames follow BIDS
entity rules with metric-specific `datatype` folders (`xcorr`, `oaec`, `wpli`)
and single-token suffixes (`pairs`, `detail`, `clusters`, `provenance`). Job
manifests live in `results/connectivity/manifests/`; failure records in
`results/connectivity/failures/`. Slurm logs go to `logs/connectivity/`.

Each metric is written atomically and completed by a BIDS `provenance` JSON,
contains input stat fingerprints, software versions, git state,
implementation/config hashes, deterministic seed, exclusions, runtime, and
peak resident memory. An existing completed result is skipped only on exact
config-hash match; a mismatch requires `--overwrite`.

## Commands

Build the five-dataset manifest:

```bash
python -m src.connectivity.pairwise.cli build-manifest \
  --output /tmp/connectivity_all.tsv
```

Run the D0092 prototype (1,000 permutations by default):

```bash
bash scripts/slurm/run_pairwise_prototype.sh
```

For a short smoke test:

```bash
CONNECTIVITY_PAIR_LIMIT=5 CONNECTIVITY_N_PERM=20 \
  bash scripts/slurm/run_pairwise_prototype.sh
```

After prototype resource review, submit five formal arrays:

```bash
bash scripts/slurm/submit_pairwise_connectivity.sh
```

The submission wrapper uses `common,scavenger`, account `coganlab`, 8 CPUs,
32 GB, 24 hours, four concurrent tasks per dataset (about 20 total), and one
entity per task. Each task loads/aligned inputs once, runs xcorr, OAEC, and
wPLI sequentially, and shares its trial-shuffle schedule.
