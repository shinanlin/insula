# Semantic analyses (Lexical Delay)

Goal: test whether anterior insula (AIC) carries **lexical-semantic**
information that cannot be reduced to acoustics, phonology, or lexical
familiarity alone.

This package is intentionally separate from `src/decoding/` lexicality
decoding. Word vs Nonword is a **lexical-status** baseline, not a semantic
content claim.

## Claim ladder (do not skip rungs)

| Tier | Question | Justified claim |
|------|----------|-----------------|
| A | Can AIC decode Word vs Nonword, especially in Delay / Decision? | Lexical status |
| B | Do neural distances among real words track semantic geometry after phonology/frequency controls? | Lexical-semantic structure |
| C | Do continuous semantic features encode AIC HGA (encoding / mTRF-style)? | Feature-level semantic sensitivity |
| D | Does semantic structure generalize across Decision↔Repeat or tasks? | Goal-invariant semantic content |

**Primary target claim (Tier B):** AIC representational geometry during
maintenance/decision windows correlates with semantic similarity among real
words after residualizing phonology and word frequency.

## Why Lexical Delay

- ~42 real words + matched nonwords, Decision vs Repeat, explicit Delay.
- Existing pipelines already parse `condition → lexicality / token`.
- Nonwords lack stable meaning → semantic analyses use **Word trials only**.
- Picture Naming (few items) is not the primary semantic design; optional later.

## Recommended order

1. Build stimulus feature table (`features.py`).
2. Tier A sanity: reuse existing lexicality decoding (do not reimplement).
3. Tier B RSA on Delay × AIC × Word-only (`rsa.py`, `run_rsa.py`).
4. Residualize phonology + log frequency; split Decision vs Repeat.
5. Tier C encoding if RSA is positive.
6. Tier D only after B/C are stable.

## GloVe embeddings (Word tokens)

Task subset (gitignored) lives under `src/semantic/embedding/`:

- `stimulus_tokens_word.npy` — Word token order
- `embeddings_glove300.npy` — `(n_word, 300)` GloVe vectors
- `embeddings_meta.json` — source / missing-token report

Regenerate (downloads full GloVe into group cache if needed):

```bash
sbatch scripts/slurm/build_glove_embeddings.sh
# or: python src/semantic/build_embeddings.py
```

Full GloVe cache: `/hpc/group/coganlab/nanlinshi/cache/embeddings/glove/`.

## Semantic ridge encoding (v1)

Per-subject GloVe → HGA ridge with token-group CV. Train-fold NaNs are
handled via per-token `mixup` (same pattern as `src/decoding/decoder.py`).

**Significance (channel × time cluster):**

- **Observed:** out-of-fold Pearson `r(ch,t)` from whole-epoch multi-output ridge
  (no sliding window; time axis = epoch samples).
- **Null:** per CV fold, shuffle token→GloVe mapping on the **training set only**
  (same logic as `decode_permutation_scores` in `src/decoding/decoder.py`).
- **Correction:** per channel, `ieeg.calc.stats.time_cluster` on the time axis
  (`src/semantic/stats.py`).
- Defaults: `--n_perm 500`, `--p_thresh 0.05`. Use `--n_perm 0` for `r` only.

```bash
python src/semantic/run_encoding.py --subject D0092 --n_perm 500 --n_jobs 2
sbatch scripts/slurm/run_semantic_ridge_smoke.sh   # D0092, n_perm=50
sbatch scripts/slurm/run_semantic_ridge_all.sh     # 52 subjects, n_perm=500
```

Results: `results/semantic/LexicalDelay/sub-{ID}/..._ridge_glove.h5`

| Dataset | Shape | Notes |
|---------|-------|-------|
| `r` | channels × times | observed encoding |
| `r_null` / `baseline` | ch × t × n_perm | permutation null |
| `mask` | ch × t | cluster-corrected significance |
| `p_values` | ch × t | point-wise permutation p |

Existing H5 files are extended in place when re-run with `--n_perm > 0`.

## Visualization

Interactive exploration notebook (display only, no `img/` export):

```text
notebooks/semantic_encode_viz.ipynb
```

Loads all subject H5 files via `src/semantic/load_encoding_results.py`, merges
Hammers parcellation, and plots:

- insula brain maps on `cvs_avg35_inMNI152` (vizpub fig4 style)
- group ROI time courses AIC / PIC / STG (fig2 style)
- channel × time heatmaps for significant electrodes

Brain plots use parcellation `x_t,y_t,z_t` (CVS template tkRAS), **not** native
`x,y,z`, when pooling electrodes across subjects.

Significance in the notebook uses cluster-corrected `mask` from H5
(`ch_sig_any` = any significant time point per channel). Re-run encoding with
`--n_perm > 0` if older H5 files contain only `r`.

## Data roots

- BIDS: `/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/`
- Decoding derivatives: `derivatives/decoding(bipolar)/`
- Packaged HGA (this repo): `results/LexicalDelay(bipolar)(hammers)/`
- Stimuli wav: `BIDS/stimuli/*.wav`

## Environment

```bash
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
```

Full runs go through SLURM (add `scripts/slurm/` launchers when runners mature).

## Worktree

This work lives on branch `semantic` in worktree:

```text
/hpc/group/coganlab/nanlinshi/insula-semantic
```

Do not mix long semantic experiments into dirty `main` checkouts.
