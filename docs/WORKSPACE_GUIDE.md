# Workspace Guide

This document describes how the Insula analysis workspace is organized and highlights conventions that future work should preserve.

## Operating Environment

This is an HPC-centered project. Full analyses should be submitted through SLURM scripts in `scripts/`, not run as long interactive commands.

The expected runtime environment is:

```bash
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
```

Some GPU-capable scripts also load CUDA, commonly:

```bash
module purge
module load CUDA/11.4
```

Use the `ieeg` conda environment for testing and running unless a script explicitly documents another environment.

## Top-Level Layout

- `src/`: all source files. This includes packaging, decoding, univariate statistics, cross-correlation, connectivity, encoding, reaction-time, and utility code.
- `scripts/`: SLURM launchers that run source files from `src/`. This directory is the practical orchestration layer of the project.
- `results/`: generated analysis outputs. Each folder usually corresponds to a task or task-derived analysis.
- `logs/`: SLURM stdout/stderr and per-combination logs.
- `viz/`: older and working visualization notebooks.
- `vizpub/`: publication-oriented visualization notebooks. The user-facing name may be `viz_pub`, but the actual directory in this workspace is `vizpub`.
- `img/`: exported figures and image outputs.
- Grant materials live in a separate worktree: `../insula-grant`.
- Legacy aparc-era analysis (frozen): `../insula-analysis-legacy` — see `LEGACY.md` there.
- `notebooks/`: exploratory or ad hoc analysis notebooks.
- `tests/`: pytest tests.

## `src/` Subpackages

Production Python is grouped by analysis family:

| Subpackage | Role | Key entry |
|------------|------|-----------|
| `src/hga/` | BIDS epoch + parcellation → `results/{task}(bipolar)({atlas})/` | `src/hga/package_highgamma.py` |
| `src/decoding/` | ROI / cross-ROI / cross-condition decoding | `src/decoding/run_decoding.py` |
| `src/xcorr/` | Insula–IFG cross-correlation and viewer export | `src/xcorr/run_xcorr.py` |
| `src/univariate/` | Cluster permutation contrasts on packaged HGA | `src/univariate/contrasts.py` |
| `src/reaction_time/` | Reaction-time encoding models | `src/reaction_time/run_reaction_time.py` |
| `src/connectivity/` | VAR/PDC connectivity | `src/connectivity/run_connectivity.py` |
| `src/encoding/` | mTRF encoding models | `src/encoding/encoder.py` |
| `src/semantic/` | Lexical-semantic RSA / encoding (Lexical Delay) | `src/semantic/run_rsa.py` |
| `src/archive/` | Legacy aparc-era univariate scripts | `src/univariate/contrasts.py` (replacement) |
| `src/paths.py` | Shared `RESULTS_ROOT`, `hga_results_dir()` | — |

Semantic work is developed on branch `semantic` in worktree
`/hpc/group/coganlab/nanlinshi/insula-semantic`. See
`src/semantic/README.md` and `src/semantic/design.md`.

## Main Pipeline Families

### HGA packaging

Packaging scripts prepare BIDS-derived HGA CSVs under `results/`:

- `src/hga/package_highgamma.py` (production; dual-atlas aparc2009s / hammers)
- `src/hga/package_ave_cord.py` (all zscore channels + parcellation coords → `*_coord.csv` for Fig 1 coverage)
- `src/hga/package_zscore.py` (z-scored band packaging)
- `src/hga/package_roi_mask.py` (legacy)
- `src/hga/package_sig_channel.py` (legacy)
- `src/encoding/package_mtrf.py` (mTRF features)

The main SLURM launchers are `scripts/slurm/package_hga_all_tasks.sh` and
`scripts/slurm/package_hga_aparc_two_tasks.sh`.

### Electrode Parcellation and Insula Subregions

General ROI labels are assigned in subject-native FreeSurfer space. Standard
MNI and CVS coordinates are derived for analysis/display and do not replace
native labels. The project rationale, baseline workflow, and insula inclusion
rules are documented in `docs/PARCELLATION.md`. Technical deep dives:

- `docs/PARCELLATION_PIPELINE.md`
- `pipeline/D44_MAPER_worklog.md`
- `pipeline/README.md`

The reusable Faillenot/MAPER workflow is under `pipeline/`. Do not use the
first D44 MAPER products made from uncorrected atlas geometry or sampled via
the fused image scanner affine.

The preprocessing source of truth is the separate repository:

```text
/hpc/group/coganlab/nanlinshi/seeg-preprocessing/
```

Shared stages live under its `common/` and `lib/` directories. Task-specific
code is checked out as worktrees under
`/hpc/group/coganlab/nanlinshi/seeg-preprocessing-worktrees/`. This Insula
repository consumes preprocessing derivatives; it should not grow another
copy of the general parcellation implementation.

### Within-ROI Decoding

Within-ROI decoding uses ROI labels as the `--subject` argument in many scripts. These are group-level ROI codes, not patient IDs.

Important files include:

- `src/decoding/decoder.py`
- `src/decoding/run_decoding.py`
- `src/decoding/run_decoding_resolved.py`
- `src/decoding/run_decoding_generalized.py`
- `src/decoding/condition_decoding.py`
- `scripts/decoding.sh`
- `scripts/decoding_resolved.sh`
- `scripts/generalized_within.sh`
- `scripts/condition_decoding_resolved.sh`

### Cross-ROI and Cross-Condition Decoding

These analyses test generalization across regions, conditions, windows, and tasks.

Important files include:

- `src/decoding/cross_decoder.py`
- `src/decoding/direct_cross_decoder.py`
- `src/decoding/run_cross_roi_resolved.py`
- `src/decoding/run_cross_roi_generalized.py`
- `src/decoding/run_cross_condition_generalized.py`
- `src/decoding/run_cross_condition_window.py`
- `scripts/cross_roi_resolved.sh`
- `scripts/cross_roi_generalized.sh`
- `scripts/cross_condition_generalized.sh`
- `scripts/cross_condition_window.sh`

### Univariate Statistics

Univariate analyses use task and phase-specific contrasts over HGA.

Important files include:

- `src/univariate/contrasts.py`
- `scripts/run_univariate.sh`
- `scripts/run_univariate_nodelay.sh`

### Cross-Correlation and Coupling

Cross-correlation scripts support insula-IFG and related temporal coupling analyses.

Important files include:

- `src/xcorr/run_xcorr.py`
- `src/xcorr/run_xcorr_pair_permutation.py`
- `src/xcorr/run_xcorr_wave.py`
- `src/xcorr/run_xcorr_ortho.py`
- `src/xcorr/extract_roi_xcorr_waveforms.py`
- `src/xcorr/generate_xcorr_viewer.py`
- `src/xcorr/batch_xcorr_viewer.py`
- `src/xcorr/cross_correlation_insula_ifg.py`
- `scripts/run_xcorr.sh`
- `scripts/run_xcorr_pair_permutation.sh`
- `scripts/submit_all_xcorr_perm.sh`

### Connectivity and Encoding

Connectivity analyses include VAR/PDC and permutation/cluster support.

Important files include:

- `src/connectivity/run_connectivity.py`
- `src/connectivity/var.py`
- `src/archive/` — legacy univariate cluster scripts (`run_perm_cluster.py`, etc.); use `src/univariate/contrasts.py` instead
- `scripts/connectivity.sh`

Encoding and mTRF analyses include:

- `src/encoding/encoder.py`
- `src/encoding/package_mtrf.py`
- `scripts/mtrf.sh`

## Task Names

Task names are CamelCase and should match the BIDS task naming used in scripts and result paths.

Common task names:

- `LexicalDelay`
- `LexicalNoDelay`
- `PictureNaming`
- `PhonemeSequence`
- `SentenceRep`
- `TIMIT`
- `CrossTask`

## Conditions, Phases, and Datatypes

Common condition values:

- `Repeat`
- `Decision`
- `Passive`

Common phase values:

- `Stimulus`
- `Delay`
- `Go`
- `Response`

Common decoding datatype values:

- `phoneme`
- `token`
- `articulator`
- `lexicality`
- `word`

Common references:

- `bipolar`
- `car`

## ROI Naming

Many decoding scripts use ROI labels as `--subject`. This is intentional.

Examples:

- `AICl`, `AICr`
- `PICl`, `PICr`
- `SMCl`, `SMCr`
- `IFGl`, `IFGr`
- `STGl`, `STGr`

The final `l` or `r` indicates hemisphere. Do not assume these are participant IDs.

## Results Layout

Generated results are stored under `results/`. Folder names often combine the task, analysis modifier, and reference in parentheses.

Examples:

- `results/LexicalDelay(bipolar)`
- `results/LexicalDelay(roi)(bipolar)`
- `results/LexicalDelay(cross_roi)(bipolar)`
- `results/PhonemeSequence(roi)(bipolar)`
- `results/LexicalDelay(bipolar)(hammers)/sub-*/HGA/*_coord.csv` — all zscore-channel coords (Fig 1 coverage)
- `results/qc/hammers/sub-D0094/` — Stage 3 parcellation slice QC (`index.csv`, `png/`, subject PDF)

Parcellation QC lives under `results/qc/{atlas}/sub-{SUBJECT}/` (subject + atlas
keyed, not task-keyed). Generated by `pipeline/05_visual_qc/plot_parcellation_slices.py`.

This parenthesized grammar is already used by notebooks and scripts, so new work should follow it unless a refactor updates all downstream readers.

## External Data Roots

Input data are not stored in this repository. Scripts commonly refer to BIDS roots under `/cwork/ns458/`, including:

- `/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/`
- `/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/`
- `/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/`
- `/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/`
- `/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/`

MRI reconstruction resources are commonly read from:

- `/cwork/ns458/ECoG_Recon/`

Parcellation and MAPER derivatives are commonly read from:

- `/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/parcellation/`
- `/cwork/ns458/atlases/Hammersmith_n30r95/`
- `/cwork/ns458/maper_run/`

MAPER fusion volumes are subject-level, but extracted bipolar tables are
task-specific. Cross-task merges must include task and reference in their
keys; a subject ID alone is insufficient.

## Path Policy

The current workspace path is:

- `/hpc/group/coganlab/nanlinshi/insula`

Some older scripts still reference:

- `/hpc/home/ns458/coganlab/nanlinshi/insula`

For new scripts and documentation, prefer the group path. If an older script uses the home path and is known to work, preserve it until that script is intentionally modernized and tested.

## Gitignored Outputs

The following directories are generated-output locations and are ignored by git:

- `logs/`
- `results/`
- `img/`

## Known Non-Standard Patterns

- There is no historical root `README.md`; this documentation set is the first project-level guide.
- Several scripts choose the active task by commenting and uncommenting blocks in shell files.
- Some notebooks manually edit `sys.path`; prefer `from src.decoding...` imports.
- `vizpub/` is the actual directory name even if people refer to it as `viz_pub`.
- Several filenames contain legacy typos, such as `viz/univarite.ipynb` and `notebooks/containmination.ipynb`.
- Some legacy notebooks still reference `results/exlude_insula.csv`; that exclusion list is retired. See `docs/PARCELLATION.md`.
- `src/connectivity/var.py` contains VAR/PDC connectivity code; the filename does not mean it is a generic variable/config file.
- `tmp/` contains exploratory scripts that should not be treated as stable production entry points.

## Missing or Fragile References

During the workspace audit, a few referenced files were not present in the current tree:

- `src/package_stats.py`
- `src/atlas.csv`
- `src/run_decoder_patterns_resolved.py`

Do not assume these are unused. They may exist in another branch, be generated, or represent stale references. Check downstream scripts and tests before removing or renaming references.
