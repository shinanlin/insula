# Insula R01 Analysis Workspace

This repository contains analysis scripts, SLURM launchers, notebooks, and figure-generation code for an Insula R01 project investigating the functional architecture of speech within the human insula.

The central question is how speech-related functions are organized within the insula. The current manuscript examines spatial differentiation of sensory, sustained, and motor response profiles, their modulation by task demands, the speech-related information they carry, and their relationship to behavior. Sustained delay-period articulator decoding and motor-group reaction-time prediction provide complementary evidence for this organization. Functional profiles are characterized before interpreting their anatomical distribution; broad anterior/posterior labels do not define the functional groups. Core analyses use human sEEG, high-gamma activity, multivariate decoding, and reaction-time prediction. The broader workspace also supports CCEP, connectivity, and single-unit analyses.

## Paper main-analysis policy

Main site selection, Figure 1 and NMF discovery use **Repeat + sound only**. Picture/text are separate comparisons. Read [MAIN_ANALYSIS.md](docs/MAIN_ANALYSIS.md); shared parameters and colors live in `src/main_analysis.py`. Current sound-only results and corrections are documented in [the correction report](docs/sound_only_correction_2026-10-17.md).

## Quick Start

All real analysis runs are expected to happen on the HPC through SLURM jobs.

```bash
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
sbatch scripts/slurm/<job>.sh
```

Use the `ieeg` conda environment for testing and running code. Most long-running jobs should be submitted through files in `scripts/slurm/`, not launched interactively.

## Directory Map

- `src/`: source files for packaging, decoding, univariate statistics, connectivity, cross-correlation, encoding, behavior, and utility analyses.
- `scripts/`: SLURM scripts and launchers. Production entry points live in `scripts/slurm/`.
- `pipeline/`: MAPER fusion for Hammersmith atlas propagation (Stages 0–1).
- `results/`: current packaged HGA outputs (`{Task}(bipolar)(hammers)/`, etc.).
- Legacy aparc-era outputs live in worktree `../insula-analysis-legacy/results/`.
- `viz/`: previous visualization code, mainly Jupyter notebooks.
- `vizpub/`: publication-oriented visualization notebooks. This is the workspace directory corresponding to the informal name `viz_pub`.
- `img/`: exported figures and image outputs.
- `viewer/`: HGA Explorer and related interactive tools.
- `tests/`: pytest tests for selected analysis modules and runners.
- `docs/`: project documentation — start at `docs/README.md`.

Grant materials live in a separate worktree: `/hpc/group/coganlab/nanlinshi/insula-grant`.

## Main Workflows

The codebase is organized around a few recurring pipeline families:

- **Parcellation and packaging:** MAPER in `pipeline/01–03`, then `src/hga/package_highgamma.py --atlas hammers` via `scripts/slurm/package_hga_*.sh`. See `docs/PARCELLATION.md`.
- **Within-ROI decoding:** `src/run_decoding*.py`, `src/decoder.py`, and related `scripts/decoding*.sh`.
- **Cross-ROI and cross-condition decoding:** `src/run_cross_*.py`, `src/cross_decoder.py`, `src/direct_cross_decoder.py`.
- **Univariate statistics:** `src/univariate/contrasts.py`, launched from `scripts/run_univariate*.sh`.
- **Cross-correlation and insula-IFG coupling:** `src/run_xcorr*.py`, `src/extract_roi_xcorr_waveforms.py`, and viewer-generation scripts.
- **Connectivity:** `src/connectivity/run_connectivity.py`, `src/connectivity/var.py`.
- **Legacy univariate (archived):** `src/archive/` — aparc-era cluster scripts superseded by `src/univariate/contrasts.py`.
- **Encoding and mTRF analyses:** `src/encoding/encoder.py`, `src/encoding/package_mtrf.py`.
- **Figures:** notebooks in `viz/` and `vizpub/`, usually exporting SVG files into `img/`.

## Documentation

Start at [`docs/README.md`](docs/README.md). Key documents:

- **Electrode parcellation (rationale and baseline):** [`docs/PARCELLATION.md`](docs/PARCELLATION.md)
- Project background and task battery: [`docs/PROJECT_BACKGROUND.md`](docs/PROJECT_BACKGROUND.md)
- Workspace structure and conventions: [`docs/WORKSPACE_GUIDE.md`](docs/WORKSPACE_GUIDE.md)
- Code and SLURM style: [`docs/CODE_STYLE.md`](docs/CODE_STYLE.md)
- Plotting and figure style: [`docs/PLOTTING_STYLE.md`](docs/PLOTTING_STYLE.md)
- Parcellation technical reference: [`docs/PARCELLATION_PIPELINE.md`](docs/PARCELLATION_PIPELINE.md)
- D0044 MAPER pilot history: [`pipeline/D44_MAPER_worklog.md`](pipeline/D44_MAPER_worklog.md)
- MAPER operations: [`pipeline/README.md`](pipeline/README.md)
- Canonical preprocessing repository:
  `/hpc/group/coganlab/nanlinshi/seeg-preprocessing/`

## Important Conventions

- Treat SLURM scripts as the authoritative way to run full analyses.
- Paper outputs and code use `/cwork/ns458/insula-paper`; original insula-functional data remain read-only reference inputs.
- Keep generated artifacts in `results/`, `logs/`, and `img/`; these directories are gitignored.
- Preserve legacy file names that downstream code already references, even when they contain typos.
- For new Python entry points, follow the `rootutils.setup_root(..., indicator=".project-root")` pattern used by the newer `src/run_*.py` files.
