# Insula R01 Analysis Workspace

This repository contains analysis scripts, SLURM launchers, notebooks, and figure-generation code for an Insula R01 project focused on the role of the anterior insular cortex in speech production.

The working hypothesis is that the anterior insula performs domain-general cognitive operations, such as maintenance and goal/action-directed control, over speech-specific neural representations, such as lexical status and articulator identity. The project uses human sEEG, high-gamma activity, multivariate decoding, CCEP, connectivity, and single-unit data to test this cognitive-motor interface account.

## Quick Start

All real analysis runs are expected to happen on the HPC through SLURM jobs.

```bash
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
sbatch scripts/<job>.sh
```

Use the `ieeg` conda environment for testing and running code. Most long-running jobs should be submitted through files in `scripts/`, not launched interactively.

## Directory Map

- `src/`: source files for packaging, decoding, univariate statistics, connectivity, cross-correlation, encoding, behavior, and utility analyses.
- `scripts/`: SLURM scripts that run the source files in `src/`. This is the main orchestration layer for the HPC.
- `results/`: generated outputs. Each task or task-derived analysis gets its own folder, such as `LexicalDelay(roi)(bipolar)`.
- `viz/`: previous visualization code, mainly Jupyter notebooks.
- `vizpub/`: publication-oriented visualization notebooks. This is the workspace directory corresponding to the informal name `viz_pub`.
- `img/`: exported figures and image outputs.
- `grant/`: R01 grant notebooks, figures, and related materials.
- `tests/`: pytest tests for selected analysis modules and runners.
- `tmp/`: one-off exploratory scripts that should not become production entry points without being promoted into `src/` or `scripts/`.

## Main Workflows

The codebase is organized around a few recurring pipeline families:

- BIDS and feature packaging: `src/package_*.py`, launched from `scripts/package.sh`.
- Within-ROI decoding: `src/run_decoding*.py`, `src/decoder.py`, and related `scripts/decoding*.sh`.
- Cross-ROI and cross-condition decoding: `src/run_cross_*.py`, `src/cross_decoder.py`, `src/direct_cross_decoder.py`.
- Univariate statistics: `src/univariate_contrasts.py`, launched from `scripts/run_univariate*.sh`.
- Cross-correlation and insula-IFG coupling: `src/run_xcorr*.py`, `src/extract_roi_xcorr_waveforms.py`, and viewer-generation scripts.
- Connectivity: `src/run_connectivity.py`, `src/var.py`, `src/run_perm_cluster.py`.
- Encoding and mTRF analyses: `src/encoder.py`, `src/package_mtrf.py`.
- Figures: notebooks in `viz/`, `vizpub/`, and `grant/`, usually exporting SVG files into `img/` or `grant/grantfig/`.

## Documentation

- Project background and task battery: `docs/PROJECT_BACKGROUND.md`
- Workspace structure and known non-standard patterns: `docs/WORKSPACE_GUIDE.md`
- Code and SLURM style: `docs/CODE_STYLE.md`
- Plotting and figure style: `docs/PLOTTING_STYLE.md`
- Electrode coordinates, native parcellation, and MAPER status: `docs/PARCELLATION_PIPELINE.md`
- D0044 MAPER pilot history and corrected-bug record: `docs/D44_MAPER_worklog.md`
- Reusable Faillenot/MAPER workflow: `pipeline/README.md`
- Canonical preprocessing repository and task worktrees:
  `/hpc/group/coganlab/nanlinshi/seeg-preprocessing/`

## Important Conventions

- Treat SLURM scripts as the authoritative way to run full analyses.
- Prefer the workspace path `/hpc/group/coganlab/nanlinshi/insula` for new documentation and scripts.
- Keep generated artifacts in `results/`, `logs/`, and `img/`; these are gitignored except for known exceptions such as `results/exlude_insula.csv`.
- Preserve legacy file names that downstream code already references, even when they contain typos.
- For new Python entry points, follow the `rootutils.setup_root(..., indicator=".project-root")` pattern used by the newer `src/run_*.py` files.
