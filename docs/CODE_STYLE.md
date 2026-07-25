# Code Style Guide

This guide describes conventions for Python source files, SLURM scripts, paths, and tests in the Insula analysis workspace.

## Environment

Use the `ieeg` conda environment for running and testing code:

```bash
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
```

Full analyses should be run through SLURM scripts in `scripts/`. Avoid launching long-running analyses interactively.

## Repository Boundaries

- Put reusable analysis logic in `src/` (see subpackage layout below).
- Put SLURM entry points in `scripts/`.
- Put tests in `tests/`.
- Use `notebooks/` for exploratory work.
- Use `vizpub/` for publication figure notebooks. Grant figures live in `../insula-grant`.
- Avoid adding production logic to `tmp/`. If a temporary script becomes important, promote it into `src/` and add a matching SLURM script if needed.

## Python Entry Points

New command-line source files should follow the established runner pattern:

```python
#!/usr/bin/env python3
"""Short module description."""

import rootutils

path = rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
    cwd=True,
)
```

Use `argparse` and an explicit `main(...)` function. Keep the `if __name__ == "__main__":` block small and limited to parsing arguments and calling `main`.

## Imports

Prefer package-style imports for new code:

```python
from src.decoding.decoder import decode_permutation_scores
from src.hga.package_highgamma import load_parcellation
```

Some older files used flat imports such as `from decoder import ...`; do not
add new flat imports.

Do not rewrite old imports opportunistically unless you are testing the affected script. For new files, prefer `from src...` imports and the `rootutils` setup pattern.

Avoid notebook-style `sys.path.append(...)` in production source files. If a path workaround is needed, document why and keep it local.

## CLI Arguments

Use explicit argument names that match existing analysis vocabulary:

- `--bids_root`
- `--subject`
- `--ref`
- `--description`
- `--phase`
- `--band`
- `--datatype`
- `--n_perm`
- `--n_folds`
- `--n_jobs`

When possible, use `choices=` for fields with stable vocabularies, such as condition, phase, reference, and datatype.

Use production values in SLURM scripts. Python defaults may be lightweight development defaults, so do not assume a default value is a final analysis setting.

## Logging

Use the standard library `logging` module for source files.

Preferred format for new code:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
```

Log enough runtime context to reproduce a job:

- BIDS root.
- Subject or ROI.
- Task, condition, phase, band, datatype, and reference.
- Input shapes.
- Output path.
- Key model parameters, such as window size, step size, folds, permutations, and PCA variance.

## SLURM Scripts

SLURM scripts should live in `scripts/` and call source files in `src/`.

Recommended structure:

```bash
#!/bin/bash

#SBATCH --job-name=<job_name>
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/<job_name>_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/<job_name>_%a.err
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
```

For new scripts, prefer the group path:

```text
/hpc/group/coganlab/nanlinshi/insula
```

Older scripts may still use:

```text
/hpc/home/ns458/coganlab/nanlinshi/insula
```

Do not change working paths in old scripts without testing the script on the cluster.

For array jobs, flatten all combinations into arrays and include a bounds check against `SLURM_ARRAY_TASK_ID`. The newer `scripts/decoding_resolved.sh` file is a good model.

## BIDS and Results Naming

Use CamelCase task names:

- `LexicalDelay`
- `LexicalNoDelay`
- `PictureNaming`
- `PhonemeSequence`
- `SentenceRep`
- `TIMIT`
- `CrossTask`

Use existing condition and phase names:

- Conditions: `Repeat`, `Decision`, `Passive`.
- Phases: `Stimulus`, `Delay`, `Go`, `Response`.

Use existing result path grammar:

```text
results/<Task>(<ref>)
results/<Task>(roi)(<ref>)
results/<Task>(cross_roi)(<ref>)
```

Examples:

```text
results/LexicalDelay(bipolar)
results/LexicalDelay(roi)(bipolar)
results/PhonemeSequence(roi)(bipolar)
```

Do not change this grammar in only one script. Notebooks and downstream analysis code often assume it.

Analysis-specific result roots sit beside the packaged-HGA grammar. Use a single
word with no underscores, for example `results/connectivity`.

### BIDSPath for reads and writes

Read and write all analysis artifacts with `mne_bids.BIDSPath`. Follow the
patterns in `src/xcorr/run_xcorr_pair_permutation.py` and
`src/hga/package_highgamma.py`. Do not hand-concatenate BIDS stems or filenames.

BIDS filenames are `key-value` tokens joined by `_`. Each entity value must be a
single token with no `_` inside the value (for example `LexicalDelay`,
`Response`, `xcorr`). Underscores only separate entities.

Use `BIDSPath.mkdir(exist_ok=True)` before writing. Set `check=False` for
project-local derivative layouts that are not full BIDS datasets.

Top-level folders under `results/` and `logs/` for new pipelines must be single
words (for example `connectivity`, `logs/connectivity`), not `snake_case`.

Example connectivity layout:

```text
results/connectivity/<Task>/sub-<ID>/<metric>/
  sub-<ID>_task-<Task>_proc-<Phase>_desc-<Cond>_<suffix>.<ext>
```

Where `<metric>` is the BIDS `datatype` (`xcorr`, `oaec`, `wpli`) and
`<suffix>` is a single token (`pairs`, `detail`, `clusters`, `provenance`).

## ROI-as-Subject Convention

Several scripts use ROI labels as the `--subject` argument. This is intentional for group-level ROI decoding.

Examples:

```text
AICl
PICl
SMCl
IFGl
STGl
```

The final `l` or `r` indicates hemisphere.

## Randomness and Reproducibility

Use an explicit random seed for stochastic models and cross-validation when practical. The common project seed is:

```python
RANDOM_SEED = 42
```

Record model settings in logs and output metadata when possible. Important settings include:

- Number of permutations.
- Number of folds.
- Time window and step.
- PCA variance.
- Classifier type.

## Statistics

Follow existing project utilities and statistical patterns:

- Use `MinimumNaNSplit` from `ieeg.calc.oversample` for decoding splits when handling missing data.
- Use cluster-based correction helpers for time-resolved decoding and univariate analyses.
- Use `statsmodels.stats.multitest.multipletests` when false discovery rate correction is needed.

Document the statistical unit clearly: trials, channels, ROI-level files, subjects, or task-condition combinations.

## Tests

Run tests from the repository root in the `ieeg` environment:

```bash
conda activate ieeg
pytest tests
```

Add focused tests when changing shared decoding, cross-decoding, connectivity, or statistical helper code. Runner tests can use synthetic data or monkeypatching, following patterns in `tests/`.

Do not require private BIDS data for normal unit tests.

## Legacy Names

Some names are misspelled but stable. Preserve them when existing code depends on them:

- `viz/univarite.ipynb`
- `notebooks/containmination.ipynb`
- `notebooks/containmination_v2.ipynb`

Use correct spelling for new files unless compatibility with an existing path requires the legacy name.

## Avoid

- Adding new hard-coded personal paths when a script can accept `--bids_root`, `--recon_dir`, or a clearly named variable.
- Mixing source logic into SLURM scripts.
- Saving generated analysis outputs outside `results/`, `logs/`, or `img/`.
- Adding production dependencies without documenting them.
- Refactoring old import/path behavior while making unrelated scientific changes.
