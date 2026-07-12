# Archived analysis scripts

Legacy univariate / cluster-permutation entry points from the aparc-era pipeline.
Superseded by `src/univariate/contrasts.py` (packaged HGA, Hammers atlas, SLURM via
`scripts/run_univariate*.sh`).

| Script | Purpose | Why archived |
|--------|---------|--------------|
| `conditional_difference.py` | Per-channel Decision vs Repeat cluster perm on z-scored epochs (all phases), saves under `results/{task}(bipolar)/`. | No launcher; overlaps `univariate/contrasts.py`; notebook-era imports. |
| `cross_task_contrast.py` | LexicalDelay vs LexicalNoDelay response-phase contrast per subject; hardcoded channel exclusions. | Specialized cross-task probe; aparc QC list; launcher in `scripts/archive/`. |
| `run_perm_cluster.py` | ROI-averaged (AIC/PIC/STG/…) Decision vs Repeat cluster perm via old aparc insula split. | No launcher; ROI logic predates Hammers; similar stats in `univariate/contrasts.py` and `hga/package_roi_mask.py`. |

To rerun a legacy script:

```bash
python src/archive/cross_task_contrast.py --subject D0024
```
