# Insula time-resolved reaction-time ridge

This pipeline asks whether a single strict-insula electrode predicts trial RT
in Delay- and Go-aligned high-gamma epochs.

## Locked analysis choices

- Tasks: LexicalDelay, PhonemeSequence, PictureNaming.
- Condition: Repeat.
- Phases: full Delay and Go epochs.
- Electrodes: strict Hammers `AIC` or `PIC`; mixed contacts are excluded.
- Electrode inclusion does not depend on a task-HGA significance mask.
- Target: `log(Response onset - Go onset)`; raw RT is retained in output.
- RT below 50 ms is excluded. Trial order is not modelled.
- Outer CV: shuffled GroupKFold by item, at most 10 folds.
- PictureNaming pools image/sound/text and groups by the four underlying
  concepts.
- Permutation: all training-fold RT values are shuffled without item, block,
  or condition restrictions.
- Inference: one-sided temporal clusters, with the permutation maximum taken
  jointly across Delay/Go, all strict-insula electrodes, and time.

## Output

The production root is:

```text
/hpc/group/coganlab/nanlinshi/insula-functional/results/rt/
```

Each subject has one HDF5 file per phase under
`task-<task>/sub-<subject>/`. Files contain:

- `scores/{r,r2,mae,permutation_r,oof_prediction}`
- `inference/{point_p,cluster_p_fwer,sig_mask_fwer}`
- `windows/{start,end,center}`
- trial RT, item, recording, fold, and event metadata
- strict-insula anatomical labels and native/template/MNI coordinates

Writes are atomic. `run_status.json` records successful and no-insula runs.

Create cohort summaries after all jobs finish:

```bash
python -m src.reaction_time.summarize_insula_rt_ridge
```

This writes `coverage.csv`, `electrodes.csv`, and
`significant_clusters.csv` under `results/rt/summaries/`. Existing hard NMF
cluster assignments are merged by channel when available; they are not used
as model features.

## Test and submit

```bash
pytest -q tests/test_reaction_time_alignment.py tests/test_insula_rt_ridge.py
sbatch scripts/slurm/insula_rt_ridge_smoke.sh
sbatch scripts/slurm/insula_rt_ridge_lexical_delay.sh
sbatch scripts/slurm/insula_rt_ridge_phoneme_sequence.sh
sbatch scripts/slurm/insula_rt_ridge_picture_naming.sh
```

The smoke job defaults to D0096, 20 permutations, and three time windows per
phase. Full arrays default to 1,000 permutations.
