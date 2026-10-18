# Insula Paper worktree

The paper worktree is `/cwork/ns458/insula-paper`, branch `codex/paper`. The
three-phase rerun and Figure 2 update are documented in
[the current run record](figure2_refresh_2026-10-17.md).

## Current inputs

- The Figure 1 reviewed cohort: **227 confirmed-responsive sites / 64 participants**.
- The 323-site anatomical sampling list remains the denominator; 94 sites with
  incomplete response evidence remain pending, rather than being counted negative.
- Repeat waveforms: equal available modalities within site/task, then equal available
  tasks within site. Packaged HGA is read from the original workspace; existing
  SentenceRep epochs supplement waveforms as recorded in Figure 1's manifest.
- Fit windows: Stimulus **0–0.6 s**, Delay **0–1 s**, Response **0–1 s**, including
  sampled endpoints. Response-significance screening remains **−0.5–0.5 s**.
- Negative clipping followed by L2 normalization of the entire concatenated waveform.
- No old 26-channel exclusion list, pure AIC/PIC filter, or four-phase assignments.

## Reproduction

Submit these Slurm scripts in order, with `afterok` dependencies:

1. `scripts/slurm/nmf_three_phase_fit.sh`: prepare the reviewed 227×335 matrix,
   fit **k=3** with 20 initializations, select the smallest reconstruction error,
   normalize component H rows and assign each site by largest scaled W loading.
2. `scripts/slurm/nmf_three_phase_rank.sh`: assess k=2–6 with 200 subsamples of
   80% of sites. The cophenetic maximum is reported independently; it does not
   override the requested k=3 fit.
3. `scripts/slurm/execute_fig2_notebook.sh`: execute `vizpub/fig2.ipynb`, validate
   all outputs, and archive the inputs, model, tables and SVGs under
   `docs/evidence/figure2_refresh_2026-10-17/`.

New model outputs are in this worktree's `results/nmf/`; figure data in
`results/fig2/`; SVGs in `img/fig2/` and `img/nmf/`. The original four-phase
workspace is a read-only reference. Historical launchers must not be used to
regenerate paper figures.
