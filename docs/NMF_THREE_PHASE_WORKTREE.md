# Insula Paper worktree

This worktree is at `/cwork/ns458/insula-paper`, on branch `codex/paper`, based on the frozen
four-phase checkpoint `ccee0aa`. Its purpose is to fit K=3 to Stimulus,
Delay, and Response, omitting Go from the concatenated input.

- Read-only input: packaged HGA and the frozen channel exclusion list under
  `/hpc/group/coganlab/nanlinshi/insula-functional/results/`.
- New outputs: this worktree's `results/nmf/` and `img/nmf/` on `/cwork`.
- Rank selection: `scripts/slurm/nmf_three_phase_rank.sh` compares K=2–6 with
  200 electrode subsamples per K, using the same settings as the four-phase run.
- Final fit: `scripts/slurm/nmf_three_phase_fit.sh` fits K=1–3 with 20
  initializations and exports the K=3 assignments, templates, and overview plots.

Both launchers specify `--phases stimulus delay response` and reject a
symlinked local `results/` directory. They have not been submitted. The
existing four-phase launchers retain their historical paths and must not be
used for this run.

Before downstream analyses, record the selected-rank metrics, the new NMF
manifest, the channel-label comparison to the four-phase baseline, and every
figure's input version. Scripts that embed the old worktree path require
review before they are run from this branch.
