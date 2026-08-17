# Insula Analysis Documentation

Entry point for project documentation. Start here, then drill into reference
docs as needed.

## Start here

| Document | When to read |
|----------|----------------|
| [`PARCELLATION.md`](PARCELLATION.md) | **Electrode labeling** — why Hammersmith, MAPER, insula inclusion rules, baseline workflow |
| [`PROJECT_BACKGROUND.md`](PROJECT_BACKGROUND.md) | Scientific framing, task battery, aims |
| [`WORKSPACE_GUIDE.md`](WORKSPACE_GUIDE.md) | Directory layout, naming conventions, pipeline families |

## Analysis and tools

| Document | When to read |
|----------|----------------|
| [`NMF.md`](NMF.md) | **Canonical NMF** — concat postonset, bootstrap rank selection, flat publish paths |
| [`HGA_EXPLORER.md`](HGA_EXPLORER.md) | Interactive HGA Explorer — export schema, UI behavior |
| [`../viewer/ROADMAP.md`](../viewer/ROADMAP.md) | HGA Explorer implementation roadmap |
| [`../pipeline/README.md`](../pipeline/README.md) | MAPER fusion (Hammers volume propagation) |
| [`../pipeline/D44_MAPER_worklog.md`](../pipeline/D44_MAPER_worklog.md) | D0044 MAPER pilot, bugs, validation history |

## Parcellation deep dives

| Document | When to read |
|----------|----------------|
| [`PARCELLATION_PIPELINE.md`](PARCELLATION_PIPELINE.md) | Coordinate spaces, bipolar ROI rules, technical reference |

## Style and conventions

| Document | When to read |
|----------|----------------|
| [`CODE_STYLE.md`](CODE_STYLE.md) | Python, SLURM, paths, tests |
| [`PLOTTING_STYLE.md`](PLOTTING_STYLE.md) | Figure notebooks, SVG export |

## External

- Preprocessing backbone:
  `/hpc/group/coganlab/nanlinshi/seeg-preprocessing/`
- Grant materials (separate worktree):
  `/hpc/group/coganlab/nanlinshi/insula-grant`
- Legacy analysis snapshot (frozen):
  `/hpc/group/coganlab/nanlinshi/insula-analysis-legacy`
