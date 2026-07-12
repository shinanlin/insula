# Insula HGA Explorer

Interactive multi-task HGA phase-overlap viewer for the Insula project.

| Document | Path |
|----------|------|
| Design spec | [`../../docs/HGA_EXPLORER.md`](../../docs/HGA_EXPLORER.md) |
| Implementation roadmap | [`../ROADMAP.md`](../ROADMAP.md) |
| HPC access | [`docs/ACCESS.md`](docs/ACCESS.md) |

## Quick start (local dev)

```bash
cd viewer/hga_explorer
npm install
npm run dev
```

Open the URL printed by Vite (default port **5173**).

## Build + QA

```bash
npm run build
npm run qa-data
```

## Export data

Validation cohort (3 subjects):

```bash
sbatch scripts/build_data.sh
```

Full cohort (all packaged subjects, union across tasks):

```bash
sbatch scripts/build_data_full.sh
```

Brain meshes:

```bash
sbatch scripts/export_brain_mesh.sh
```

## Serve on HPC

```bash
sbatch scripts/serve.sh
```

See [`docs/ACCESS.md`](docs/ACCESS.md) for SSH tunnel instructions (port **18081**).

## Features (v1)

- Task selector: Phoneme / Lexical / All
- Condition selector: Repeat / Decision (Lexical)
- Phase Venn (stimulus, delay, go, response)
- Template + native brain toggle (single subject, when mesh exported)
- Bipolar endpoint reveal on midpoint click
- KDE + electrode brain views, ROI filter, four-phase waveforms, animation
- Onboarding tour (driver.js)
