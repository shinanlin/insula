# NMF Insula Inspector

Minimal QC viewer for NMF functional clusters: insula-only 3D brain, cluster-colored electrodes, click-to-inspect HGA waveforms, manual cluster reassignment with CSV download.

Sibling to [`../hga_explorer/`](../hga_explorer/) — no phase Venn, no full brain, no native/template toggle.

## Quick start

```bash
cd viewer/nmf_inspector

# 1. Export data + symlink insula mesh assets
bash scripts/build_data.sh

# 2. Install + dev server (port 5174)
npm install
npm run dev
```

Open http://localhost:5174/

## Data sources

- Electrodes: `results/nmf/channel_assignments.csv` (canonical concat-NMF, k=3: `sustain` / `motor` / `sensory`)
- Waveforms: packaged HGA under `results/hga/{Task}/` via `BIDSPath` (Repeat condition, `sound` modality only; four phases; LexicalNoDelay excluded)

Export writes to `public/data/`:

- `manifest.json`
- `electrodes.json`
- `traces/{subject}.json`

Insula mesh GLB is symlinked from `viewer/hga_explorer/public/assets/` (or main `insula` checkout).

## HPC serve

```bash
bash scripts/build_data.sh   # or sbatch scripts/build_data.sh
npm install && npm run build # or sbatch scripts/build_viewer.sh
sbatch scripts/serve.sh      # port 18082
```

SSH tunnel:

```bash
ssh -L 18082:<compute-node>:18082 ns458@dcc-login.oit.duke.edu
```

## Loading purity (viewer QC)

Electrodes are soft-assigned by NMF loadings; hard `functional_cluster` is still argmax. The viewer also scores **margin = top − second loading**:

| Tier | Rule | 3D style |
|------|------|----------|
| `pure` | margin ≥ 0.2 | solid cluster color |
| `borderline` | 0.1 ≤ margin &lt; 0.2 | faded solid |
| `mixed` | margin &lt; 0.1 | wireframe (hollow) |

Use the **Purity** chips (with cluster filter) to inspect ambiguous electrodes before downstream hard-label analyses.

## Manual overrides

Use the reassign dropdown per channel, then **Download overrides CSV**:

```text
channel,functional_cluster_original,functional_cluster_manual
```

Does not overwrite `results/nmf/channel_assignments.csv` in v1.
