# Insula HGA Explorer — Design Notes

Status: draft for iterative correction.  
Goal: build a Sternberg-isomorphic interactive multi-task HGA web viewer for the Insula project.  
Module path: `insula/viewer/hga_explorer`  
Canonical doc: `docs/HGA_EXPLORER.md` (renamed from `PHASE_OVERLAP_VIEWER.md`)  
Implementation roadmap: [`viewer/ROADMAP.md`](../viewer/ROADMAP.md)

## 1. What this is (and is not)

### Is
- A React + Vite + Three.js local/HPC web app, modeled on
  `/hpc/group/coganlab/nanlinshi/sternberg/viewer/phase_overlap`.
- Core interaction: select trial phases → show electrode overlap on brain →
  inspect ROI counts, electrode list, HGA waveforms, and optional phase animation.
- Multi-task capable: the Sternberg **Load** dimension is replaced by **Task**.
- Multi-condition capable: v1 single-select condition; v2 adds condition Venn.
- v1 adds two Insula-specific brain features beyond Sternberg:
  1. Toggle **native brain** vs **template brain** (multi-subject → template)
  2. On midpoint click, reveal **clickable bipolar endpoints** with ROI/label detail
  3. Toggle **insula mode**: fig2-style ghost full brain +
     highlighted insula parcels + insula-only electrodes (template and native)

### Is not
- Not a Fig2 publication notebook interactive port.
- Not an xcorr / Insula×IFG viewer rewrite.
- Not a one-off static PyVista HTML export.
- **Not MAPER-dependent in v1.** MAPER fields are out of scope until a later
  `src/package_HGA.py` update and a follow-on viewer revision.

## 2. Sternberg → Insula mapping

| Sternberg concept | Insula concept | Notes |
|---|---|---|
| Single task (`Sternberg`) | Multiple tasks | Primary structural difference |
| Load chips (`3/5/7/9/all`) | Task selector | No load dimension in Insula HGA |
| Phases: Encoding, Maintenance, Probe, Response | Phases: Stimulus, Delay, Go, Response | Names differ; Venn logic same |
| `hga_by_load` | `hga_by_task` (proposed) | Mean masked in-window HGA per task |
| `description=load3/5/7/9` | `task=...` (+ optional `description`) | Filtering key changes |
| ROI bar (Destrieux / AIC-PIC style) | **aparc ROI labels as packaged** | No AIC/PIC split in v1; MAPER deferred |
| Average / template brain only | **Native brain + template brain toggle** | Multi-subject → force template |
| Midpoint electrodes only | Midpoint + **clickable bipolar endpoints** | Endpoints show ROI / label detail |
| Condition fixed or ignored | **Single-select condition** in v1 | v2: condition Venn in addition to phase Venn |
| Split JSON + lazy traces/animation | Same architecture | Startup: manifest + electrodes only |

## 3. Proposed UI behavior

Preserve Sternberg layout, with Insula additions:

1. Top bar: task selector (replaces load chips) + **condition selector** + tour / help
2. Left: phase Venn (Stimulus / Delay / Go / Response)
   - v1: phase Venn only
   - v2: add a **condition Venn** (not only phase Venn)
3. Center: 3D brain
   - Electrodes / KDE view modes
   - Hemisphere filter
   - **Brain space toggle: native ↔ template**
   - **Insula mode**: low-opacity full pial + opaque insula parcels;
     filters electrodes to aparc insula labels; auto camera preset toward insula center
     (template always; native when per-subject insula assets are exported)
   - **Selected electrode:** show bipolar contact-1 / contact-2 markers
     (and optional connecting segment); clear selection → hide endpoints
4. Right: ROI filter (aparc labels) + electrode / endpoint detail panel
5. Bottom: per-phase HGA waveforms + Play animation

### Task selector

v1 decision (closest to Load semantics):
- Single-select task, plus optional `all`
- For a specific task: waveforms, sphere size, KDE weights, and animation follow that task
- For **`all`**:
  - **Inclusion:** an electrode enters the selection if it is significant in
    **any** included task (partial-task allowed; do not require presence in all tasks).
  - **Waveforms:** show **aggregate cross-task waveforms** (mean ± SEM over whichever
    tasks have traces for that midpoint channel).
  - **Sphere size / KDE weights:** **mean** of available task HGAs for that electrode
    (same partial-task rule: average over tasks that have a value).

### Condition selector

- v1 includes **multiple conditions** where available (e.g. LexicalDelay `Decision` /
  `Repeat`; PhonemeSequencing `Repeat`).
- v1: **single-select only** — one active condition at a time.
- **Default condition: `Repeat`** whenever it exists for the active task / condition set.
- When task=`all` and available conditions differ across tasks: expose the
  **union** of conditions. Selecting a condition that a given task lacks simply
  omits that task from aggregates for electrodes/traces that do not have it.
- v2: add a **condition Venn** so overlap can be explored across conditions, not only
  across phases.

### Brain space toggle (v1 requirement)

- **Template:** electrodes and mesh in the shared average / template space
  (Sternberg-like `cvs_avg35` path).
- **Native:** subject-native brain mesh + native electrode coordinates.
  Native pial meshes come from **`/cwork/ns458/ECoG_Recon`** subject surfaces.
  **Always export / show both hemispheres** (lh + rh), not implanted-hemisphere only.
- Switching space must keep selection state (task / condition / Venn / clicked
  electrode) when possible; mesh and coordinate fields swap together.
- **Multi-subject rule:** if more than one subject is selected, **automatically
  switch to template brain**. Native mode is for single-subject viewing.

### Insula mode

Matches the static fig2 insula panel logic (`vizpub/fig2.ipynb`):

- Full pial at **opacity 0.05** (anatomical ghost)
- Insula aparc parcels (`aparc.a2009s`) as a separate mesh at **opacity 0.6**
- Electrodes filtered by aparc `label` (insula parcel name patterns only)
- Camera preset: top-down toward bilateral insula centroid
- KDE mode: vertex colors masked to insula vertices on the decimated pial mesh

**Template brain** — always available (multi-subject safe when insula toggle is on).

Export assets (generated by `export/export_insula_brain_mesh.py`):

- `public/assets/cvs_avg35_insula_pial.glb`
- `public/assets/cvs_avg35_pial_insula_mask.json`
- `public/assets/cvs_avg35_insula.meta.json`

**Native brain** — single subject only; requires per-subject insula export.
Validation cohort (v1): **D0094, D0071, D0084**.

Export assets (generated by `export/export_native_insula_brain_mesh.py`):

- `public/assets/native/{subject}_insula_pial.glb`
- `public/assets/native/{subject}_pial_insula_mask.json`
- `public/assets/native/{subject}_insula.meta.json`

Mask vertex count must match `{subject}_pial.meta.json` `n_vertices` (native
decimation uses `target_faces=80_000`). Expand to full cohort by re-running the
export script with additional subject ids.

### Bipolar endpoint reveal (v1 requirement)

- Default: only bipolar **midpoint** electrodes are shown (as today in Sternberg).
- After clicking a specific electrode: show that channel’s two physical contacts
  (bipolar ends). Do not show endpoints for unselected midpoints.
- Clearing the midpoint selection hides endpoints again.
- Endpoints are **clickable**, not display-only:
  - Clicking an endpoint selects it and shows related info (ROI / aparc label,
    contact name, hemi, coords in the active brain space, parent bipolar channel).
  - Detail panel must support both midpoint and endpoint selections.
- **Waveforms always use the parent bipolar midpoint channel**, even when an
  endpoint is selected. Endpoints have no separate signal view.

## 4. Phase model

Canonical viewer phases (lowercase ids):

- `stimulus`
- `delay`
- `go`
- `response`

Default Venn selection: `stimulus`, `delay`, `go`, `response`.

Note: Sternberg Venn UI currently targets 2–3 phases. Insula v1 wants all four
available/selected by default; if the existing Venn component caps at 3, adapt
or replace that control rather than dropping a phase.

Two different time concepts (do not confuse them):

1. **Display waveform range** — x-axis of the plotted HGA trace.  
   Unified across phases: **[−0.5, 1.0] s**. Already decided.
2. **Significance window** — the peri-onset interval used to decide whether an
   electrode counts as significant for a given phase (Venn membership / overlap
   flags). This follows the shared preprocessing pipeline, not ad-hoc viewer cuts.

### Significance windows (canonical)

Source of truth:
[`seeg-preprocessing/PIPELINE.md`](../../seeg-preprocessing/PIPELINE.md)
→ band-stats stage → `lib/epoch_config.TIME_REGION_BY_PHASE`.

Canonical significance windows (**all tasks**, all phases that exist):

| Phase | Significance window (s) | Display waveform range (s) |
|---|---|---|
| Stimulus | **0.0 – 0.5** | −0.5 – 1.0 |
| Delay | **0.0 – 0.5** | −0.5 – 1.0 |
| Go | **0.0 – 0.5** | −0.5 – 1.0 |
| Response | **−0.5 – 0.5** | −0.5 – 1.0 |

Export / Venn membership must stay consistent with that pipeline. Prefer using
packaged HGA `mask` values that were produced under these windows; if export
re-derives significance, it must use the same `TIME_REGION_BY_PHASE` bounds.

Related pipeline epoch defaults (for context, not viewer display):
- Task-epoch extraction: `tmin=-1.0`, `tmax=1.5` s
- Baseline HG power saved on `(-0.5, 0)` s relative to cue/Start

## 5. Data sources

v1 input root:

- `results(nw)/` packaged HGA

v1 is **MAPER-agnostic**: use aparc / existing packaged columns only. Do not
require `maper_*` fields for export or UI.

v1 task set:

- `PhonemeSequencing(bipolar)`
- `LexicalDelay(bipolar)`

Confirmed phase / condition shape:

- PhonemeSequencing — Stimulus / Delay / Go / Response; desc Repeat
- LexicalDelay — same phases; desc Decision / Repeat

Subjects present in **both** tasks under `results(nw)/` (n=26):  
D0023, D0024, D0028, D0029, D0035, D0042, D0053, D0054, D0055, D0057, D0059,
D0063, D0066, D0068, D0069, D0070, D0071, D0077, D0079, D0084, D0086, D0094,
D0096, D0100, D0102, D0103.

### Validation cohort (v1 first export)

Pick **3 subjects that completed both tasks** (not 3 per task):

- **D0094**
- **D0071**
- **D0084**

Later expansion (not v1):

- Other tasks under `results/` or future `results(nw)/` packages

### Condition / description

- Export and UI must retain available conditions (not Repeat-only).
- v1 UI: single-select condition control; **default `Repeat`**.
- task=`all`: condition choices = **union** across tasks.
- v2 UI: condition Venn in addition to phase Venn.

## 6. Proposed data layout (split, Sternberg-like)

```text
viewer/hga_explorer/public/data/
  manifest.json
  electrodes.json
  traces/{subject}.json          # lazy
  animation/{subject}/{phase}.json
  kde/roi/{subject}/mean.json
  assets/
    template brain mesh (e.g. cvs_avg35_pial.glb)
    native brain meshes from /cwork/ns458/ECoG_Recon (per subject, both hemispheres)
```

### Electrode record (v1-critical schema)

This section is the main Insula delta. Export must support template/native
rendering and on-click bipolar ends.

#### v1 JSON contract (frozen)

Electrode records in `electrodes.json` (split layout) or the `electrodes` array
(monolith / mock) use these field names. Align with Sternberg where possible.

**Identity / labels**

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | `{subject}\|{channel}` (Sternberg convention) |
| `subject` | string | e.g. `D0094` |
| `channel` | string | bipolar midpoint name |
| `roi` | string | packaged aparc ROI (no AIC/PIC remapping) |
| `label` | string | aparc center label |
| `hemi` | string | `L` or `R` |

**Phase / task metrics**

| Field | Type | Notes |
|-------|------|-------|
| `active_phases` | string[] | phases with `mask=true` under pipeline significance windows |
| `phase_flags` | object | `{ stimulus, delay, go, response }` booleans |
| `hga_by_task` | object | `{ PhonemeSequencing: float\|null, LexicalDelay: float\|null }` |
| `region_ids` | string[] | Venn region membership (export-derived) |

Condition-aware traces keyed by task + `description` (Repeat / Decision); trace
bundle shape defined in Phase 1b export.

**Coordinates — midpoint**

| Field | Type | Notes |
|-------|------|-------|
| `x`, `y`, `z` | number | template / projected display coords |
| `x_native`, `y_native`, `z_native` | number | native midpoint coords |

**Coordinates — bipolar endpoints** (revealed on midpoint click; Phase 4 UI)

| Field | Type | Notes |
|-------|------|-------|
| `x1_native` … `z2_native` | number | contact 1/2 native coords |
| `x1_template` … `z2_template` | number | contact 1/2 template coords |
| `contact_1`, `contact_2` | string | contact names |
| `contact_1_label`, `contact_2_label` | string | aparc labels per contact |

Optional detail-panel fields (Phase 4): `contact_1_roi`, `contact_1_hemi`,
`contact_2_roi`, `contact_2_hemi` when export can derive them.

**Excluded:** no `maper_*` fields in v1.

**Minimal electrode example**

```json
{
  "id": "D0094|D0094_LPAS2-3",
  "subject": "D0094",
  "channel": "D0094_LPAS2-3",
  "roi": "SFG",
  "label": "ctx_lh_G_front_sup",
  "hemi": "L",
  "active_phases": ["delay", "go"],
  "phase_flags": { "stimulus": false, "delay": true, "go": true, "response": false },
  "hga_by_task": { "PhonemeSequencing": 0.42, "LexicalDelay": 0.38 },
  "x": -5.83, "y": 2.14, "z": 38.38,
  "x_native": -2.25, "y_native": -19.13, "z_native": 70.75,
  "x1_native": -0.5, "y1_native": -19.25, "z1_native": 70.5,
  "x2_native": -4.0, "y2_native": -19.0, "z2_native": 71.0,
  "x1_template": -3.81, "y1_template": 2.0, "z1_template": 38.10,
  "x2_template": -7.73, "y2_template": 2.16, "z2_template": 38.72,
  "contact_1": "LPAS2", "contact_2": "LPAS3",
  "contact_1_label": "ctx_lh_G_front_sup",
  "contact_2_label": "Left-Cerebral-White-Matter"
}
```

**Minimal manifest example** (split layout)

```json
{
  "version": 1,
  "layout": "split",
  "metadata": {
    "source": "results(nw)",
    "tasks": ["PhonemeSequencing", "LexicalDelay"],
    "conditions": {
      "PhonemeSequencing": ["Repeat"],
      "LexicalDelay": ["Repeat", "Decision"]
    },
    "default_condition": "Repeat",
    "phases": ["stimulus", "delay", "go", "response"],
    "significance_windows": {
      "stimulus": [0.0, 0.5],
      "delay": [0.0, 0.5],
      "go": [0.0, 0.5],
      "response": [-0.5, 0.5]
    },
    "display_waveform_range": [-0.5, 1.0],
    "subjects": ["D0094", "D0071", "D0084"],
    "hga_size_scale": 1.2
  },
  "paths": {
    "electrodes": "electrodes.json",
    "traces": "traces/{subject}.json",
    "animation": "animation/{subject}/{phase}.json",
    "kde": "kde/roi/{subject}/mean.json",
    "template_mesh": "../assets/cvs_avg35_pial.glb",
    "native_mesh": "../assets/native/{subject}_pial.glb"
  }
}
```

**Monolith / mock top level** (`hga_explorer_mock.json`):

```json
{
  "metadata": { "...": "same keys as manifest.metadata" },
  "electrodes": [ "..." ],
  "regions": [ "..." ],
  "traces": {}
}
```

Notes:

- Packaged HGA in `results(nw)/` includes midpoint and endpoint columns from
  `src/package_HGA.py` (aparc parcellation + bipolar endpoint fields).
- No `maper_*` fields in v1 electrode JSON.

### Manifest (proposed)

- task list (`PhonemeSequencing`, `LexicalDelay`)
- condition list per task + union list for task=`all`; default `Repeat`
- phase list + pipeline significance windows + unified waveform display range `[-0.5, 1.0]`
- subject list (validation first: D0094, D0071, D0084)
- paths for traces / animation / kde
- template mesh path
- native mesh path pattern from `/cwork/ns458/ECoG_Recon`
- `hga_size_scale` (cohort p95 over selected metric)
- note that `all` tasks uses aggregate cross-task waveforms and partial-task inclusion

## 7. Export pipeline (proposed module layout)

Mirror Sternberg self-contained module under Insula:

```text
viewer/hga_explorer/
  export/
    compute_hga_explorer.py
    hga_explorer_geometry.py
    hga_explorer_animation.py
    hga_explorer_kde.py
    export_average_brain_mesh.py
    export_native_brain_mesh.py   # new vs Sternberg
  scripts/
    build_data.sh
    qa_export.py
    serve.sh
    export_brain_mesh.sh
    connect_tunnel.sh
  src/          # React app
  docs/ACCESS.md
  README.md
```

Module directory name: **`insula/viewer/hga_explorer`** (not `phase_overlap`).
Sternberg remains the architectural reference; Insula naming should reflect
multi-task HGA exploration rather than Sternberg’s phase-overlap product name.

Key export changes vs Sternberg:

1. Discover HGA across `results(nw)/{task}(bipolar)` for PhonemeSequencing + LexicalDelay
2. Replace load aggregation with task aggregation
3. Remap phase names Stimulus / Delay / Go / Response
4. Drop encoding-load cutoffs; use pipeline significance windows from
   `seeg-preprocessing` (`TIME_REGION_BY_PHASE`)
5. Unified waveform export/display window: **−0.5 to 1.0 s**
6. Emit dual coordinate frames (native + template) for midpoints
7. Emit bipolar endpoint coordinates + contact aparc labels for clickable reveal
   (after `package_HGA.py` gains these fields)
8. Export / reference native subject pial meshes from `/cwork/ns458/ECoG_Recon`
   (always both hemispheres)
9. Preserve multiple conditions in export; v1 UI single-selects one (default Repeat);
   task=`all` uses condition union
10. Do **not** require or export MAPER metadata in v1
11. Keep aparc `roi` / `label` as packaged; no AIC/PIC split

## 8. Reuse vs rewrite

High reuse from Sternberg viewer:

- Brain viewer components (Electrodes / KDE / animation playback) as a base
- Venn selection pipeline (adapt for four-phase default)
- Lazy data loading + split layout
- HPC serve + SSH tunnel docs pattern
- Onboarding tour structure

Must rewrite / adapt:

- `loads.js` → `tasks.js`
- export `compute_hga_by_load` → `compute_hga_by_task`
- phase constants, pipeline significance windows, waveform x-range `[-0.5, 1.0]`
- brain space toggle (native ↔ template); multi-subject forces template
- selected-electrode bipolar endpoint overlay; endpoints clickable with detail
- single-select condition control (default Repeat; v2: condition Venn)
- README / ACCESS copy
- any Sternberg-specific encoding cutoff logic

## 9. Non-goals for v1

- MAPER six-region styling, filters, or required `maper_*` columns
- AIC/PIC remapping of aparc ROIs
- Fig2 publication notebook parity
- Xcorr coupling visualization
- Condition Venn (deferred to v2; v1 is single-select condition only)
- Full-cohort polish before the 3-subject validation export works

## 10. Suggested v1 milestone

1. Scaffold `insula/viewer/hga_explorer` from Sternberg structure
2. Update `src/package_HGA.py` to emit bipolar endpoint coords + contact aparc fields
3. Export validation subjects **D0094, D0071, D0084** from `results(nw)/`
   (both PhonemeSequencing + LexicalDelay)
4. Task selector wired through size / waveform / animation;
   `all` = partial-task inclusion + aggregate cross-task waveforms;
   size/KDE = mean of available task HGAs
5. Single-select condition control (default `Repeat`; task=`all` uses condition union)
6. Phase Venn on Stimulus / Delay / Go / Response (default all four)
7. Native ↔ template brain toggle; selecting multiple subjects auto-switches to template;
   native meshes from `/cwork/ns458/ECoG_Recon` (**both hemispheres**)
8. Click midpoint → show bipolar ends; click endpoint → ROI/label detail;
   waveforms always remain on parent midpoint channel
9. Waveforms plotted on unified `[−0.5, 1.0]` s axis
10. Significance / Venn membership aligned to pipeline windows
    (Stimulus/Delay/Go `0–0.5`; Response `−0.5–0.5`)
11. QA script checks null HGA, missing phases, native/template coords, endpoint + contact labels
12. Local `npm run dev`, then Slurm `serve.sh` + tunnel

## 11. Open questions for correction

Resolved:

1. First task set: **PhonemeSequencing + LexicalDelay**
2. Input root: **`results(nw)/` only** for v1
3. Default Venn phases: **stimulus, delay, go, response**
4. Display waveform range: **unified −0.5 to 1.0 s**
5. MAPER: **out of scope for v1**; revisit after future packaging work
6. Task `all`: **partial-task inclusion** (significant in any task) + **aggregate cross-task waveforms**
7. Conditions: **include all available**; v1 **single-select**; default **`Repeat`**;
   task=`all` uses **condition union**; v2 **condition Venn**
8. ROI bar: **packaged aparc labels**; no AIC/PIC split in v1
9. Multi-subject selection: **auto-switch to template brain**
10. Bipolar endpoint source: forthcoming **`src/package_HGA.py`** changes
11. Template endpoint field names: **`x1_template` / `y1_template` / `z1_template`** (and contact 2)
12. Endpoints: **clickable**, with ROI / label / related info in the detail panel
13. Viewer module path / name: **`insula/viewer/hga_explorer`**
14. Waveforms: **always midpoint bipolar channel**, even when an endpoint is selected
15. Significance windows: follow **`seeg-preprocessing/PIPELINE.md`** /
    `TIME_REGION_BY_PHASE` — Stimulus/Delay/Go `(0, 0.5)`; Response `(-0.5, 0.5)`
16. Native mesh source: **`/cwork/ns458/ECoG_Recon`** subject pial
17. Doc filename: **`docs/HGA_EXPLORER.md`**
18. Validation subjects (both tasks): **D0094, D0071, D0084**
19. task=`all` sphere size / KDE weights: **mean** of available task HGAs
20. Native mesh export: **always both hemispheres**

Still open / needs user input:

1. Anything else to lock before scaffolding `viewer/hga_explorer`?

## 12. Correction log

- 2026-07-09: initial draft from background investigation; Fig2/xcorr explicitly out of scope; Load → Task is the main isomorphism change.
- 2026-07-09: corrections — default Venn = stimulus/delay/go/response; waveform display unified to [−0.5, 1.0]; v1 uses `results(nw)/` with PhonemeSequencing + LexicalDelay; v1 is MAPER-agnostic; electrode schema elevated for native↔template brain toggle and on-click bipolar endpoint reveal; MAPER deferred until later `package_HGA.py` work.
- 2026-07-09: corrections — `all` = aggregate cross-task waveforms; conditions included with v1 single-select and v2 condition Venn; aparc ROI labels only; multi-subject forces template; endpoint coords from upcoming `package_HGA.py`; `*_template` endpoint naming; endpoints clickable with ROI/label detail.
- 2026-07-09: clarifications — “strict window” renamed/explained as significance window vs display range; module renamed to `viewer/hga_explorer`; `all` allows partial-task electrodes; waveforms always midpoint even when endpoint selected; added follow-up open questions.
- 2026-07-09: corrections — significance windows locked to seeg-preprocessing PIPELINE (`0–0.5` / Response `-0.5–0.5`); default condition Repeat; task=`all` condition union; native mesh from ECoG_Recon; doc renamed to `HGA_EXPLORER.md`; validation subjects D0094, D0071, D0084 (both tasks).
- 2026-07-09: corrections — task=`all` size/KDE uses mean of available task HGAs; native meshes always both hemispheres.
