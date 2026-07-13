# Electrode Parcellation — Rationale and Baseline Workflow

This document is the **entry point** for anatomical labeling in the Insula R01
analysis workspace. It records the scientific motivation, design consensus, and
operational steps. Technical deep dives live in the reference documents linked
at the end.

**Scope:** insula repository only. Electrode CSV generation is implemented in
`seeg-preprocessing/common/parcellation.py` (Hammers support pending merge of the
`hammers-parcellation` branch). MAPER fusion infrastructure lives under
`pipeline/` in this repository.

---

## 1. Why parcellation matters for this project

This project asks what the **insula contributes to speech**. sEEG conclusions
depend on whether each electrode is truly in insula versus a neighboring
region that also participates in speech.

Several adjacent areas—**IFG**, **STG**, and opercular cortex—have clear roles
in speech production and perception. To isolate insula-specific effects, we must
**separate insula from these neighbors**, not lump them together.

Anatomical misclassification is not a minor QC issue here; it can change which
electrodes enter insula analyses and therefore the scientific story.

---

## 2. Why Destrieux aparc is insufficient for insula

We initially used FreeSurfer **aparc.a2009s (Destrieux)** labels. For whole-brain
ROI work this is fine. For **insula-specific** speech analyses it has a systematic
problem:

- Destrieux groups **opercular IFG** and related gyri with insula-like labels
  (e.g. `G_insular_short`, `S_circular_insula`, insular ring patterns).
- On visual inspection, many electrodes along the **insular gyri rings** sit
  extremely close to **operculum**. Under stricter anatomical criteria, those
  rings belong to **operculum**, not insula.
- Because IFG is strongly engaged in speech, mislabeling opercular contacts as
  insula contaminates the very contrast we care about.

We therefore treat aparc as a **whole-brain reference atlas**, not as the primary
insula definition for this project.

---

## 3. Why Hammersmith + MAPER

### MAPER (propagation)

**MAPER** propagates the Hammersmith n30r95 atlas database into each subject's
native FreeSurfer space via multi-atlas registration and label fusion. In volume
space it generally delineates insula **more plausibly** than Destrieux alone.

MAPER is the **infrastructure** that produces subject-native fused segmentations
(`f30-seg95-*.nii.gz` under `/cwork/ns458/maper_run/`). It is implemented in
`pipeline/01–03` in this repository.

### Hammersmith (label system)

After careful review, we adopt **Hammersmith gross labels** for insula analyses
because they provide explicit **AIC** (anterior inferior cortex) and **PIC**
(posterior insula) regions with boundaries that better respect the
insula–operculum distinction relevant to our data.

Hammersmith is the **label vocabulary** we use in production CSVs and downstream
analyses—not a replacement for native coordinate geometry (still derived from
each subject's `orig.mgz` header).

### Relationship in one line

```text
MAPER  →  native fused volume  →  parcellation.py  →  plot_parcellation_slices.py  →  package_highgamma.py
         (insula/pipeline)       (seeg-preprocessing)  (insula Stage 3 QC)           (insula Stage 4)
```

---

## 4. Design principles (current consensus)

| Principle | Detail |
|-----------|--------|
| Dual atlas, separate columns | **Hammers** for insula-primary analyses; **aparc2009s** for whole-brain ROI / comparison. Never merge atlas columns in one table. |
| MAPER ≠ final labels | MAPER produces volumes; electrode labels come from `parcellation.py` reading those volumes with subject `orig.mgz` geometry. |
| Insula inclusion (Hammers) | `roi ∈ {AIC, PIC}` and `mix == False` on bipolar midpoints. |
| Insula inclusion (aparc, comparison) | `roi` or `label` matching insula patterns (`INS`, `Insula`, etc.) when running aparc-only panels. |
| Retired QC | `filter_insula_electrodes.py`, `purify_insula.sh`, and `exlude_insula.csv` were aparc-era remedies; **do not use** for new work. |

The `mix` flag marks bipolar pairs whose two contacts map to **different tissue
ROIs** (see `docs/PARCELLATION_PIPELINE.md` §4). For insula purity we exclude
mixed pairs rather than maintaining a hand-curated exclusion CSV.

---

## 5. Baseline workflow

### Stage 0–1 — MAPER fusion (insula `pipeline/`, per subject or batch)

One-time cluster setup:

```bash
# See pipeline/README.md and pipeline/ENVIRONMENT.md
pipeline/01_build_container/
pipeline/02_prepare_atlas/
```

Per subject (or use `pipeline/03_run_maper/batch_submit_maper.sh`):

```bash
SUBJECT=D0045
RUN=/cwork/ns458/maper_run/$SUBJECT
# prepare target, generate launchlist, then:
sbatch --export=SUBJECT=$SUBJECT,RUN=$RUN pipeline/03_run_maper/run_maper.sbatch
```

**Output:** `/cwork/ns458/maper_run/<SUBJECT>/output/f30-seg95-<SUBJECT>.nii.gz`

### Stage 2 — Hammers electrode CSV (seeg-preprocessing)

After MAPER fusion exists for a subject, run parcellation (single subject or
batch via insula launcher):

```bash
# seeg-preprocessing repo (hammers branch merge pending)
python common/parcellation.py \
  --bids_root /cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS \
  --recon_dir /cwork/ns458/ECoG_Recon \
  --subject D0094 \
  --atlas hammers \
  --ref bipolar
```

Batch launcher in this repo:

```bash
sbatch scripts/slurm/run_hammers_parcellation.sbatch
```

**Output:**
`BIDS/derivatives/parcellation/sub-<SUBJECT>/<ref>/sub-<SUBJECT>_hammers.csv`

### Stage 3 — Visual QC (insula)

After parcellation CSV exists, render native MRI slice PNGs for all pure insula
electrodes (`mix=False`):

```bash
python pipeline/05_visual_qc/plot_parcellation_slices.py \
  --parcellation-csv /cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/parcellation/sub-D0094/bipolar/sub-D0094_hammers.csv \
  --atlas hammers \
  --recon-dir /cwork/ns458/ECoG_Recon \
  --fused /cwork/ns458/maper_run/D0094/output/f30-seg95-D0094.nii.gz
```

This runs automatically after Hammers parcellation when using
`scripts/slurm/run_hammers_parcellation.sbatch`.

**Output:** `results/qc/{atlas}/sub-<SUBJECT>/` (`index.csv`, `png/*.png`, and one subject PDF)

See `pipeline/05_visual_qc/README.md` for filter rules.

### Stage 4 — Package HGA (insula)

```bash
python src/hga/package_highgamma.py \
  --bids_root /cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/ \
  --band highgamma \
  --ref bipolar \
  --atlas hammers
```

SLURM: `scripts/slurm/package_hga_all_tasks.sh`

**Output:** `results/{Task}(bipolar)(hammers)/sub-*/HGA/*_time.csv`

### Stage 5 — Analysis and figures

- Publication insula panels: `vizpub/fig2v1.ipynb` (dual-atlas capable)
- Interactive explorer: `viewer/hga_explorer/` with default atlas **hammers**
- Export QA: `viewer/hga_explorer/scripts/qa_export.py` (Hammers AIC/PIC spot
  check)

---

## 6. aparc path (reference only)

aparc does **not** require MAPER:

```bash
python common/parcellation.py --atlas aparc2009s ...
python src/hga/package_highgamma.py --atlas aparc2009s ...
```

Use aparc for whole-brain ROI decoding and cross-atlas comparison. Do **not**
combine aparc insula labels with Hammers `mix=False` rules in the same analysis
without explicit dual-atlas reporting.

---

## 7. Reference documents

| Document | Contents |
|----------|----------|
| [`pipeline/README.md`](../pipeline/README.md) | MAPER steps 01–03, commands |
| [`pipeline/ENVIRONMENT.md`](../pipeline/ENVIRONMENT.md) | Container, paths, versions |
| [`PARCELLATION_PIPELINE.md`](PARCELLATION_PIPELINE.md) | Coordinate columns, bipolar consensus, technical rules |
| [`../pipeline/D44_MAPER_worklog.md`](../pipeline/D44_MAPER_worklog.md) | D0044 pilot history and bug fixes |
| [`HGA_EXPLORER.md`](HGA_EXPLORER.md) | Explorer export schema and insula filters |

---

## 8. Follow-up (not done in this pass)

- Merge and review `hammers-parcellation` in **seeg-preprocessing** (Stage 2
  production code).
- Clean **legacy notebook** references to `results/exlude_insula.csv` (~20
  files); active fig2 work uses `fig2v1` with Hammers `mix=False` instead.
- Slim `PARCELLATION_PIPELINE.md` §8+ (`maper_*` electrode fields, removed
  `pipeline/04_extract_labels`); mark as deferred unless manuscript requires
  six-region MAPER tables.
- Historical paths in `pipeline/D44_MAPER_worklog.md` may cite old `scripts/` MAPER
  entry points; canonical paths are under `pipeline/`.

---

## Changelog

- 2026-07-12: Initial baseline — Hammers primary for insula, aparc reference,
  retired filter_insula / exlude_insula, MAPER stays in insula `pipeline/`.
