# Electrode Parcellation and Insula Subregion Pipeline

This document records the current, approved interpretation of electrode
coordinates and anatomical labels. It separates the general native-space
parcellation from the Insula-specific Faillenot/MAPER analysis.

## 1. Scope and source of truth

Anatomical labels are assigned in each subject's native FreeSurfer space.
Neither fsaverage nor MNI152 participates in the primary ROI lookup.

The general parcellation uses:

- native electrode coordinates;
- the subject's `orig.mgz` geometry;
- the subject-native `aparc.a2009s+aseg.mgz`;
- header-derived `vox2ras_tkr`, not hand-written `128/126` constants.

MNI and CVS coordinates are derived display/analysis coordinates. They must
never overwrite native labels.

The canonical preprocessing implementation is maintained outside this
analysis repository:

```text
/hpc/group/coganlab/nanlinshi/seeg-preprocessing/
```

Shared code, including native parcellation, is on `main` under `common/`.
Each task is a separate git branch checked out under:

```text
/hpc/group/coganlab/nanlinshi/seeg-preprocessing-worktrees/
```

The five current worktrees are `lexical_delay`, `lexical_nodelay`,
`phoneme_seq`, `picture_naming`, and `sentence_rep`. Dataset `BIDS/code`
paths are symlinks into the corresponding task worktree. Do not recover or
edit an old copied `BIDS/code/parcellation.py`; changes to the shared
algorithm belong in `seeg-preprocessing/common/parcellation.py` and must be
propagated through the worktree branches normally.

## 2. Coordinate columns

Current parcellation CSVs use three coordinate representations, all in mm:

| Columns | Space | Intended use |
|---|---|---|
| `x, y, z` | subject-native FreeSurfer tkRAS | Native `mne.viz.Brain`, native atlas lookup after `inv(vox2ras_tkr)` |
| `x_mni, y_mni, z_mni` | FSL MNI152NLin6Asym scanner RAS | Volumetric MNI analyses, standard-space atlases, Nilearn |
| `x_t, y_t, z_t` | `cvs_avg35_inMNI152` FreeSurfer tkRAS | Direct comparison with that subject's surface vertices and MNE surface display |

The MNI-to-CVS conversion must be computed from
`cvs_avg35_inMNI152/mri/orig.mgz`; do not use a surface-centroid translation.
The former fsaverage-to-CVS centroid shift is obsolete.

SEEG contacts and bipolar midpoints are volumetric points. Projecting them to
the nearest pial vertex is a display operation, not part of registration or
anatomical labeling.

## 3. Native atlas lookup

For a tkRAS point in mm, obtain continuous voxel coordinates using the
subject header:

```python
ijk = np.linalg.inv(orig.header.get_vox2ras_tkr()) @ [x, y, z, 1]
```

Use the subject-native atlas on the same conformed grid. Do not use
`inv(image.affine)` on tkRAS coordinates, and do not use a hard-coded
`128/128/126` conversion.

### Historical MATLAB results

The MATLAB CSVs in `ECoG_Recon/<subject>/elec_recon/` are historical label
outputs. `electrodes.tsv` stores coordinates, not MATLAB labels.

The old Python implementation reproduced MATLAB exactly. That establishes
translation fidelity, not anatomical correctness. MATLAB's `+1`, axis flip,
and 1-based indexing implicitly produce the legacy `126` term and a fixed
approximately 2 mm lookup displacement relative to the FreeSurfer header.
The header-based output is the current method; MATLAB results are retained as
an audit comparison only.

## 4. General bipolar ROI rule

Keep the two physical endpoint labels and the midpoint label. For ROI
consensus, inspect gross ROIs in spatial order:

```text
Contact 1 -> midpoint -> Contact 2
```

White matter, Unknown, and hypointensity do not vote in tissue ROI consensus.

- One remaining tissue ROI: return it, `mix=False`.
- Two or more tissue ROIs: return their ordered unique names joined by `–`,
  `mix=True`.
- Only white matter/Unknown: return `WM` if any white matter is present;
  otherwise `Unknown`, with `mix=False`.

Examples:

```text
INS + WM + INS          -> INS, False
STG + WM + WM           -> STG, False
INS + Subcentral + INS  -> INS–Subcentral, True
```

Neighborhood or sphere results are sensitivity/QC measures. They must not
silently replace the exact-point primary label.

## 5. Faillenot/MAPER Insula subregions

Faillenot is a downstream, Insula-specific analysis. It does not replace the
general Destrieux parcellation.

Hammersmith n30r95 IDs used here are:

| IDs | Faillenot region | AP grouping |
|---|---|---|
| 86/87 | ASG | Anterior |
| 88/89 | MSG | Anterior |
| 90/91 | PSG | Anterior |
| 92/93 | anterior pole / anterior inferior cortex | Anterior |
| 94/95 | ALG | Posterior |
| 20/21 | PLG | Posterior |

Even IDs are left and odd IDs are right. In this atlas, AIC means *anterior
inferior cortex*, not the whole anterior insular cortex.

The production candidate is native-space MAPER: propagate all 30 individual
atlases to the subject, then fuse the labels. The lightweight group
probability-map warp is retained as an independent comparison, not a gold
standard.

Two fixes are mandatory:

1. Correct each downloaded atlas segmentation's geometry to match its paired
   T1 before building MAPER ancillaries.
2. Sample the fused native segmentation with the subject's
   `inv(vox2ras_tkr)`, never with `inv(fused.affine)`.

The exact voxel is the primary MAPER result. A 2 mm sphere is a sensitivity
analysis and must report both the winning fraction and the number of Insula
voxels contributing to it.

## 6. Validation status

D0044 is a successful corrected pilot, not yet a cohort-wide validation.

- Exact-point comparison: 9/11 comparable Insula channels agreed with the
  lightweight method (81.8%).
- With the 2 mm fallback: 10/12 agreed (83.3%).
- The two disagreements, LI7-8 and RI5-6, were PSG/ALG boundary differences.
- Among 22 AMT channels, the lightweight method labeled 6 as Insula while
  corrected MAPER labeled 0. These are *MAPER-unconfirmed* or *suspected
  lightweight false positives*; MAPER is not an independent manual gold
  standard.

Before using MAPER labels as the manuscript's primary AIC/PIC assignment:

1. repeat the full pipeline and visual QC in at least 3-5 additional subjects;
2. inspect exact slices through every included Insula channel;
3. retain endpoint, midpoint, exact-point, and 2 mm sensitivity results;
4. add 30-atlas vote confidence rather than relying only on the fused hard
   label;
5. flag PSG/ALG boundary channels instead of presenting false certainty.

## 7. Reproducibility and current paths

Reusable MAPER code is under `pipeline/`; the operational guide is
`pipeline/README.md`. The D44 history, including failed attempts, is in
`pipeline/D44_MAPER_worklog.md`.

Important external roots:

```text
/cwork/ns458/ECoG_Recon/
/cwork/ns458/atlases/Hammersmith_n30r95/
/cwork/ns458/maper_run/
/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/parcellation/
```

MAPER subject outputs are organized under:

```text
/cwork/ns458/maper_run/<SUBJECT>/
├── output/f30-seg95-<SUBJECT>.nii.gz
└── sub-<SUBJECT>_desc-maper_insula.csv
```

The apparent absence of a copied dataset-level `code/parcellation.py` is
expected in the consolidated layout: the canonical implementation is
`seeg-preprocessing/common/parcellation.py`, while each dataset's `BIDS/code`
contains task glue through a worktree symlink.

Do not use these failed/obsolete products:

- the first MAPER fusion made from uncorrected atlas label geometry;
- the first MAPER electrode table sampled with scanner affine;
- fsaverage/CVS centroid translations;
- MNI or CVS template labels as replacements for native Destrieux labels.

## 8. Reconciling Destrieux and MAPER

The two atlases answer different questions and must remain in separate
namespaces:

- Destrieux/aparc: general, subject-native whole-brain ROI and independent
  anatomical QC.
- MAPER/Faillenot: Insula membership and six-region Anterior/Posterior
  subdivision for this project.

Do not average the atlases, let one silently overwrite the other, or convert
disagreement to a generic `Mixed` label. Preserve the current aparc columns
and prefix every MAPER-derived field with `maper_`.

### Current cohort comparison

Across the 92 currently extracted MAPER subject tables (14,884 bipolar rows),
midpoint exact-voxel results are:

| Destrieux `roi` contains INS | MAPER exact Insula | Count |
|---|---|---:|
| yes | yes | 403 |
| yes | no | 284 |
| no | yes | 147 |
| no | no | 14,050 |

These are historical atlas-agreement counts, not final inclusion counts. The
production extractor now samples Contact 1, midpoint, and Contact 2; the older
tables summarized above sampled only the midpoint.

### Production three-point fields

The production extractor saves exact MAPER labels for all three locations:

```text
maper_contact_1_id / region6 / ap
maper_center_id / region6 / ap
maper_contact_2_id / region6 / ap
maper_insula_status
maper_ap_consensus
maper_ap_mix
maper_atlas_agreement
```

Recommended `maper_insula_status` values:

- `core`: Contact 1, midpoint, and Contact 2 are all MAPER Insula.
- `partial`: at least one but not all three are MAPER Insula.
- `none`: none of the three are MAPER Insula.

AP consensus is computed only from MAPER Insula locations, in spatial order.
One AP class gives `Anterior` or `Posterior`; both give
`Anterior–Posterior, mix=True`. A 2 mm result is stored separately as
boundary sensitivity and never upgrades an exact-negative channel into
`core`.

Recommended `maper_atlas_agreement` values compare the MAPER three-point
status with the existing aparc consensus:

- `concordant_insula`
- `maper_only`
- `aparc_only`
- `concordant_noninsula`

For this four-level comparison, both `core` and `partial` mean MAPER
Insula-present. The separate `maper_insula_status` field preserves whether
membership is complete or partial.

### Analysis policy

If MAPER is adopted as the manuscript method:

1. Primary AIC/PIC analysis uses `maper_insula_status == core` and MAPER AP
   consensus.
2. `partial`, sphere-only, `maper_only`, and `aparc_only` channels remain in
   sensitivity/QC tables and receive exact-slice review; they are not silently
   promoted or discarded.
3. Destrieux remains visible in every output and can identify gross
   cross-region conflicts, but it does not veto a reviewed MAPER assignment.
4. Report agreement and sensitivity analyses so the atlas choice is
   transparent.

### Task identity is part of the key

MAPER segmentation is subject-level, but bipolar montages are task-derived.
Among 44 subjects currently present in multiple task parcellations, 19 have
different bipolar channel sets. Therefore the final table key is:

```text
task + subject + reference + channel name
```

Do not use the current fixed task-priority lookup as the final manuscript
merge. Reuse the subject's fused MAPER volume, but run label extraction once
against each task's own parcellation CSV. Identical channel sets may be
verified and deduplicated only after an explicit equality check.

The cohort runner, validator, and summary entry points are:

```text
pipeline/04_extract_labels/batch_extract_maper_labels.sh
pipeline/04_extract_labels/validate_maper_derivatives.py
pipeline/04_extract_labels/summarize_maper_derivatives.py
```

The manifest selects the unqualified canonical file
`sub-<SUBJECT>_aparc2009s.csv` when historical variants such as
`*_proc-3mm_aparc2009s.csv` coexist. A QC variant is not a second
task/reference identity and must not overwrite the canonical derivative.

### Production extraction status (2026-07-08)

The corrected task-specific run completed and passed full automated
validation:

- 175 unique task+subject combinations discovered;
- 173 ready combinations extracted successfully;
- 28,561 bipolar rows with 28,561 unique task+subject+reference+name keys;
- 500 `core`, 629 `partial`, and 27,432 `none` channels;
- D0031 and D0091 retained as `missing_fused` and not assigned empty labels.

The validated manifest is
`/cwork/ns458/maper_run/manifests/maper_extract_all_20260708T191620Z.tsv` and
the cohort summaries are under
`/cwork/ns458/maper_run/summaries/20260708T191620Z/`. This automated cohort
extraction does not replace the planned exact-slice manual QC in 3–5 subjects.
