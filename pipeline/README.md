# MAPER Insula Native-Space Labeling Pipeline

Multi-atlas propagation (MAPER: MIRTK registration + NiftySeg label fusion)
of the Hammersmith n30r95 atlas database's Faillenot six-region insula labels
onto a single subject's native FreeSurfer space, for use in sEEG electrode
anatomical assignment.

Validated end-to-end on subject D0044 (see
`insula/docs/D44_MAPER_worklog.md` for the full narrative, including two bugs
found and fixed along the way). This directory is the reusable, generalized
form of that pilot — use it directly for the next subject.

## Prerequisites

See `ENVIRONMENT.md` in this directory for exact verified versions
(conda env package versions, FreeSurfer module, apptainer version) and the
full storage-path layout this pipeline assumes.

- Duke DCC access, `ieeg` conda environment (for `nibabel`/`numpy`/`pandas`
  in steps 02/04) and FreeSurfer on PATH (for `mri_convert` in step 03).
- Apptainer/Singularity on the compute node (already true on DCC).
- Hammersmith n30r95 atlas database downloaded to
  `$ATLAS_ROOT/raw/{Hammers-MRIs-n30.tar,Hammers-labelsets-n30r95.tar}`
  (licensed download from brain-development.org; not redistributed here).
- Subject's FreeSurfer recon-all output (`orig.mgz`, `brainmask.mgz`,
  `aparc.a2009s+aseg.mgz`) and BIDS electrode coordinates in
  `*_space-ACPC_electrodes.tsv` (tkRAS mm) or equivalent.

## Directory map

```
pipeline/
  01_build_container/     one-time: build maper.sif (Apptainer)
  02_prepare_atlas/       one-time (shared across subjects): fix atlas
                           geometry bug, build MAPER ancillaries tree
  03_run_maper/            per-subject: prepare target T1, generate
                           launchlist, run 30-way array job + fusion
  04_extract_labels/       per-subject: sample fused labels at electrodes
```

Steps 01 and 02 run ONCE per cluster/atlas-download and are then reused,
unchanged, for every subject. Steps 03 and 04 run PER SUBJECT.

## Step-by-step: onboarding a new subject

Assume `SUBJECT=D0045`, atlas root `/cwork/<user>/atlases/Hammersmith_n30r95`,
run directory `/cwork/<user>/maper_run`, container at
`/hpc/group/coganlab/nanlinshi/maper_tool/maper.sif` (already built — skip
step 1 entirely if reusing the existing .sif).

### 0. One-time (skip if already done for this cluster)

```bash
# 01: build the container (see 01_build_container/README notes in maper_container.def header)
sbatch scripts/build_maper_container.sbatch     # ~30-60 min

# 02a: fix the Hammersmith atlas geometry bug (see docs/D44_MAPER_worklog.md sec.4)
conda activate ieeg
python 02_prepare_atlas/prepare_hammers_native_pairs.py \
    --atlas-root /cwork/<user>/atlases/Hammersmith_n30r95

# 02b: build the MAPER ancillaries tree (onepad/posnorm from MAPER's own
# downloader, then seg/seg95 overwritten with the geometry-corrected labels)
mkdir -p /cwork/<user>/maper_run/ancillaries
apptainer exec -B /cwork/<user> maper.sif /opt/maper/hammers_mith-ancillaries.sh \
    <atlas_download_dir_containing_Hammers_mith-n30r95> /cwork/<user>/maper_run/ancillaries
bash 02_prepare_atlas/setup_maper_ancillaries.sh \
    /cwork/<user>/atlases/Hammersmith_n30r95 /cwork/<user>/maper_run/ancillaries
```

### 1. Per subject

```bash
RUN=/cwork/<user>/maper_run/$SUBJECT     # keep each subject's run isolated
                                          # (or reuse a shared RUN dir and
                                          # just swap target/ + output/ per subject —
                                          # ancillaries/ is shared either way)
mkdir -p "$RUN"
ln -s /cwork/<user>/maper_run/ancillaries "$RUN/ancillaries"   # share the atlas ancillaries

# 03a: prepare target T1 (skull-stripped, same grid as orig.mgz)
bash 03_run_maper/prepare_target.sh "$SUBJECT" \
    /cwork/<user>/ECoG_Recon/${SUBJECT#D0}/mri "$RUN/target"

# 03b: generate the 30-line launchlist
bash 03_run_maper/generate_launchlist.sh "$SUBJECT" "$RUN" \
    /hpc/group/coganlab/nanlinshi/maper_tool/maper.sif

# 03c: submit the 30-way array job (fusion auto-triggers after all 30 finish)
sbatch --export=SUBJECT=$SUBJECT,RUN=$RUN scripts/run_maper.sbatch
# monitor: squeue -u $USER ; check logs/maper_<jobid>_*.out
# expected: ~4 min/task wall, ~1.6GB peak RSS, ~10 min total incl. fusion
# output: $RUN/output/f30-seg95-${SUBJECT}.nii.gz  (fused hard labels, 95 regions)

# 04: extract labels at bipolar endpoints and midpoint
conda activate ieeg
python 04_extract_labels/extract_maper_parcellation.py \
    --task LexicalDecRepDelay \
    --subject $SUBJECT \
    --fused "$RUN/output/f30-seg95-${SUBJECT}.nii.gz" \
    --tissue "$RUN/output/f30-seg95-${SUBJECT}-tc3crisp.nii.gz" \
    --orig /cwork/<user>/ECoG_Recon/${SUBJECT#D0}/mri/orig.mgz \
    --propagated-dir "$RUN/output/$SUBJECT" \
    --parcellation-csv path/to/sub-${SUBJECT}_aparc2009s.csv \
    --contacts-tsv path/to/sub-${SUBJECT}_electrodes.tsv \
    --lut 04_extract_labels/Hammers_mith_atlases_n30r95_label_indices_SPM12_20160111.txt \
    --output sub-${SUBJECT}_desc-maper95_bipolar.csv \
    --sensitivity-output sub-${SUBJECT}_desc-maper95Sphere2mm_bipolar.csv
```

The subject-level fused segmentation is reusable across tasks, but the
electrode table is task-specific. Run step 04 separately for every task's
parcellation CSV and include task/reference in the output identity. Do not
use `04_extract_labels/maper_parcellation.py`'s fixed task priority as the
final manuscript merge: among 44 subjects shared by multiple current tasks,
19 have different bipolar channel sets.

The production extractor applies the same ordered three-point rule as the
general aparc parcellation. It preserves the source aparc table unchanged and
appends only `maper_` fields. Exact membership is `core` (3/3 Insula points),
`partial` (1/3 or 2/3), or `none` (0/3); `maper_insula_points` retains the
exact count. The separate 2 mm sphere table is boundary sensitivity only.
Thirty propagated segmentations provide vote confidence at every location.

For cohort extraction and validation:

```bash
bash 04_extract_labels/batch_extract_maper_labels.sh --dry-run
bash 04_extract_labels/batch_extract_maper_labels.sh --submit
python 04_extract_labels/validate_maper_derivatives.py \
    --manifest /cwork/<user>/maper_run/manifests/maper_extract_all_<STAMP>.tsv \
    --failure-manifest /cwork/<user>/maper_run/manifests/maper_extract_retry_<STAMP>.tsv
python 04_extract_labels/summarize_maper_derivatives.py \
    --manifest /cwork/<user>/maper_run/manifests/maper_extract_all_<STAMP>.tsv \
    --output-dir /cwork/<user>/maper_run/summaries/<STAMP>
```

The manifest uses exactly one parcellation table per task and subject. It
prefers `sub-<SUBJECT>_aparc2009s.csv`; historical files such as
`*_proc-3mm_aparc2009s.csv` are not additional task/reference runs and must
not share or overwrite the canonical output.

The 2026-07-08 production extraction passed validation for 173 ready
task+subject combinations (28,561 unique bipolar channels). D0031 and D0091
remain explicitly `missing_fused`. See `docs/PARCELLATION_PIPELINE.md` for
status and cohort summary paths.

## Two bugs baked into this pipeline's fixes — do not regress them

1. **Atlas ancillary geometry (step 02).** The raw Hammersmith seg95 label
   volumes carry a displaced NIfTI affine relative to their paired T1
   (16-30mm translation offset, wrong `sform_code`). `setup_maper_ancillaries.sh`
   overwrites `ancillaries/seg/seg95/` with `prepare_hammers_native_pairs.py`'s
   geometry-corrected copies. **Never point `ancillaries/seg/seg95/` at the
   raw downloaded labels directly** — verify affine match to the paired
   `onepad/` T1 (the setup script prints a one-liner to check this) before
   trusting any run.
2. **Electrode coordinate frame (step 04).** BIDS electrode x/y/z are
   FreeSurfer tkRAS, not the fused segmentation's own scanner-space affine.
   `extract_maper_parcellation.py` always builds voxel indices from
   `inv(vox2ras_tkr)` read off the subject's own `orig.mgz`, and warns if
   that transform's translation doesn't look like the volume-center
   convention. **Never use the fused NIfTI's own affine to convert
   coordinates** — it does not carry tkRAS semantics.

Full narrative and quantitative validation (six-region centroid ordering,
175-channel MAPER-vs-lightweight-method comparison, and AMT disagreement
rates) is in `insula/docs/D44_MAPER_worklog.md`. Exact-point agreement was
9/11 (81.8%); agreement with the 2 mm fallback was 10/12 (83.3%). AMT
channels labeled as Insula only by the lightweight method are suspected
false positives, not manually proven false positives.

## Known limitations / before generalizing further

- Validated on one subject (D0044) so far. Recommended: repeat on 3-5 more
  subjects before adopting MAPER native labels as the manuscript's primary
  insula-electrode assignment (see worklog sec. 7).
- Faillenot et al. (2017) report Dice≈0.79 for MAPER against manual labels
  in **healthy** leave-one-out validation; sEEG patients' structural
  anomalies (resections, edema, electrode artifact) may push registration
  error higher than that baseline.
- PSG/ALG boundary electrodes may disagree by one region between MAPER and
  the lightweight (population-atlas-warped) method; consider flagging such
  electrodes as boundary/transitional rather than forcing a single label.
- Keep aparc/Destrieux and MAPER columns side by side. MAPER is the candidate
  primary Insula/AP classification; Destrieux remains whole-brain anatomical
  QC. Atlas disagreement must be explicit, not collapsed into `Mixed` or
  resolved by silent overwrite.
