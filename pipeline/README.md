# MAPER Insula Native-Space Labeling Pipeline

Multi-atlas propagation (MAPER: MIRTK registration + NiftySeg label fusion)
of the Hammersmith n30r95 atlas database onto a subject's native FreeSurfer
space. Fused volumes under `maper_run` are used to generate Hammersmith
parcellation CSVs; analysis reads those CSVs directly.

Validated end-to-end on subject D0044 (see
`docs/D44_MAPER_worklog.md`). Steps 01–03 produce reusable fused segmentations;
hammers electrode labels are written to `derivatives/parcellation/sub-*_hammers.csv`
via `seeg-preprocessing` `parcellation.py --atlas hammers`.

## Prerequisites

See `ENVIRONMENT.md` in this directory for verified versions and storage paths.

- Duke DCC access, `ieeg` conda environment, FreeSurfer on PATH.
- Apptainer/Singularity on compute nodes.
- Hammersmith n30r95 atlas under `$ATLAS_ROOT/raw/...`.
- Subject FreeSurfer recon-all and BIDS electrode coordinates.

## Directory map

```
pipeline/
  01_build_container/     one-time: build maper.sif
  02_prepare_atlas/       one-time: fix atlas geometry, build ancillaries
  03_run_maper/           per-subject: target T1, 30-way fusion → seg95 volume
```

Steps 01–02 run once per cluster/atlas download. Step 03 runs per subject when
regenerating fused volumes.

## Per-subject MAPER fusion (step 03)

```bash
SUBJECT=D0045
RUN=/cwork/<user>/maper_run/$SUBJECT
mkdir -p "$RUN"
ln -s /cwork/<user>/maper_run/ancillaries "$RUN/ancillaries"

bash 03_run_maper/prepare_target.sh "$SUBJECT" \
    /cwork/<user>/ECoG_Recon/${SUBJECT#D0}/mri "$RUN/target"
bash 03_run_maper/generate_launchlist.sh "$SUBJECT" "$RUN" \
    /hpc/group/coganlab/nanlinshi/maper_tool/maper.sif
sbatch --export=SUBJECT=$SUBJECT,RUN=$RUN pipeline/03_run_maper/run_maper.sbatch
# output: $RUN/output/f30-seg95-${SUBJECT}.nii.gz
```

## Hammers parcellation CSV (analysis input)

After fusion exists, run `seeg-preprocessing` parcellation for each subject:

```bash
conda activate ieeg
python path/to/seeg-preprocessing/common/parcellation.py \
    --atlas hammers \
    --subject $SUBJECT \
    ...
```

This writes `derivatives/parcellation/sub-${SUBJECT}_hammers.csv` (plus the
existing `*_aparc2009s.csv`). Downstream packaging uses:

```bash
python src/package_HGA.py --bids_root <BIDS>/ --band highgamma --ref bipolar --atlas aparc2009s
python src/package_HGA.py --bids_root <BIDS>/ --band highgamma --ref bipolar --atlas hammers
```

Outputs land in `results(nw)/{task}({ref})(aparc2009s)/` and `results(nw)/{task}({ref})(hammers)/`.

## Known limitations

- MAPER fusion quality varies with pathology and registration error.
- Hammers `roi` may include mixed labels (e.g. `PIC–AIC`); fig2 analyses keep
  pure `AIC`/`PIC` only (`mix=False`).
- Keep aparc and hammers packaged HGA separate; do not merge atlas columns in
  `package_HGA.py`.

## Two bugs — do not regress

1. **Atlas ancillary geometry (step 02).** Use geometry-corrected seg95 copies in
   `ancillaries/seg/seg95/`, not raw downloaded labels.
2. **Electrode coordinates (parcellation).** Convert tkRAS via subject `orig.mgz`
   `vox2ras_tkr`, not the fused NIfTI affine alone.

Full validation narrative: `docs/D44_MAPER_worklog.md`.
