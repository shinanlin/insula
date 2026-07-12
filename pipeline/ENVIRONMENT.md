# MAPER Insula Pipeline — Environment Setup (Duke DCC)

This note records the exact software environment the D0044 MAPER pipeline was
built and validated against, so a future run on a new subject (or a new
cluster) reproduces the same behavior instead of re-discovering the two bugs
documented in `D44_MAPER_worklog.md`.

## 1. Conda environment (`ieeg`)

Used for steps 02 (atlas geometry fix) and 04 (label extraction at
electrodes) — anywhere the pipeline needs `nibabel`/`numpy`/`pandas`/`scipy`.

```bash
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
```

Versions verified working (2026-07-06):

| package | version |
|---|---|
| python  | 3.11.11 |
| nibabel | 5.3.2   |
| numpy   | 2.0.2   |
| pandas  | 2.2.3   |
| scipy   | 1.16.1  |

No other conda env on this account (`mtrf-gpu`, `embedding`,
`acpcdetect-2.2/runtime`) is relevant to this pipeline.

## 2. FreeSurfer (module system)

Step 03 (`prepare_target.sh`) needs `mri_convert` on `PATH`. It is not on
`PATH` by default — load the environment module:

```bash
module load FreeSurfer/7.2.0
which mri_convert   # -> /opt/apps/rhel8/freesurfer-7.2.0/bin/mri_convert
```

`FreeSurfer/7.2.0` is the only FreeSurfer module currently installed on this
cluster (`module -t avail | grep -i freesurfer`). If a newer module appears
later, re-verify `mri_convert` output grid conventions before switching —
the pipeline assumes FreeSurfer's tkRAS volume-center convention (see bug 2
in the worklog).

## 3. Apptainer / Singularity

```bash
apptainer --version   # -> apptainer version 1.4.5-3.el9 (verified 2026-07-06)
```

Already on `PATH` on both the login node and compute nodes — no module load
needed. The container is built once (`01_build_container/`) and reused
read-only for every subject; no apptainer action is needed per-subject
beyond `apptainer exec` in `run_maper.sbatch`.

**Base image pin:** `maper_container.def` pins `Bootstrap: docker` /
`From: ubuntu:22.04`. Do not downgrade to `ubuntu:20.04` — apptainer's
unprivileged fakeroot shim (`faked`) requires glibc ≥2.33/2.34, and
20.04 ships glibc 2.31, which fails the build (see worklog sec. 3.2). 22.04
ships glibc 2.35 and both MIRTK and NiftySeg compile cleanly on it.

## 4. Storage layout (paths this pipeline assumes)

| purpose | path |
|---|---|
| workspace / git repo | `/hpc/group/coganlab/nanlinshi/insula` |
| Slurm scripts | `insula/scripts/` |
| Slurm logs | `insula/logs/` |
| built container | `/hpc/group/coganlab/nanlinshi/maper_tool/maper.sif` |
| atlas database + geometry-corrected labels | `/cwork/ns458/atlases/Hammersmith_n30r95/` |
| per-run MAPER working dir (target/ancillaries/output) | `/cwork/ns458/maper_run/` |
| subject FreeSurfer recon | `/cwork/ns458/ECoG_Recon/<SUBJECT_NUM>/mri/` |
| subject BIDS electrodes | `/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/sub-<SUBJECT>/ieeg/` |

`/hpc/home/ns458` is capped at ~20GB and must never hold conda envs, caches,
or pipeline outputs — everything data-heavy lives under `/cwork/ns458`
(scratch, large but not backed up) or `/hpc/group/coganlab/nanlinshi`
(group storage, backed up, used for the container image and the git repo).

## 5. One-time vs per-subject re-verification checklist

Re-run these sanity checks if moving to a new cluster or after any module
update — both were the direct cause of one of the two documented bugs:

1. After building/rebuilding `maper.sif`: run its `%test` section
   (`apptainer test maper.sif`) — confirms `mirtk`, `seg_maths`, `maper` are
   all on `PATH` inside the container and prints `BUILD_INFO.txt` (base
   image, compiler, and the three tools' git commit hashes).
2. After `setup_maper_ancillaries.sh`: confirm the printed affine-match
   check between `ancillaries/onepad/aXX.nii.gz` and
   `ancillaries/seg/seg95/aXX.nii.gz` is <0.05mm — this is bug 1's
   regression guard.
3. After `extract_maper_insula.py`: confirm its printed tkRAS
   volume-center-convention warning does NOT fire (translation should be
   close to [128, -128, 128] for a standard 256^3 1mm FreeSurfer conformed
   volume) — this is bug 2's regression guard.
