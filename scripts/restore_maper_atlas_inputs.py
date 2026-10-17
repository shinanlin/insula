#!/usr/bin/env python3
"""Recover original n30r95 inputs, retaining raw archives on persistent storage.

Uses the author's release and the project's existing geometry-correction step.
Only prepares/validates inputs; no participant MAPER jobs or NMF are started.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from urllib.request import urlopen

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

BASE = Path('/cwork/ns458/insula-paper')
CACHE = Path('/hpc/group/coganlab/nanlinshi/cache/maper/hammers_n30r95')
ANC = Path('/cwork/ns458/maper_run/restore_20261017/ancillaries')
OUT = BASE / 'results/audit/globus_recon_restore_2026-10-17'
URL = 'https://soundray.org/hammers-n30r95/'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    raw = CACHE / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    downloads = []
    for name in ['Hammers-MRIs-n30.tar', 'Hammers-labelsets-n30r95.tar']:
        path = raw / name
        if not path.is_file():
            partial = path.with_suffix('.tar.part')
            with urlopen(URL + name, timeout=60) as response, partial.open('wb') as f:
                shutil.copyfileobj(response, f)
            with tarfile.open(partial) as tar:
                members = tar.getmembers()
                if any(Path(m.name).is_absolute() or '..' in Path(m.name).parts or
                       m.issym() or m.islnk() for m in members):
                    raise ValueError(f'Unsafe atlas archive: {name}')
            partial.rename(path)
        downloads.append(dict(url=URL + name, path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
        print(f'Restored {name} ({path.stat().st_size} bytes)', flush=True)
    subprocess.run([sys.executable, str(BASE / 'pipeline/02_prepare_atlas/prepare_hammers_native_pairs.py'),
                    '--atlas-root', str(CACHE)], check=True)
    ANC.mkdir(parents=True, exist_ok=True)
    companion = Path('/hpc/group/coganlab/nanlinshi/cache/maper/hammers_mith-ancillaries-n30r95.tar')
    with tarfile.open(companion) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if not member.isfile() or not parts or parts[0] not in {'onepad','posnorm'}:
                continue
            if Path(member.name).is_absolute() or '..' in parts:
                raise ValueError('Unsafe companion archive member')
            dest = ANC / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, dest.open('wb') as target:
                shutil.copyfileobj(source, target)
    corrected = CACHE / 'derivatives/individual_native_pairs/geometry_corrected_labels'
    records = []
    for i in range(1,31):
        src = corrected / f'a{i:02d}_labels_r95_geometry-fixed.nii.gz'
        dst = ANC / f'seg/seg95/a{i}.nii.gz'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        t1 = ANC / f'onepad/a{i}.nii.gz'
        transformation = ANC / f'posnorm/a{i}.dof.gz'
        a, b = nib.load(t1), nib.load(dst)
        grid_standardized = False
        max_index_displacement = 0.0
        constant_padding_changes = 0
        if a.shape[:3] != b.shape[:3] or not np.allclose(a.affine, b.affine, atol=1e-3):
            original_t1 = nib.load(CACHE / f'derivatives/individual_native_pairs/extracted/a{i:02d}.nii.gz')
            same_voxels = (original_t1.shape == a.shape and
                           np.array_equal(np.asarray(original_t1.dataobj), np.asarray(a.dataobj)))
            print(f'Atlas a{i}: raw MRI/companion voxels identical={same_voxels}; '
                  f'raw affine={original_t1.affine.tolist()}; companion affine={a.affine.tolist()}', flush=True)
            if a.shape[:3] != b.shape[:3]:
                raise ValueError(f'Atlas a{i}: source/companion label grid shapes differ')
            # Some companion grids round 0.9375 mm spacing to 0.938 mm.
            # Regrid labels in physical coordinates with nearest-neighbour
            # interpolation. Accept this repair only if every voxel maps to
            # its original index AND the complete label array is unchanged.
            # Extend the nearest boundary value: constant padding would
            # erase edge labels when a centre moves slightly below index 0.
            corners = np.array([[x,y,z] for x in [0,a.shape[0]-1]
                                for y in [0,a.shape[1]-1] for z in [0,a.shape[2]-1]])
            mapped = nib.affines.apply_affine(np.linalg.inv(b.affine) @ a.affine, corners)
            max_index_displacement = float(np.max(np.abs(mapped - corners)))
            if max_index_displacement >= .5:
                raise ValueError(f'Atlas a{i}: geometry offset is not subvoxel grid rounding')
            original_labels = np.asarray(b.dataobj)
            constant = resample_from_to(b, (a.shape[:3], a.affine), order=0)
            constant_padding_changes = int(np.count_nonzero(
                original_labels != np.asarray(constant.dataobj)))
            regridded = resample_from_to(b, (a.shape[:3], a.affine), order=0, mode='nearest')
            if not np.array_equal(original_labels, np.asarray(regridded.dataobj)):
                raise ValueError(f'Atlas a{i}: companion-grid resampling changes label voxels')
            nib.save(regridded, dst)
            b = nib.load(dst)
            grid_standardized = True
        if a.shape[:3] != b.shape[:3] or not np.allclose(a.affine, b.affine, atol=1e-3):
            raise ValueError(f'Atlas a{i}: geometry verification failed after header repair')
        if not transformation.is_file() or transformation.stat().st_size == 0:
            raise ValueError(f'Missing atlas a{i} pretransformation')
        label_values = np.unique(np.asarray(b.dataobj))
        if not np.isin(np.arange(86,96), label_values).all():
            raise ValueError(f'Atlas a{i}: expected insular short-gyrus labels missing')
        records.append(dict(atlas_id=f'a{i}', t1=str(t1), labels=str(dst),
                            transformation=str(transformation), labels_sha256=sha(dst),
                            geometry_matches=True, n_labels=len(label_values),
                            companion_grid_standardized_without_changing_labels=grid_standardized,
                            max_source_index_displacement_voxels=max_index_displacement,
                            constant_padding_would_change_label_voxels=constant_padding_changes))
    desc = 'id, onepad, pretransformation, seg95\n' + ''.join(
        f'a{i}, onepad/a{i}.nii.gz, posnorm/a{i}.dof.gz, seg/seg95/a{i}.nii.gz\n'
        for i in range(1,31))
    (ANC / 'src-description.csv').write_text(desc)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_page=URL,
        raw_archives=downloads, persistent_cache=str(CACHE), ancillaries=str(ANC),
        n_atlases=30, all_geometry_matches=True, atlas_records=records,
        participant_maper_started=False, model_fitted=False,
        companion_archive_sha256=sha(companion))
    (OUT / 'atlas_input_restore.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Validated 30 complete atlas/T1/transform sets; participant jobs not started.', flush=True)
