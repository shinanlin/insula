#!/usr/bin/env python3
"""Verify restored MRI geometry and re-sample all 131 Hammers-zero contacts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from audit_zero_endpoints import classify, load_lut, read_recon_contact_positions

BASE = Path('/cwork/ns458/insula-paper')
OUT = BASE / 'results/audit/globus_recon_restore_2026-10-17'
OLD = BASE / 'docs/evidence/zero_endpoints_2026-10-17'
RECON = Path('/cwork/ns458/ECoG_Recon')
SHARED = Path('/hpc/group/coganlab/ECoG_Recon')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    files = pd.read_csv(OUT / 'transfer_file_manifest.csv')
    bad = []
    hashes = []
    for row in files.itertuples():
        path = Path(row.destination_path)
        if not path.is_file() or path.stat().st_size != row.bytes:
            bad.append(row.destination_path)
        else:
            hashes.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    if bad:
        raise ValueError(f'Missing or size-mismatched restored files: {bad}')
    targets = pd.read_csv(OLD / 'anatomy_availability.csv')
    points = pd.read_csv(OLD / 'unique_zero_contacts.csv')
    lut = load_lut(Path('/hpc/group/coganlab/nanlinshi/seeg-preprocessing/common/FreeSurferColorLUT.txt'))
    geometry, records, shared_comparisons = [], [], []
    restored_hashes = {r['path']: r['sha256'] for r in hashes}
    for item in targets.itertuples():
        folder = RECON / item.recon_subject
        orig = nib.load(folder / 'mri/orig.mgz')
        brain = nib.load(folder / 'mri/brainmask.mgz')
        aparc = nib.load(folder / 'mri/aparc.a2009s+aseg.mgz')
        good = all(orig.shape == volume.shape and np.allclose(orig.affine, volume.affine, atol=1e-4)
                   for volume in [brain, aparc])
        geometry.append(dict(subject=item.subject, recon_subject=item.recon_subject,
                             shape=str(orig.shape), geometry_matches=good))
        if not good:
            raise ValueError(f'Geometry mismatch after restore: {item.recon_subject}')
        labels = np.asarray(aparc.dataobj)
        inv_tkr = np.linalg.inv(orig.header.get_vox2ras_tkr())
        contacts = read_recon_contact_positions(folder)
        shared_folder = SHARED / item.recon_subject
        shared_contacts = read_recon_contact_positions(shared_folder)
        for name in ['orig.mgz', 'brainmask.mgz', 'aparc.a2009s+aseg.mgz']:
            shared_file = shared_folder / 'mri' / name
            if shared_file.is_file():
                shared_sha = sha(shared_file)
                restored_sha = restored_hashes[str(folder / 'mri' / name)]
                shared_comparisons.append(dict(subject=item.subject,
                    recon_subject=item.recon_subject, filename=name,
                    restored_sha256=restored_sha, shared_sha256=shared_sha,
                    identical=restored_sha == shared_sha, shared_path=str(shared_file)))
        for point in points.loc[points.subject.eq(item.subject)].itertuples():
            xyz = np.array([point.x_native, point.y_native, point.z_native], dtype=float)
            ijk = np.rint(nib.affines.apply_affine(inv_tkr, xyz)).astype(int)
            if not ((ijk >= 0).all() and (ijk < np.asarray(orig.shape)).all()):
                raise ValueError(f'Endpoint outside restored native image: {point.subject}_{point.contact}')
            label = lut.get(int(labels[tuple(ijk)]), str(int(labels[tuple(ijk)])))
            candidates = contacts.get(point.contact, [])
            distance = min([float(np.linalg.norm(xyz - p[0])) for p in candidates], default=np.nan)
            shared_distance = min([float(np.linalg.norm(xyz - p[0]))
                for p in shared_contacts.get(point.contact, [])], default=np.nan)
            records.append(dict(subject=point.subject, contact=point.contact, channels=point.channels,
                recon_subject=item.recon_subject, x_native=xyz[0], y_native=xyz[1], z_native=xyz[2],
                voxel_i=int(ijk[0]), voxel_j=int(ijk[1]), voxel_k=int(ijk[2]),
                old_cached_aparc_label=point.cached_aparc_label,
                restored_native_aparc_label=label, restored_class=classify(label),
                label_matches_cached=label == point.cached_aparc_label,
                recon_contact_coord_delta_mm=distance,
                recon_contact_coord_matches=bool(np.isfinite(distance) and distance <= 1e-3),
                shared_contact_coord_delta_mm=shared_distance,
                shared_contact_coord_matches=bool(np.isfinite(shared_distance) and shared_distance <= 1e-3),
                source=str(folder / 'mri/aparc.a2009s+aseg.mgz')))
        print(f'Checked {item.subject}/{item.recon_subject}', flush=True)
    result = pd.DataFrame(records)
    assert len(result) == 131
    result.to_csv(OUT / 'restored_zero_contact_voxel_checks.csv', index=False)
    pd.DataFrame(geometry).to_csv(OUT / 'restored_geometry_checks.csv', index=False)
    pd.DataFrame(shared_comparisons).to_csv(OUT / 'shared_backup_source_comparison.csv', index=False)
    (OUT / 'restored_file_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        subjects_checked=len(geometry), files_checked=len(hashes), points_checked=len(result),
        all_geometry_matches=all(r['geometry_matches'] for r in geometry),
        all_cached_labels_reproduced=bool(result.label_matches_cached.all()),
        n_contact_coord_matches=int(result.recon_contact_coord_matches.sum()),
        n_restored_or_shared_contact_coord_matches=int((
            result.recon_contact_coord_matches | result.shared_contact_coord_matches).sum()),
        shared_mri_comparisons=shared_comparisons,
        class_counts=result.restored_class.value_counts().to_dict(),
        label_disagreements=result.loc[~result.label_matches_cached].to_dict('records'),
        coordinate_disagreements=result.loc[~result.recon_contact_coord_matches].to_dict('records'),
        model_fitted=False, maper_rerun=False)
    (OUT / 'restored_verification_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
