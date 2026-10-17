#!/usr/bin/env python3
"""Temporary SHI-100 native contact screen; no BIDS, bipolar, or neural data.

Run on a Slurm compute node with the ieeg environment. This calls only the
canonical atlas-loading/point-labeling helpers, never parcellation.main().
Missing MAPER products mean UNKNOWN, never zero AIC contacts. A negative
contact-center screen requires image QC before deferring preprocessing and
does not exclude an AIC bipolar midpoint between non-AIC contacts.

Example (MAPER IDs are explicit; D152 variants remain separate):
  python scripts/screen_recon_aic_contacts.py \
    --subjects D150 D152A D152B D154 \
    --maper-id D150=D0150 --maper-id D152A=D0152A \
    --maper-id D152B=D0152B --maper-id D154=D0154 \
    --output /cwork/ns458/insula_recon_screen/run_01

This script does not generate missing MAPER products. A/B may share a MAPER
ID only after the source MRI identity and geometry have been verified.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


UPSTREAM = Path('/hpc/group/coganlab/nanlinshi/seeg-preprocessing/common/parcellation.py')


def load_pipeline(path):
    spec = importlib.util.spec_from_file_location('canonical_contact_parcellation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_contacts(recon):
    """Read native tkRAS mm and validate identity/order against recon exports."""
    folder = recon / 'elec_recon'
    candidates = sorted(folder.glob('*_elec_locations_RAS.txt'))
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one native RAS export: {candidates}')
    path = candidates[0]
    prefix = path.name.removesuffix('_elec_locations_RAS.txt')
    contacts = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 7 or not fields[1].isdigit():
            raise ValueError(f'Malformed coordinate row: {path}:{number}')
        shaft, contact, x, y, z, hemi, kind = fields
        xyz = np.array([x, y, z], dtype=float)
        if not np.isfinite(xyz).all() or hemi not in {'L', 'R'}:
            raise ValueError(f'Invalid coordinates/hemisphere: {path}:{number}')
        contacts.append(dict(contact=shaft + contact, shaft=shaft,
                             contact_number=int(contact), x=xyz[0], y=xyz[1],
                             z=xyz[2], recon_hemi=hemi, contact_type=kind))
    names = [r['contact'] for r in contacts]
    if not names or len(set(names)) != len(names):
        raise ValueError(f'Empty or duplicate contact names: {path}')
    names_path = folder / f'{prefix}.electrodeNames'
    lines = names_path.read_text().splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith('Name,')), None)
    if header is None:
        raise ValueError(f'Cannot find electrodeNames header: {names_path}')
    identities = [line.split() for line in lines[header + 1:] if line.strip()]
    expected = [[r['contact'], r['contact_type'], r['recon_hemi']] for r in contacts]
    if identities != expected:
        raise ValueError('RAS rows disagree with electrodeNames identity/order/type')
    lepto = np.loadtxt(folder / f'{prefix}.LEPTO', skiprows=2, ndmin=2)
    coords = np.array([[r['x'], r['y'], r['z']] for r in contacts])
    if lepto.shape != coords.shape or not np.allclose(lepto, coords, atol=1e-5, rtol=0):
        raise ValueError('RAS export disagrees with native LEPTO coordinates')
    return contacts, path, prefix


def screen_subject(pipeline, subject, maper_id, recon_root, maper_root):
    recon = recon_root / subject
    summary = dict(recon_subject=subject, maper_id=maper_id,
                   status='error', n_contacts=None, n_aic=None, n_pic=None,
                   n_aic_left=None, n_aic_right=None, n_unknown=None,
                   n_invalid=None, aic_contacts='', reason='')
    contacts, coordinate_path, prefix = read_contacts(recon)
    summary.update(n_contacts=len(contacts), coordinate_file=str(coordinate_path),
                   coordinate_sha256=hashlib.sha256(coordinate_path.read_bytes()).hexdigest(),
                   electrode_prefix=prefix)
    for row in contacts:
        row.update(recon_subject=subject, maper_id=maper_id, atlas_label='', roi='',
                   hemi='', status='not_evaluated')
    output = maper_root / maper_id / 'output'
    required = [output / f'f30-seg95-{maper_id}.nii.gz',
                output / f'f30-seg95-{maper_id}-tc3crisp.nii.gz',
                recon / 'mri/orig.mgz']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        summary.update(status='pending_inputs', reason='Missing: ' + '; '.join(missing))
        return summary, contacts
    atlas, geometry, lut, crosswalk, _ = pipeline._resolve_atlas(
        'hammers', str(recon_root), subject, maper_id,
        str(Path(pipeline.__file__).parent), maper_run=str(maper_root))
    for row in contacts:
        xyz = np.array([row['x'], row['y'], row['z']])
        if row['contact_type'] != 'D':
            row['status'] = 'non_depth_requires_coordinate_review'
            continue
        try:
            label = pipeline._label_coordinate(xyz, geometry, atlas, lut)
        except ValueError as exc:
            row.update(status='invalid_coordinate', detail=str(exc))
            continue
        roi, _ = pipeline.consensus_roi([label], crosswalk)
        hemi = pipeline._hemisphere([label], fallback_x=row['x'])
        row.update(atlas_label=label, roi=roi, hemi=hemi, status='labeled')
        if roi in {'AIC', 'PIC'} and hemi != row['recon_hemi']:
            row['status'] = 'hemisphere_conflict'
    aic = [r for r in contacts if r['status'] == 'labeled' and r['roi'] == 'AIC']
    invalid = sum(r['status'] != 'labeled' for r in contacts)
    summary.update(n_aic=len(aic), n_pic=sum(r['status'] == 'labeled' and r['roi'] == 'PIC' for r in contacts),
                   n_aic_left=sum(r['hemi'] == 'L' for r in aic),
                   n_aic_right=sum(r['hemi'] == 'R' for r in aic),
                   n_unknown=sum(r['status'] == 'labeled' and r['roi'] == 'Unknown' for r in contacts),
                   n_invalid=invalid, aic_contacts=';'.join(r['contact'] for r in aic),
                   status=('incomplete_contact_review' if invalid else
                           'aic_detected_needs_qc' if aic else 'no_aic_at_contact_centers_needs_qc'),
                   reason='Contact-center anatomy only; review native slices before deferring preprocessing. '
                          'Task availability and future bipolar midpoint labels are not established.',
                   atlas_inputs=[dict(path=str(p), bytes=p.stat().st_size,
                                      mtime_ns=p.stat().st_mtime_ns) for p in required])
    return summary, contacts


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subjects', nargs='+', required=True, help='Exact recon folders, including A/B suffix')
    parser.add_argument('--maper-id', action='append', default=[], metavar='RECON=MAPER',
                        help='Explicit MAPER ID; default pads recon number, preserving A/B')
    parser.add_argument('--recon-root', type=Path, default=Path('/cwork/ns458/ECoG_Recon'))
    parser.add_argument('--maper-root', type=Path, default=Path('/cwork/ns458/maper_run'))
    parser.add_argument('--upstream', type=Path, default=UPSTREAM)
    parser.add_argument('--output', type=Path, required=True, help='New, separate output directory; never BIDS derivatives')
    args = parser.parse_args()
    if len(set(args.subjects)) != len(args.subjects):
        parser.error('Duplicate subjects')
    ids = {}
    for item in args.maper_id:
        parts = item.split('=')
        if len(parts) != 2 or parts[0] in ids or not re.fullmatch(r'[A-Za-z0-9_-]+', parts[1]):
            parser.error(f'Invalid/duplicate --maper-id: {item}')
        ids[parts[0]] = parts[1]
    if set(ids) - set(args.subjects):
        parser.error('--maper-id specifies a subject outside --subjects')
    for subject in args.subjects:
        match = re.fullmatch(r'D(\d+)([AB]?)', subject)
        if not match:
            parser.error(f'Expected a D-series recon folder: {subject}')
        ids.setdefault(subject, f'D{int(match[1]):04d}{match[2]}')
    pipeline = load_pipeline(args.upstream)
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for subject in args.subjects:
        try:
            summary, contacts = screen_subject(pipeline, subject, ids[subject], args.recon_root, args.maper_root)
            write_csv(args.output / f'{subject}_contacts.csv', contacts)
        except Exception as exc:
            summary = dict(recon_subject=subject, maper_id=ids[subject], status='error',
                           n_aic=None, n_pic=None, reason=f'{type(exc).__name__}: {exc}')
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_csv(args.output / 'summary.csv', [{k: v for k, v in r.items() if k != 'atlas_inputs'} for r in summaries])
    code_files = [args.upstream, Path(__file__), args.upstream.parent / 'hammers.csv',
                  args.upstream.parent / 'Hammers_mith_atlases_n30r95_label_indices_SPM12_20160111.txt']
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                      coordinate_space='FreeSurfer native tkRAS millimeters; RAS verified against LEPTO',
                      method='Hammers seg95, tissue==2 gray matter, exact nearest voxel via vox2ras_tkr',
                      scope='physical depth contacts; no bipolar, task selection, signal preprocessing, or MNI registration',
                      source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
                      subjects=summaries)
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return 2 if any(r['status'] in {'error', 'pending_inputs', 'incomplete_contact_review'} for r in summaries) else 0


if __name__ == '__main__':
    raise SystemExit(main())
