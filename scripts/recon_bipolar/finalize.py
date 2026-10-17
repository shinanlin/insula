"""Publish only final subject tables using the project's BIDS-style layout.

After byte-for-byte verification, delete the explicitly supplied intermediate
run directory. Never pass a recon, MAPER, or task BIDS directory as --staging.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from schema import MAIN_COLUMNS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New derivatives/parcellation directory')
    args = parser.parse_args()
    staging, output = args.staging.resolve(), args.output.resolve()
    allowed = Path('/cwork/ns458/insula_recon_screen').resolve()
    if staging.parent != allowed or not staging.name.startswith('shi100_'):
        raise ValueError('Only an immediate shi100_* staging run under insula_recon_screen may be removed')
    if output.exists() or staging == output or staging in output.parents:
        raise ValueError('Output must be a new directory outside staging')
    validation = json.loads((staging / 'validation.json').read_text())
    if validation['status'] != 'passed':
        raise ValueError('Anatomical validation did not pass')
    plans = []
    total = 0
    for number in (148, 149, 150, 152, 153, 154, 157, 158):
        subject = f'D{number:04d}'
        src = staging / subject / f'sub-{subject}_bipolar_parcellation.csv'
        with src.open(newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != MAIN_COLUMNS:
                raise ValueError(f'Unexpected schema: {src}')
            rows = list(reader)
        if not rows or any(r['subject'] != subject for r in rows):
            raise ValueError(f'Empty table or subject mismatch: {src}')
        total += len(rows)
        dest = output / f'sub-{subject}' / 'bipolar' / f'sub-{subject}_aparc2009sHammers.csv'
        plans.append((src, dest, digest(src)))
    if total != validation['bipolar_pairs']:
        raise ValueError('Pair count differs from validated result')
    for src, dest, _ in plans:
        dest.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(src, dest)
    for src, dest, expected in plans:
        if digest(src) != expected or digest(dest) != expected:
            raise ValueError('File changed during publication; staging retained')
    shutil.rmtree(staging)
    print(json.dumps(dict(status='passed', subjects=8, rows=total,
                          output=str(output), removed_staging=str(staging),
                          verification='All final CSV SHA256 hashes match source tables exactly.')))


if __name__ == '__main__':
    main()
