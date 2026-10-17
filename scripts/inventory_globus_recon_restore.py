#!/usr/bin/env python3
"""Read Windows Globus recon metadata and prepare an explicit restore batch.

Uses the existing Globus CLI login without printing credentials. This script
does not submit transfers, modify the Windows source, or start model jobs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import threading
import time

import pandas as pd
from globus_cli.login_manager import LoginManager

BASE = Path('/cwork/ns458/insula-paper')
OUT = BASE / 'results/audit/globus_recon_restore_2026-10-17'
TARGETS = BASE / 'docs/evidence/zero_endpoints_2026-10-17/anatomy_availability.csv'
SOURCE_COLLECTION = 'bfd3e08e-22c9-11f0-9e52-0ed8719d94b5'
DEST_COLLECTION = '1ad66c7c-4f60-11e8-900c-0a6d4e044368'
SOURCE_ROOT = '/C/Users/ns458/Box/CoganLab/ECoG_Recon_Full'
DEST_ROOT = '/cwork/ns458/ECoG_Recon'
MRI_FILES = {'orig.mgz', 'brainmask.mgz', 'aparc.a2009s+aseg.mgz',
             'aparc+aseg.mgz', 'aseg.mgz', 'wm.mgz', 'ribbon.mgz'}
REQUIRED_MRI = {'orig.mgz', 'brainmask.mgz', 'aparc.a2009s+aseg.mgz'}
tls = threading.local()


def client():
    if not hasattr(tls, 'client'):
        tls.manager = LoginManager()
        tls.client = tls.manager.get_transfer_client()
    return tls.client


def read_subject(item):
    subject, recon = item
    if not re.fullmatch(r'D\d+[AB]?', recon):
        raise ValueError(f'Unexpected reconstruction name: {recon!r}')
    records, listings, errors = [], [], []
    for folder in ['mri', 'elec_recon']:
        remote = f'{SOURCE_ROOT}/{recon}/{folder}/'
        try:
            # Box Drive sometimes needs a first stat request to hydrate a
            # directory. Retry transient listing errors before declaring it
            # unavailable; no Windows filesystem changes are requested here.
            for attempt in range(3):
                try:
                    response = client().operation_ls(SOURCE_COLLECTION, path=remote)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            data = response.data
            listings.append(dict(source_path=remote, response=data))
            for f in data.get('DATA', []):
                name = f['name']
                if f.get('type') != 'file':
                    continue
                keep = name in MRI_FILES if folder == 'mri' else (
                    'electrodeNames' in name or 'elec_locations_RAS' in name or
                    name.endswith(('.PIAL', '.PIALVOX', '.LEPTO', '.LEPTOVOX', '.POSTIMPLANT')))
                if not keep:
                    continue
                records.append(dict(subject=subject, recon_subject=recon,
                    source_path=remote + name,
                    destination_path=f'{DEST_ROOT}/{recon}/{folder}/{name}',
                    folder=folder, name=name, bytes=f.get('size'),
                    source_last_modified=f.get('last_modified'),
                    destination_already_exists=Path(f'{DEST_ROOT}/{recon}/{folder}/{name}').exists()))
        except Exception as exc:
            errors.append(dict(subject=subject, recon_subject=recon, source_path=remote,
                               error_type=type(exc).__name__, message=str(exc)))
    mri = {r['name'] for r in records if r['folder'] == 'mri'}
    row = dict(subject=subject, recon_subject=recon,
               required_mri_complete=REQUIRED_MRI <= mri,
               missing_required_mri='|'.join(sorted(REQUIRED_MRI - mri)),
               contact_names_found=any('electrodeNames' in r['name'] for r in records),
               contact_RAS_found=any('elec_locations_RAS' in r['name'] for r in records),
               n_files=len(records), bytes=sum(r['bytes'] or 0 for r in records),
               n_listing_errors=len(errors))
    return row, records, listings, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-failures', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(TARGETS)
    items = sorted(set(zip(targets.subject, targets.recon_subject)))
    summaries, records, listings, errors = [], [], [], []
    if args.retry_failures:
        old_subjects = pd.read_csv(OUT / 'windows_subject_inventory.csv')
        retry_subjects = set(old_subjects.loc[old_subjects.n_listing_errors.gt(0), 'subject'])
        items = [item for item in items if item[0] in retry_subjects]
        summaries = old_subjects.loc[~old_subjects.subject.isin(retry_subjects)].to_dict('records')
        old_files = pd.read_csv(OUT / 'transfer_file_manifest.csv')
        records = old_files.loc[~old_files.subject.isin(retry_subjects)].to_dict('records')
        listings = json.loads((OUT / 'source_listings.json').read_text())
        if not (OUT / 'listing_errors_initial.json').exists():
            (OUT / 'listing_errors_initial.json').write_text((OUT / 'listing_errors.json').read_text())
        print(f'Retry transient source-listing failures for {len(items)} subjects', flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for summary, files, dirs, issues in executor.map(read_subject, items):
            summaries.append(summary); records.extend(files); listings.extend(dirs); errors.extend(issues)
            print(f"{summary['subject']}/{summary['recon_subject']}: "
                  f"{summary['n_files']} files, required MRI {summary['required_mri_complete']}", flush=True)
    table = pd.DataFrame(summaries)
    files = pd.DataFrame(records)
    table.to_csv(OUT / 'windows_subject_inventory.csv', index=False)
    files.to_csv(OUT / 'transfer_file_manifest.csv', index=False)
    # Highest-priority boundary case is listed first in the transfer batch.
    files['priority'] = files.subject.ne('D0125').astype(int)
    files = files.sort_values(['priority','subject','folder','name'])
    batch = '\n'.join(shlex.quote(r.source_path) + ' ' + shlex.quote(r.destination_path)
                      for r in files.itertuples()) + '\n'
    (OUT / 'restore_files.batch').write_text(batch)
    meta = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                source_collection=SOURCE_COLLECTION, source_root=SOURCE_ROOT,
                destination_collection=DEST_COLLECTION, destination_root=DEST_ROOT,
                n_subjects=len(table), n_required_mri_complete=int(table.required_mri_complete.sum()),
                n_files=len(records), total_bytes=int(table.bytes.sum()),
                n_listing_errors=len(errors), model_fitted=False, transfer_submitted=False,
                target_manifest_sha256=hashlib.sha256(TARGETS.read_bytes()).hexdigest(),
                transfer_batch_sha256=hashlib.sha256(batch.encode()).hexdigest())
    (OUT / 'inventory_summary.json').write_text(json.dumps(meta, indent=2) + '\n')
    (OUT / 'source_listings.json').write_text(json.dumps(listings, indent=2) + '\n')
    (OUT / 'listing_errors.json').write_text(json.dumps(errors, indent=2) + '\n')
    print(json.dumps(meta, indent=2), flush=True)
