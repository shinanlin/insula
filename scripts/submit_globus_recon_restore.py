#!/usr/bin/env python3
"""Review or submit the verified Windows-to-cwork anatomy restoration.

The explicit file list is fixed by inventory, with checksum verification and
no destination deletions. A saved submission ID makes retries idempotent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
from globus_cli.login_manager import LoginManager
from globus_sdk import MISSING, TransferData

OUT = Path('/cwork/ns458/insula-paper/results/audit/globus_recon_restore_2026-10-17')


def json_ready(value):
    """Mirror the SDK's omission of optional MISSING fields for the review file."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items() if item is not MISSING}
    if isinstance(value, list):
        return [json_ready(item) for item in value if item is not MISSING]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    meta = json.loads((OUT / 'inventory_summary.json').read_text())
    files = pd.read_csv(OUT / 'transfer_file_manifest.csv')
    assert meta['n_subjects'] == meta['n_required_mri_complete'] == 54
    assert meta['n_listing_errors'] == 0
    assert len(files) == meta['n_files'] and not files.destination_path.duplicated().any()
    assert all(p.startswith(meta['source_root'] + '/') for p in files.source_path)
    assert all(p.startswith(meta['destination_root'] + '/') for p in files.destination_path)
    assert not files.destination_already_exists.any(), 'Review existing destination files before copying'
    transfer = TransferData(meta['source_collection'], meta['destination_collection'],
        label='Insula native MRI restore 54 participants 2026-10-17',
        sync_level='checksum', verify_checksum=True, preserve_timestamp=False,
        encrypt_data=True, skip_source_errors=False, fail_on_quota_errors=True,
        delete_destination_extra=False,
        notify_on_succeeded=False, notify_on_failed=False, notify_on_inactive=False)
    files['priority'] = files.subject.ne('D0125').astype(int)
    for r in files.sort_values(['priority','subject','folder','name']).itertuples():
        transfer.add_item(r.source_path, r.destination_path, recursive=False)
    payload = json.dumps(json_ready(transfer), indent=2)
    payload_sha = hashlib.sha256(payload.encode()).hexdigest()
    (OUT / 'reviewed_transfer_payload.json').write_text(payload + '\n')
    print(json.dumps(dict(subjects=54, files=len(files), bytes=meta['total_bytes'],
        source_root=meta['source_root'], destination_root=meta['destination_root'],
        verify_checksum=True, delete_destination_extra=False,
        encrypt_data=True, preserve_timestamp=False, submit=args.submit), indent=2), flush=True)
    if not args.submit:
        return
    manager = LoginManager()
    client = manager.get_transfer_client()
    # Collection policy requires a fresh Duke identity in this CLI session.
    # An authentication error here stops before any transfer is submitted.
    client.operation_ls(meta['destination_collection'], path=meta['destination_root'] + '/')
    receipt_path = OUT / 'transfer_receipt.json'
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if prior['reviewed_payload_sha256'] != payload_sha:
            raise ValueError('Existing transfer receipt belongs to a different payload')
        print(json.dumps(client.get_task(prior['task_id']).data, indent=2), flush=True)
        return
    submission_path = OUT / 'submission_id.json'
    if submission_path.exists():
        sub = json.loads(submission_path.read_text())
        assert sub['reviewed_payload_sha256'] == payload_sha
    else:
        sub = dict(submission_id=client.get_submission_id()['value'],
                   reviewed_payload_sha256=payload_sha)
        submission_path.write_text(json.dumps(sub, indent=2) + '\n')
    transfer['submission_id'] = sub['submission_id']
    result = client.submit_transfer(transfer)
    receipt = dict(**result.data, reviewed_payload_sha256=payload_sha,
        submitted_utc=datetime.now(timezone.utc).isoformat(),
        source_collection=meta['source_collection'], destination_collection=meta['destination_collection'],
        n_files=len(files), total_bytes=meta['total_bytes'], model_fitted=False)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
