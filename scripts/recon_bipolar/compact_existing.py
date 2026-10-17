"""Split existing results into 14-column public tables and keyed QC tables.

Preserves CSV values verbatim and verifies lossless reconstruction. No atlas
sampling or pairing is repeated. Execute on Slurm.
"""
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from schema import MAIN_COLUMNS


def read(path):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write(path, columns, rows):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


root = Path(sys.argv[1])
plans = []
files = sorted(root.glob('D*/sub-*_bipolar_parcellation.csv'))
assert len(files) == 8
for path in files:
    columns, rows = read(path)
    assert 'center' in columns and 'midpoint_label' not in columns
    assert len({r['name'] for r in rows}) == len(rows)
    renamed = [{('midpoint_label' if k == 'center' else k): v for k, v in r.items()} for r in rows]
    assert set(MAIN_COLUMNS) <= set(renamed[0])
    audit_columns = ['subject', 'name'] + [c for c in renamed[0] if c not in MAIN_COLUMNS]
    audit = path.parent / 'qc/bipolar_pair_audit.csv'
    assert not audit.exists(), audit
    plans.append((path, audit, renamed, audit_columns, hashlib.sha256(path.read_bytes()).hexdigest()))

report = []
for path, audit, rows, audit_columns, original_hash in plans:
    write(audit, audit_columns, rows)
    write(path, MAIN_COLUMNS, rows)
    cols, public = read(path)
    _, detail = read(audit)
    assert cols == MAIN_COLUMNS and len(public) == len(detail) == len(rows)
    assert [{**p, **a} for p, a in zip(public, detail)] == rows
    report.append(dict(subject=rows[0]['subject'], rows=len(rows),
                       original_sha256=original_hash,
                       public_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest()))

shutil.copyfile(Path(__file__).with_name('README.txt'), root / 'README.txt')
revision = dict(status='passed', created_utc=datetime.now(timezone.utc).isoformat(),
                main_columns=MAIN_COLUMNS, change='center renamed midpoint_label; other fields moved to keyed QC',
                validation='All original cell values recovered exactly from public and QC tables.',
                subjects=report)
(root / 'schema_revision.json').write_text(json.dumps(revision, indent=2) + '\n')
print(json.dumps(dict(status='passed', subjects=len(report), rows=sum(r['rows'] for r in report), columns=len(MAIN_COLUMNS))))
