#!/usr/bin/env python3
"""Locate existing evidence for the 94 unresolved sites, without rewriting inputs."""
from collections import Counter
import json
import os
from pathlib import Path

import pandas as pd

from audit_zero_endpoints import ROOTS

BASE = Path('/cwork/ns458/insula-paper')
OLD = BASE / 'results/audit/four_task_selection_2026-10-17'
OUT = BASE / 'results/audit/response_completeness_2026-10-17'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    task = pd.read_csv(OLD/'task_site_response_audit.csv')
    eligible = task.loc[task.anatomy_status.eq('include')]
    groups = eligible.groupby('channel')
    target = {name for name, rows in groups if not rows.any_significant.any()
              and not rows.all_three_phases_available.all()}
    assert len(target) == 94
    targets = eligible.loc[eligible.channel.isin(target)].copy()
    targets.to_csv(OUT/'original_94_task_rows.csv', index=False)
    subjects = set(targets.subject)
    files = []
    roots = [root/'derivatives' for root in ROOTS.values()]
    roots += [Path('/hpc/group/coganlab/nanlinshi')/name/'results' for name in
              ['insula', 'insula-functional', 'insula-analysis-legacy', 'insula-grant']]
    roots += [Path('/cwork/ns458')/name for name in ['insula_rebuttal','cache']]
    seen = set()
    for root in roots:
        count = 0
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {'.git','node_modules','__pycache__'}]
            for name in names:
                path = Path(directory)/name
                # File names retain BIDS subject, band and condition entities.
                if not any(subject in name or ('sub-'+subject) in directory for subject in subjects):
                    continue
                is_csv = name.endswith(('_time.csv','_zscore.csv','_coord.csv'))
                is_epoch = name.endswith(('.h5','.hdf5','.fif')) and ('highgamma' in name or 'statistics' in directory)
                is_archive = name.endswith(('.tar','.tar.gz','.tgz','.zip'))
                if not (is_csv or is_epoch or is_archive):
                    continue
                absolute = str(path)
                if absolute in seen:
                    continue
                seen.add(absolute)
                stat = path.stat()
                files.append(dict(path=absolute, root=str(root), bytes=stat.st_size,
                                  mtime_ns=stat.st_mtime_ns,
                                  kind='csv' if is_csv else 'epoch_or_stats' if is_epoch else 'archive'))
                count += 1
        print(f'{root}: {count} candidate files', flush=True)
    pd.DataFrame(files).to_csv(OUT/'existing_file_inventory.csv', index=False)
    coordinates = pd.read_pickle(OLD/'all_coordinate_rows.pkl.gz')
    coords = coordinates.loc[coordinates.channel.isin(target)].copy()
    coords.to_csv(OUT/'original_94_coordinate_evidence.csv', index=False)
    summary = dict(target_sites=len(target), subjects=len(subjects),
                   task_rows=len(targets), incomplete_task_rows=int(targets.response_status.eq('incomplete').sum()),
                   by_task=targets.groupby('task').response_status.value_counts().unstack(fill_value=0).to_dict('index'),
                   inventory_files=len(files), file_kinds=dict(Counter(r['kind'] for r in files)))
    (OUT/'inventory_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
