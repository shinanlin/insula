#!/usr/bin/env python3
"""Inventory four-task coordinates, native anatomy and response evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py
import pandas as pd

from audit_zero_endpoints import ROOTS

BASE = Path('/cwork/ns458/insula-paper')
SOURCE = Path('/hpc/group/coganlab/nanlinshi/insula-functional/results/hga')
OUT = BASE / 'results/audit/four_task_selection_2026-10-17'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, manifest, stats = [], [], []
    for task, root in ROOTS.items():
        for path in sorted((SOURCE / task).glob('sub-*/HGA/*_coord.csv')):
            table = pd.read_csv(path)
            table['task'] = task
            table['source_path'] = str(path)
            frames.append(table)
            manifest.append(dict(path=str(path), bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        files = sorted((root / 'derivatives/statistics').glob(
            'sub-*/bipolar/*desc-Repeat_highgamma.h5'))
        for path in files:
            record = dict(task=task, path=str(path), bytes=path.stat().st_size)
            try:
                with h5py.File(path, 'r') as f:
                    record['keys'] = {k: str(f[k].shape) for k in f if isinstance(f[k],h5py.Dataset)}
                    if 'ch_names' in f:
                        record['ch_names'] = [c.decode() if isinstance(c,bytes) else str(c)
                                              for c in f['ch_names'][:]]
            except Exception as e:
                record['error'] = str(e)
            stats.append(record)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_pickle(OUT / 'all_coordinate_rows.pkl.gz')
    geometry = ['x_native','y_native','z_native'] + [a+str(i)+'_native' for i in [1,2] for a in 'xyz']
    fields = ['task','subject','channel','label','roi','mix','hemi','contact_1','contact_2',
              'contact_1_label','contact_2_label'] + geometry
    variants = all_rows.drop_duplicates(subset=fields).copy()
    variants.to_csv(OUT / 'coordinate_variants.csv', index=False)
    native = []
    for subject in sorted(set(all_rows.subject)):
        number = int(subject[1:].rstrip('AB'))
        for recon in sorted(set([f'D{number}', f'D{number}A', f'D{number}B'])):
            folder = Path('/cwork/ns458/ECoG_Recon') / recon
            if folder.is_dir():
                native.append(dict(subject=subject, recon_subject=recon,
                    **{name: (folder / 'mri' / (name+'.mgz')).is_file()
                       for name in ['orig','brainmask','aparc.a2009s+aseg','aseg','ribbon']},
                    ras_files=[str(p) for p in (folder/'elec_recon').glob('*elec_locations_RAS*.txt')]))
    summary = dict(coordinate_files=len(manifest), coordinate_rows=len(all_rows),
        coordinate_variants=len(variants), unique_channel_names=all_rows.channel.nunique(),
        by_task={t:dict(rows=len(g),channels=g.channel.nunique(),subjects=g.subject.nunique(),
            midpoint_insula_channels=g.loc[g.label.astype(str).str.startswith('insula '),'channel'].nunique(),
            old_roi_mixfalse_channels=g.loc[g.roi.isin(['AIC','PIC']) & ~g.mix.astype(bool),'channel'].nunique())
            for t,g in all_rows.groupby('task')},
        available_stats_files=dict(Counter(s['task'] for s in stats)),
        native_recons=native, figures_generated=False, nmf_rerun=False)
    (OUT/'coordinate_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'statistics_inventory.json').write_text(json.dumps(stats,indent=2)+'\n')
    (OUT/'inventory_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='native_recons'},indent=2),flush=True)


if __name__=='__main__':
    main()
