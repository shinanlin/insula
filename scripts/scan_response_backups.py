#!/usr/bin/env python3
"""Read existing CSV masks and compare backup versions against current exports."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd

from inventory_response_completeness import BASE, OLD, OUT

WINDOWS = {'stimulus': (0., .6), 'delay': (0., 1.), 'response': (-.5, .5)}
SOURCE = Path('/hpc/group/coganlab/nanlinshi/insula-functional/results/hga')


def entity(path, label):
    match = re.search(r'(?:^|_)'+label+r'-([^_]+)', Path(path).name)
    return match[1] if match else ''


def file_key(path):
    task = entity(path, 'task').replace('PhonemeSequencing', 'PhonemeSequence')
    phase = entity(path, 'proc').lower()
    phase = {'audio':'stimulus','resp':'response'}.get(phase,phase)
    return task, entity(path,'sub'), phase, entity(path,'recording') or 'sound'


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def scan(path, targets):
    key = file_key(path)
    record = dict(path=str(path), task=key[0], subject=key[1], phase=key[2], modality=key[3],
                  bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    try:
        frame = pd.read_csv(path, usecols=lambda c:c in {'time','channel','mask','value','description','phase','modality'})
        if not {'time','channel','mask','value'}.issubset(frame):
            record['error']='missing_required_columns'
            return record, {}, []
        frame['time'] = pd.to_numeric(frame.time, errors='raise')
        signatures, evidence = {}, []
        for channel, rows in frame.groupby('channel',sort=False):
            rows = rows.sort_values('time')
            times = rows.time.to_numpy(dtype='float64')
            mask = rows['mask'].eq(True).to_numpy()
            values = rows.value.to_numpy(dtype='float64')
            if len(np.unique(times)) != len(times):
                raise ValueError('duplicate_channel_time_rows')
            signatures[channel] = dict(n=len(times), time=digest(np.round(times,9)),
                                       mask=digest(mask),value=digest(np.round(values,7)))
            if channel in targets:
                lo,hi = WINDOWS[key[2]]
                window = (times>lo)&(times<hi)
                if window.any():
                    evidence.append(dict(task=key[0], subject=key[1], phase=key[2], modality=key[3],
                        channel=channel, significant=bool(mask[window].any()),
                        n_time_samples=int(window.sum()), t_min=float(times[window].min()),
                        t_max=float(times[window].max()), mask_hash=signatures[channel]['mask'],
                        time_hash=signatures[channel]['time'], source_path=str(path)))
        record.update(channels=len(signatures), target_channels=len(evidence),
                      mask_true_samples=int(frame['mask'].eq(True).sum()))
        return record, signatures, evidence
    except Exception as exc:
        record['error']=str(exc)
        return record, {}, []


def main():
    inventory = pd.read_csv(OUT/'existing_file_inventory.csv')
    target_rows = pd.read_csv(OUT/'original_94_task_rows.csv')
    targets = set(target_rows.channel)
    keys = {(r.task,r.subject) for r in target_rows.itertuples()}
    paths=[]
    for row in inventory.itertuples():
        path=Path(row.path)
        key=file_key(path)
        if row.kind!='csv' or not path.name.endswith(('_time.csv','_zscore.csv')):
            continue
        if entity(path,'desc')!='Repeat' or key[2] not in WINDOWS or key[:2] not in keys:
            continue
        # Different sentence epochs and CAR are not interchangeable with bipolar task epochs.
        if '(sentence)' in str(path) or '(car)' in str(path) or '/car/' in str(path):
            continue
        paths.append(path)
    print(f'Scanning {len(paths)} existing Repeat CSVs',flush=True)
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(lambda p:scan(p,targets),paths),1):
            results.append(result)
            if i%100==0 or i==len(paths):print(f'CSV {i}/{len(paths)}',flush=True)
    with (OUT/'backup_scan.pkl').open('wb') as f:pickle.dump(results,f)
    pd.DataFrame(r[0] for r in results).to_csv(OUT/'backup_file_checks.csv',index=False)
    pd.DataFrame(e for r in results for e in r[2]).to_csv(OUT/'backup_candidate_evidence.csv',index=False)
    current = {file_key(r[0]['path']):r for r in results if Path(r[0]['path']).is_relative_to(SOURCE)}
    comparisons=[]
    for meta,sigs,evidence in results:
        key=file_key(meta['path'])
        if key not in current:
            comparisons.append(dict(path=meta['path'], validation='no_current_reference', shared_channels=0))
            continue
        ref,rsigs,_=current[key]
        shared=set(sigs)&set(rsigs)
        nt=sum(sigs[c]['time']==rsigs[c]['time'] for c in shared)
        nm=sum(sigs[c]['time']==rsigs[c]['time'] and sigs[c]['mask']==rsigs[c]['mask'] for c in shared)
        nv=sum(sigs[c]['time']==rsigs[c]['time'] and sigs[c]['value']==rsigs[c]['value'] for c in shared)
        is_current=meta['path']==ref['path']
        valid=is_current or (len(shared)>=3 and nt==nm==nv==len(shared))
        comparisons.append(dict(path=meta['path'],reference_path=ref['path'],shared_channels=len(shared),
            same_time_channels=nt,same_mask_channels=nm,same_value_channels=nv,
            validation='current_reference' if is_current else 'matches_current' if valid else 'different_or_insufficient_overlap'))
    comparison=pd.DataFrame(comparisons)
    comparison.to_csv(OUT/'backup_version_comparison.csv',index=False)
    print(json.dumps(dict(files=len(results),validation=Counter(comparison.validation),
                         errors=[r[0] for r in results if r[0].get('error')]),indent=2),flush=True)


if __name__=='__main__':main()
