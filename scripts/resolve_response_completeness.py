#!/usr/bin/env python3
"""Recover compatible saved masks; record every unresolved cell and its cause."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle

import h5py
import numpy as np
import pandas as pd

from audit_zero_endpoints import ROOTS
from inventory_response_completeness import BASE, OLD, OUT
from scan_response_backups import WINDOWS, SOURCE, entity, file_key

GROUP=Path('/hpc/group/coganlab/nanlinshi')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extra_inventory():
    roots=[GROUP/name for name in ['insula','insula-functional','insula-analysis-legacy',
        'insula-grant','insula-aicl-delay','nnmf_esm_bipolar_reproducibility_v2',
        'seeg-preprocessing','seeg-preprocessing-worktrees','sharedspace','results','tmp']]
    roots += [Path('/work/ns458'),Path('/cwork/ns458/tmp_scripts')]
    skip={'.git','node_modules','__pycache__','miniconda3','envs','.cache','cache',
          'huggingface_cache','torch_cache','pycortex_cache','pycortex_store','logs'}
    records=[]
    for root in roots:
        for directory,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in skip]
            for name in files:
                if not name.endswith(('.h5','.hdf5','.fif','.npz','.npy','.pkl','.pickle','.pkl.gz',
                                      '.tar','.tar.gz','.tgz','.zip')):
                    continue
                path=Path(directory)/name
                stat=path.stat()
                records.append(dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
        print(f'Additional search: {root}',flush=True)
    pd.DataFrame(records).to_csv(OUT/'additional_array_archive_inventory.csv',index=False)
    return records


def main():
    # Preserve the prior frozen audit as well as the actual figure/NMF sources.
    protected_paths=[*OLD.glob('*.csv'), OLD/'summary.json']
    for root in [BASE,GROUP/'insula-functional']:
        protected_paths += [root/'vizpub/fig1.ipynb']
        protected_paths += list((root/'results/nmf').glob('*.csv'))
        protected_paths += [p for p in [root/'results/nmf/nmf_manifest.json',
                                        root/'results/nmf/exclude_channels.txt'] if p.is_file()]
    protected={str(p):sha(p) for p in protected_paths if p.is_file()}
    extra=extra_inventory()
    task_rows=pd.read_csv(OUT/'original_94_task_rows.csv')
    targets=set(task_rows.channel)
    baseline=pd.read_csv(OLD/'response_evidence.csv')
    baseline=baseline.loc[baseline.channel.isin(targets)].copy()
    candidates=pd.read_csv(OUT/'backup_candidate_evidence.csv')
    versions=pd.read_csv(OUT/'backup_version_comparison.csv')
    file_checks=pd.read_csv(OUT/'backup_file_checks.csv')
    inventory=pd.read_csv(OUT/'existing_file_inventory.csv')
    with (OUT/'backup_scan.pkl').open('rb') as f:scan=pickle.load(f)
    refs={file_key(r[0]['path']):r for r in scan if Path(r[0]['path']).is_relative_to(SOURCE)}
    paths_by_key=defaultdict(list)
    for p in inventory.path:
        if '/bipolar/' in p and entity(p,'desc')=='Repeat' and file_key(p)[2] in WINDOWS:
            if '/statistics/' in p and p.endswith('_highgamma.h5'):
                paths_by_key[file_key(p)].append(p)
    full_stat_records=[]
    full_stat_inventory=[]
    for key,paths in paths_by_key.items():
        for p in paths:
            with h5py.File(p,'r') as f:
                names=[c.decode() if isinstance(c,bytes) else str(c) for c in f['ch_names'][:]]
                mask=f['mask'][:]
                item=dict(path=p,key=list(key),n_channels=len(names),mask_shape=list(mask.shape),
                          target_channels=sorted(set(names)&targets))
                if key not in refs:
                    item['status']='no_validated_time_grid'
                else:
                    grid=np.sort(pd.read_csv(refs[key][0]['path'],usecols=['time']).time.unique())
                    if len(grid)!=mask.shape[1]:
                        item['status']='time_grid_length_mismatch'
                    else:
                        lo,hi=WINDOWS[key[2]]
                        win=(grid>lo)&(grid<hi)
                        item['status']='validated_grid'
                        for i,ch in enumerate(names):
                            if ch in targets and win.any():
                                full_stat_records.append(dict(task=key[0],subject=key[1],phase=key[2],
                                    modality=key[3],channel=ch,significant=bool(mask[i,win].any()),
                                    n_time_samples=int(win.sum()),t_min=float(grid[win].min()),
                                    t_max=float(grid[win].max()),source_path=p,
                                    evidence='existing_full_channel_statistics',evidence_conflict=False))
            full_stat_inventory.append(item)
    (OUT/'target_full_statistics_inventory.json').write_text(json.dumps(full_stat_inventory,indent=2)+'\n')
    accepted=candidates.merge(versions[['path','validation']],left_on='source_path',right_on='path')
    accepted=accepted.loc[accepted.validation.eq('matches_current')].copy()
    accepted['evidence']='version_matched_backup_mask'
    accepted['evidence_conflict']=False
    direct=pd.DataFrame(full_stat_records)
    evidence=pd.concat([baseline,accepted,direct],ignore_index=True)
    evidence.to_csv(OUT/'response_evidence_reviewed.csv',index=False)
    recovered=[]
    cell_rows=[]
    coord=pd.read_csv(OUT/'original_94_coordinate_evidence.csv')
    compatible_paths=set(versions.loc[versions.validation.eq('matches_current'),'path'])
    candidate_groups={k:g for k,g in candidates.groupby(['task','channel','phase','modality'])}
    e_groups={k:g for k,g in evidence.groupby(['task','channel','phase','modality'])}
    expected_by_row={}
    for item in task_rows.itertuples():
        rowkey=(item.task,item.channel)
        old=baseline.loc[baseline.task.eq(item.task)&baseline.channel.eq(item.channel)]
        existing={(r.phase,r.modality) for r in old.itertuples() if not r.evidence_conflict}
        missing={tuple(v.split('/')) for v in str(item.missing_phase_modality_cells).split('|') if '/' in v}
        expected=existing|missing
        expected_by_row[rowkey]=expected
        for phase,modality in sorted(missing):
            key=(item.task,item.channel,phase,modality)
            filekey=(item.task,item.subject,phase,modality)
            found=candidate_groups.get(key,pd.DataFrame())
            new=e_groups.get(key,pd.DataFrame())
            conflict=bool(len(new) and new.significant.nunique()>1)
            if len(new) and not conflict:
                status='recovered'
                for record in new.to_dict('records'):
                    if record['evidence']!='packaged_mask':recovered.append(record)
            elif conflict:
                status='conflicting_compatible_evidence'
            elif len(found):
                status='backup_version_unverified'
            else:
                status='no_saved_mask_for_channel'
            cc=coord.loc[coord.task.eq(item.task)&coord.channel.eq(item.channel)
                &coord.description.eq('Repeat')&coord.phase.str.lower().eq(phase)&coord.modality.eq(modality)]
            stats=paths_by_key.get(filekey,[])
            expected_name=f'sub-{item.subject}_task-{item.task}_proc-{phase.title()}'
            if item.task=='PictureNaming' or modality!='sound':expected_name+=f'_recording-{modality}'
            expected_name+='_desc-Repeat_highgamma.h5'
            cell_rows.append(dict(task=item.task,subject=item.subject,channel=item.channel,phase=phase,
                modality=modality,status=status,current_packaged_file_exists=filekey in refs,
                present_in_repeat_coordinate_export=bool(len(cc)),
                archived_sig_flag=bool(cc.significant.any()) if len(cc) else None,
                original_statistics_present=bool(stats),statistics_paths='|'.join(stats),
                expected_statistics_path=str(ROOTS[item.task]/'derivatives/statistics'/f'sub-{item.subject}'/'bipolar'/expected_name),
                backup_mask_found=bool(len(found)),backup_files='|'.join(sorted(set(found.source_path))) if len(found) else '',
                compatible_backup_found=bool(len(found) and set(found.source_path)&compatible_paths),
                recovered_significant=bool(new.significant.any()) if len(new) and not conflict else None,
                reason='已从版本一致的现有统计补齐' if status=='recovered' else
                    '全通道原统计当前缺失；旧导出含该通道，但不能确认与当前统计版本一致' if len(found) else
                    '全通道原统计当前缺失；所查现存导出也没有该通道的有效 mask'))
    cells=pd.DataFrame(cell_rows)
    cells.to_csv(OUT/'missing_cells_diagnosis.csv',index=False)
    pd.DataFrame(recovered,columns=list(evidence.columns)).to_csv(OUT/'recovered_evidence.csv',index=False)
    updated=[]
    for item in task_rows.itertuples():
        row=item._asdict()
        expected=expected_by_row[(item.task,item.channel)]
        available=set()
        positive=set()
        for phase,modality in expected:
            one=e_groups.get((item.task,item.channel,phase,modality),pd.DataFrame())
            if len(one) and one.significant.nunique()==1:
                available.add((phase,modality))
                if one.significant.any():positive.add(phase)
        for phase in WINDOWS:
            needed={c for c in expected if c[0]==phase}
            row[phase+'_available']=bool(needed) and needed.issubset(available)
            row[phase+'_significant']=phase in positive
        complete=all(row[p+'_available'] for p in WINDOWS)
        row.update(all_three_phases_available=complete,any_significant=bool(positive),
                   response_status='responsive' if positive else 'not_responsive' if complete else 'incomplete',
                   missing_phase_modality_cells='|'.join('/'.join(c) for c in sorted(expected-available)))
        updated.append(row)
    updated=pd.DataFrame(updated)
    updated.to_csv(OUT/'original_94_updated_task_response.csv',index=False)
    sites=[]
    for channel,rows in updated.groupby('channel'):
        one=cells.loc[cells.channel.eq(channel)]
        positive=bool(rows.any_significant.any())
        complete=bool(rows.all_three_phases_available.all())
        sites.append(dict(位点=channel,参与者=rows.subject.iloc[0],任务='|'.join(rows.task),
            原缺失任务数=len(one[['task']].drop_duplicates()),原缺失阶段模态数=len(one),
            原统计文件当前存在数=int(one.original_statistics_present.sum()),
            旧版导出可找到mask的缺失单元数=int(one.backup_mask_found.sum()),
            本次已补齐单元数=int(one.status.eq('recovered').sum()),
            尚缺单元数=int((~one.status.eq('recovered')).sum()),
            补齐后结论='显著响应' if positive else '完整评估且未显著' if complete else '证据不完整，不能判为无响应',
            缺失详情=' | '.join(r.task+':'+r.missing_phase_modality_cells for r in rows.itertuples() if r.missing_phase_modality_cells),
            原因='存在版本未核实的历史 mask' if one.backup_mask_found.any() else '未找到该通道的现存 mask'))
    site_df=pd.DataFrame(sites)
    site_df.to_csv(OUT/'94个位点逐项核对.csv',index=False,encoding='utf-8-sig')
    source_inputs={r.path:dict(bytes=r.bytes,mtime_ns=r.mtime_ns,sha256=r.sha256) for r in file_checks.itertuples()}
    unchanged=[]
    for p,info in source_inputs.items():
        stat=Path(p).stat()
        if stat.st_size!=info['bytes'] or stat.st_mtime_ns!=info['mtime_ns']:
            unchanged.append(p)
    protected_changed=[p for p,h in protected.items() if sha(p)!=h]
    # Confirm recovery has not replaced any of the prior positive/negative evidence.
    assert len(site_df)==94 and len(task_rows)==149
    assert not protected_changed and not unchanged
    summary=dict(original_sites=94,original_incomplete_task_rows=129,
        original_missing_cells=len(cells),current_original_statistics_cells=int(cells.original_statistics_present.sum()),
        missing_cells_in_existing_coordinate_exports=int(cells.present_in_repeat_coordinate_export.sum()),
        missing_cells_with_archived_sig_flag=int(cells.archived_sig_flag.eq(True).sum()),
        cells_with_backup_masks=int(cells.backup_mask_found.sum()),
        sites_with_backup_masks=int(cells.loc[cells.backup_mask_found,'channel'].nunique()),
        compatible_recovered_cells=int(cells.status.eq('recovered').sum()),
        after_site_status=site_df.补齐后结论.value_counts().to_dict(),
        diagnosis_by_cell=cells.status.value_counts().to_dict(),
        by_task=cells.groupby('task').agg(missing_cells=('channel','size'),
            sites=('channel','nunique'),backup_cells=('backup_mask_found','sum'),
            stats_present=('original_statistics_present','sum')).to_dict('index'),
        scanned_csv_files=len(file_checks),csv_scan_errors=int(file_checks.get('error',pd.Series(dtype=str)).notna().sum()),
        additional_array_files=len(extra),protected_files=len(protected),protected_changed=protected_changed,
        source_csv_files_unchanged=len(source_inputs),source_metadata_changed=unchanged,
        figures_regenerated=False,nmf_rerun=False,statistics_rerun=False)
    (OUT/'protected_files.json').write_text(json.dumps(protected,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    with pd.ExcelWriter(OUT/'94个位点响应记录核对.xlsx',engine='openpyxl') as writer:
        site_df.to_excel(writer,sheet_name='94个位点',index=False)
        cells.to_excel(writer,sheet_name='缺失阶段模态及原因',index=False)
        updated.to_excel(writer,sheet_name='149条任务记录',index=False)
        versions.to_excel(writer,sheet_name='备份版本核对',index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes='A2'
            sheet.auto_filter.ref=sheet.dimensions
            for col in sheet.columns:
                letter=col[0].column_letter
                sheet.column_dimensions[letter].width=min(65,max(14,len(str(col[0].value))+3))
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
