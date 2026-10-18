#!/usr/bin/env python3
"""Verify non-CSV backups and freeze the response-completeness audit."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import h5py
import mne
import pandas as pd

from audit_zero_endpoints import ROOTS
from inventory_response_completeness import BASE, OUT
from scan_response_backups import entity, file_key, WINDOWS


def main():
    targets=pd.read_csv(OUT/'original_94_task_rows.csv')
    subjects=set(targets.subject)
    inventory=pd.read_csv(OUT/'existing_file_inventory.csv')
    extra=pd.read_csv(OUT/'additional_array_archive_inventory.csv')
    files=sorted(set(inventory.loc[inventory.kind.eq('epoch_or_stats'),'path'])
                 |set(extra.loc[extra.path.str.endswith(('.h5','.hdf5')),'path']))
    h5_records=[]
    cached_path=OUT/'other_h5_evidence_checked.csv'
    cached=pd.read_csv(cached_path).to_dict('records') if cached_path.is_file() else None
    for path in ([] if cached else files):
        p=Path(path)
        if not any(subject in p.name or ('sub-'+subject) in path for subject in subjects):continue
        if entity(p,'desc')!='Repeat' or file_key(p)[2] not in WINDOWS:continue
        if not any(task in path or task=='PhonemeSequence' and 'PhonemeSequencing' in path for task in ROOTS):continue
        record=dict(path=path,bytes=p.stat().st_size)
        try:
            with h5py.File(p,'r') as f:
                record['root_keys']='|'.join(f.keys())
                datasets={k:list(f[k].shape) for k in f if isinstance(f[k],h5py.Dataset)}
                record['root_dataset_shapes']=json.dumps(datasets)
                record['evoked_mask_schema']=all(k in f for k in ['mask','pvals','ch_names','sig_ch_names'])
                if record['evoked_mask_schema']:
                    record['classification']='original_band_statistics'
                elif '/RT/' in path:
                    record['classification']='reaction_time_model_output'
                elif '/epoch(' in path:
                    record['classification']='saved_epochs_without_original_statistics'
                else:
                    record['classification']='other_downstream_output'
        except Exception as exc:
            record['classification']='unreadable'
            record['error']=str(exc)
        h5_records.append(record)
    if cached:
        h5_records=cached
    # MNE writes FIF epochs even when these historical paths end in .h5.
    # Inspect their headers with the producer's reader rather than calling them corrupt HDF5.
    for record in h5_records:
        if record['classification']=='unreadable' and '/epoch(' in record['path']:
            epochs=mne.read_epochs(record['path'],preload=False,verbose='error')
            record.update(classification='mne_fif_epochs_without_response_statistics',error='',
                          mne_channels=len(epochs.ch_names),mne_time_samples=len(epochs.times))
    h5=pd.DataFrame(h5_records)
    h5.to_csv(OUT/'other_h5_evidence_checked.csv',index=False)
    archives=[]
    for p in extra.loc[extra.path.str.endswith('.zip'),'path']:
        with zipfile.ZipFile(p) as f:
            names=f.namelist()
            archives.append(dict(path=p,member_count=len(names),first_five_members=names[:5],
                                 possible_statistics_members=[n for n in names if n.endswith(('.h5','.hdf5','.npz'))]))
    (OUT/'archive_contents_checked.json').write_text(json.dumps(archives,indent=2)+'\n')
    availability=[]
    for task,subject in sorted(set(zip(targets.task,targets.subject))):
        root=ROOTS[task]/'derivatives'
        epoch=root/'epoch(bipolar)'/f'sub-{subject}'
        power=list((epoch/'epoch(band)(power)').glob('*highgamma.h5'))
        repeat=[p for p in power if entity(p,'desc')=='Repeat']
        availability.append(dict(task=task,subject=subject,
            full_power_epoch_files_all_conditions=len(power),
            repeat_power_phase_files=len([p for p in repeat if file_key(p)[2] in WINDOWS]),
            repeat_power_baseline_files=len([p for p in repeat if entity(p,'proc')=='baseline']),
            full_zscore_epoch_files=len(list((epoch/'epoch(band)(zscore)').glob('*highgamma.h5'))),
            sig_epoch_files=len(list((epoch/'epoch(band)(sig)').glob('*highgamma.h5'))),
            full_stat_files=len(list((root/'statistics'/f'sub-{subject}'/'bipolar').glob('*desc-Repeat_highgamma.h5')))))
    available=pd.DataFrame(availability)
    available.to_csv(OUT/'upstream_data_availability.csv',index=False)
    versions=pd.read_csv(OUT/'backup_version_comparison.csv')
    candidates=pd.read_csv(OUT/'backup_candidate_evidence.csv')
    merged=candidates.merge(versions,left_on='source_path',right_on='path')
    shared=merged.shared_channels.fillna(0)
    merged['version_reason']='insufficient_common_channels'
    merged.loc[(shared>0)&(merged.same_time_channels<shared),'version_reason']='time_axis_differs'
    merged.loc[(shared>0)&(merged.same_time_channels.eq(shared))&(merged.same_mask_channels<shared),
               'version_reason']='significance_mask_differs'
    merged.loc[(shared>0)&(merged.same_time_channels.eq(shared))&(merged.same_mask_channels.eq(shared))
               &(merged.same_value_channels<shared),'version_reason']='waveform_differs'
    merged.loc[merged.validation.isin(['matches_current','current_reference']),'version_reason']='matches_current'
    merged.to_csv(OUT/'backup_candidates_with_version_reasons.csv',index=False)
    cells=pd.read_csv(OUT/'missing_cells_diagnosis.csv')
    restore=cells.groupby(['task','subject','phase','modality','expected_statistics_path'],as_index=False).agg(
        missing_channels=('channel',lambda x:'|'.join(sorted(set(x)))),
        missing_channel_count=('channel','nunique'))
    restore.to_csv(OUT/'statistics_restore_manifest.csv',index=False)
    sites=pd.read_csv(OUT/'94个位点逐项核对.csv')
    summary=json.loads((OUT/'summary.json').read_text())
    checks=dict(original_sites=len(sites),original_task_rows=len(targets),
                original_missing_cells=len(cells),current_statistics_found_for_missing_cells=int(cells.original_statistics_present.sum()),
                h5_files_checked=len(h5),h5_classes=h5.classification.value_counts().to_dict(),
                archives_checked=len(archives),
                missing_original_statistics_files=len(restore),
                backup_candidates_by_version_reason=merged.version_reason.value_counts().to_dict(),
                protected_changed=[p for p,h in json.loads((OUT/'protected_files.json').read_text()).items()
                                   if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h])
    assert len(sites)==94 and len(targets)==149 and len(cells)==597
    assert not cells.original_statistics_present.any()
    assert not cells.status.eq('recovered').any()
    assert not checks['protected_changed']
    assert sites.补齐后结论.eq('证据不完整，不能判为无响应').all()
    # All original reference rows and every matching alternative export were searched.
    assert not set(merged.loc[merged.validation.eq('matches_current'),
                          ['task','channel','phase','modality']].itertuples(index=False,name=None)) & set(
        cells[['task','channel','phase','modality']].itertuples(index=False,name=None))
    checks['passed']=True
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n')
    dest=BASE/'docs/evidence/response_completeness_2026-10-17'
    dest.mkdir(parents=True,exist_ok=True)
    frozen=['94个位点逐项核对.csv','94个位点响应记录核对.xlsx','missing_cells_diagnosis.csv',
            'original_94_task_rows.csv','original_94_updated_task_response.csv','summary.json','validation.json',
            'backup_version_comparison.csv','backup_file_checks.csv','backup_candidates_with_version_reasons.csv',
            'other_h5_evidence_checked.csv','archive_contents_checked.json','upstream_data_availability.csv',
            'protected_files.json','recovered_evidence.csv','response_evidence_reviewed.csv']
    frozen+=['statistics_restore_manifest.csv']
    for name in frozen:shutil.copy2(OUT/name,dest/name)
    report=f'''# 94 个位点的响应记录缺失核对

日期：2026-10-17。范围：仅核对上一轮解剖合格、尚无显著响应证据的 94 个位点。Figure 1、NMF、原始 HGA 打包文件和上一轮审核均保持原状。

## 结论

这 94 个位点涉及 38 位参与者、149 条任务×位点记录。其中 129 条任务×位点记录不完整，具体缺失 **597 个通道×阶段×模态单元**，对应 **{len(restore)} 份去重后的原统计文件**。

缺失有两个同时存在的环节：

1. **打包时未纳入这些通道。** 597 个缺失单元均能在同一任务、同一 Repeat 阶段／模态的全通道坐标表中找到；其历史 `significant` 标记均为 False。当前时间序列打包使用参与者×模态的显著通道并集，而不是全部采样通道。因此这些位点有采样／坐标记录，时间序列导出却没有对应通道。历史打包日志也记录了构建显著通道并集的步骤。
2. **现在无法从原路径取回对应全通道统计。** 597 个缺失单元对应的 Repeat/bipolar/highgamma 原统计文件当前均不存在。不能据此说统计从未计算过；已有导出和打包日志表明历史上做过统计和筛选。此次只能确认原文件目前不可用，无法仅凭当前文件确定删除时间或原因。

两者并非互斥：通道先未进入筛选后的导出，后来要回查它的完整 mask 时，原统计文件又已不在现存路径。

## 从现有数据追回了什么

检查了 1,572 个相关 Repeat 时间序列／zscore CSV，并搜索了项目、旧工作区、临时备份和 work 中的数组／归档文件。

- **60 个位点**的部分缺失单元在旧导出中还能找到 mask，共覆盖 **351/597 个缺失单元**。
- 其余 **34 个位点**未找到缺失单元的历史 mask；按单元计，未找到备份的是 **246/597**。60 个找到旧记录的位点也可能仍有其他单元无备份，因此不能把 60 理解为已完整恢复。
- 找到的旧记录与当前结果的共有通道在时间轴、逐时刻显著性 mask 或波形上存在差异，或缺少足够共有通道供版本核对。**没有一条缺失单元通过当前版本核对，所以本次正式补回 0 个；94 个仍保留为证据不完整。** 旧记录及来源已整理保存，没有并入统一响应名单。
- 有 530 份其他导出文件通过了与当前导出的共有通道核对，但它们同样不包含这 597 个缺失单元，不能提供额外阴性证据。

例：D0022 的 PhonemeSequence／Stimulus 旧全通道 zscore 导出与当前导出共有 3 个通道，时间轴均相同；逐时刻 mask 仅 1/3 相同，波形为 0/3 相同。该旧文件可以用于回查历史结果，不能直接补入当前口径。

## 分任务的缺失范围

| 任务 | 存在缺失的位点数 | 缺失阶段×模态单元 | 旧版 mask 可找到的单元 | 原统计当前可用的缺失单元 |
|---|---:|---:|---:|---:|
| PhonemeSequence | 39 | 117 | 68 | 0 |
| LexicalDelay | 46 | 138 | 54 | 0 |
| PictureNaming | 44 | 342 | 229 | 0 |
| SentenceRep | 0 | 0 | 0 | 0 |

任务间同一物理位点会重复出现，因此位点数不能相加。94 个位点中有 20 条任务记录本来就已完整且未显著，均保留；缺失发生在另外 129 条任务记录。

## 为什么不能直接把缺失填为“不显著”

`significant=False` 和“没有进入显著通道并集”是历史筛选名单的证据，不能替代当前三个时间窗内逐时刻 mask 的完整阴性证据。例如现存上游配置以 Stimulus／Delay 0–0.5 s 内 `mask & (pvals < 0.05)` 选通道，而当前审核按保存的 mask 在 Stimulus 0–0.6 s、Delay 0–1 s、Response −0.5–0.5 s 取并集。名单定义与当前审核窗口／判据并不等价。

此次没有改变显著性判据，没有重做置换统计，也没有将失踪 mask 默认为 False。不能从旧版本的平均波形重建原来的 trial-level 置换检验。

## 当前可用的表和下一步所需输入

- [94 个位点核对工作簿](evidence/response_completeness_2026-10-17/94个位点响应记录核对.xlsx)：逐位点、597 个缺失单元及原因、149 条任务记录、旧导出版本比较。
- [逐位点简表](evidence/response_completeness_2026-10-17/94个位点逐项核对.csv)
- [每个缺失单元、预期原文件路径及旧备份路径](evidence/response_completeness_2026-10-17/missing_cells_diagnosis.csv)
- [备份版本核对](evidence/response_completeness_2026-10-17/backup_version_comparison.csv)
- [上游 epoch／统计可用性](evidence/response_completeness_2026-10-17/upstream_data_availability.csv)
- [去重后的原统计恢复清单](evidence/response_completeness_2026-10-17/statistics_restore_manifest.csv)
- [其他 H5 文件核对](evidence/response_completeness_2026-10-17/other_h5_evidence_checked.csv)：反应时／解码等下游模型的 mask 未当成 HGA 响应 mask。
- [一致性检查](evidence/response_completeness_2026-10-17/validation.json)

要真正补齐这些阴性或阳性证据，优先恢复上述清单中的原版本全通道 `statistics/.../bipolar/*desc-Repeat_highgamma.h5`。若没有备份，再根据上游可用性表恢复相应 trial-level power epochs 和 baseline，并重建所缺统计。只重打包当前筛选过的 CSV，无法恢复已经被省略的通道 mask。

本轮未启动上述统计重建。响应计数继续保留上一轮的 **227 个有显著响应证据＋2 个完整评估且未显著＋94 个证据不完整＝323 个解剖合格位点**。

## 运行与保留

文件盘点作业 58867509、CSV 版本核对作业 58867533、逐项诊断作业 58867619；最终备份格式与一致性核对作业编号见本轮日志。1,572 个扫描 CSV 的文件属性未变，67 个受保护文件（包括 Figure 1 Notebook、NMF 结果和上一轮审核表）哈希未变。可复核脚本位于 `scripts/*response_completeness.py` 和 `scripts/scan_response_backups.py`。
'''
    (BASE/'docs/response_completeness_review_2026-10-17.md').write_text(report)
    print(json.dumps(checks,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
