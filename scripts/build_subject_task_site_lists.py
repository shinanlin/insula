#!/usr/bin/env python3
"""Export participant x task sampling/response counts and explicit site lists.

Uses the reviewed anatomy and saved-mask evidence; does not fit NMF or draw figures.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

BASE=Path('/cwork/ns458/insula-paper')
SOURCE=Path('/hpc/group/coganlab/nanlinshi/insula-functional')
AUDIT=BASE/'results/audit/four_task_selection_2026-10-17'
OUT=BASE/'results/audit/subject_task_site_lists_2026-10-17'
EVIDENCE=BASE/'docs/evidence/subject_task_site_lists_2026-10-17'
TASKS=['PhonemeSequence','LexicalDelay','PictureNaming','SentenceRep']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def join_ids(values):
    return '|'.join(sorted(set(values)))


def markdown(frame):
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join(
        '| '+' | '.join(map(str,row))+' |\n' for row in frame.itertuples(index=False,name=None))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    input_paths=[AUDIT/name for name in ['site_anatomy_audit.csv','task_site_anatomy_audit.csv',
        'task_site_response_audit.csv','response_evidence.csv','coordinate_manifest.json']]
    inputs={str(p):digest(p) for p in input_paths}
    protected_paths=[SOURCE/'vizpub/fig1.ipynb',BASE/'vizpub/fig1.ipynb']
    protected_paths+=list((SOURCE/'results/nmf').glob('*.csv'))
    protected_paths+=[SOURCE/'results/nmf/nmf_manifest.json',SOURCE/'results/nmf/exclude_channels.txt']
    protected={str(p):digest(p) for p in protected_paths if p.is_file()}
    anatomy=pd.read_csv(AUDIT/'task_site_anatomy_audit.csv')
    response=pd.read_csv(AUDIT/'task_site_response_audit.csv')
    evidence=pd.read_csv(AUDIT/'response_evidence.csv')
    sampled=anatomy.loc[anatomy.status.eq('include'),
        ['subject','task','channel','contact_1','contact_2','old_midpoint_labels']].copy()
    sampled=sampled.merge(response[['task','channel','any_significant','response_status',
        'stimulus_significant','delay_significant','response_significant','all_three_phases_available']],
        on=['task','channel'],how='left',validate='one_to_one')
    assert sampled.any_significant.notna().all()
    sampled=sampled.rename(columns={'channel':'site_id','old_midpoint_labels':'hammers_midpoint_labels',
        'any_significant':'confirmed_responsive','response_status':'response_evidence_status',
        'all_three_phases_available':'response_evidence_complete'})
    sampled['significant_phases']=sampled.apply(lambda r:'|'.join(
        p for p in ['stimulus','delay','response'] if r[p+'_significant']),axis=1)
    positive=evidence.loc[evidence.significant & ~evidence.evidence_conflict]
    positive_groups={k:g for k,g in positive.groupby(['task','channel'])}
    sampled['significant_modalities']=[join_ids(positive_groups[(r.task,r.site_id)].modality)
        if (r.task,r.site_id) in positive_groups else '' for r in sampled.itertuples()]
    sampled=sampled.sort_values(['subject','task','site_id']).reset_index(drop=True)
    responsive=sampled.loc[sampled.confirmed_responsive].copy()
    assert not sampled.duplicated(['subject','task','site_id']).any()
    assert len(sampled)==564 and len(responsive)==341

    # Include every subject/task with a coordinate export, including rows with zero eligible sites.
    recorded=set()
    for item in json.loads((AUDIT/'coordinate_manifest.json').read_text()):
        name=Path(item['path']).name
        task=re.search(r'_task-([^_]+)',name)[1]
        subject=re.search(r'^sub-([^_]+)',name)[1]
        assert task in TASKS
        recorded.add((subject,task))
    assert set(zip(sampled.subject,sampled.task)).issubset(recorded)
    per_group={k:g for k,g in sampled.groupby(['subject','task'])}
    counts=[]
    for subject,task in sorted(recorded):
        rows=per_group.get((subject,task),sampled.iloc[:0])
        sig=rows.loc[rows.confirmed_responsive]
        counts.append(dict(subject=subject,task=task,
            n_sampled_bipolar_sites=len(rows),n_confirmed_responsive_bipolar_sites=len(sig),
            sampled_site_ids=join_ids(rows.site_id),confirmed_responsive_site_ids=join_ids(sig.site_id)))
    counts=pd.DataFrame(counts)
    assert counts.n_sampled_bipolar_sites.sum()==len(sampled)
    assert counts.n_confirmed_responsive_bipolar_sites.sum()==len(responsive)
    assert counts.n_confirmed_responsive_bipolar_sites.le(counts.n_sampled_bipolar_sites).all()

    unique=[]
    for site_id,rows in sampled.groupby('site_id'):
        sig=rows.loc[rows.confirmed_responsive]
        unique.append(dict(subject=rows.subject.iloc[0],site_id=site_id,
            sampled_in_tasks=join_ids(rows.task),responsive_in_tasks=join_ids(sig.task),
            confirmed_responsive=bool(len(sig)),contact_1=rows.contact_1.iloc[0],contact_2=rows.contact_2.iloc[0]))
    unique=pd.DataFrame(unique).sort_values(['subject','site_id'])
    nmf_candidates=unique.loc[unique.confirmed_responsive].copy()
    assert len(unique)==323 and len(nmf_candidates)==227
    assert unique.subject.nunique()==65
    summary=[]
    for task in TASKS+['ALL_UNIQUE']:
        one=sampled if task=='ALL_UNIQUE' else sampled.loc[sampled.task.eq(task)]
        sig=one.loc[one.confirmed_responsive]
        summary.append(dict(task=task,
            n_subjects_with_sampled_sites=one.subject.nunique(),n_sampled_bipolar_sites=one.site_id.nunique(),
            n_subjects_with_responsive_sites=sig.subject.nunique(),n_confirmed_responsive_bipolar_sites=sig.site_id.nunique(),
            sampled_subject_ids=join_ids(one.subject),responsive_subject_ids=join_ids(sig.subject)))
    summary=pd.DataFrame(summary)
    assert summary.n_sampled_bipolar_sites.tolist()==[161,199,151,53,323]
    assert summary.n_confirmed_responsive_bipolar_sites.tolist()==[98,115,102,26,227]

    wide=[]
    counts_index=counts.set_index(['subject','task'])
    for subject,rows in unique.groupby('subject'):
        row={'subject':subject}
        for task in TASKS:
            key=(subject,task)
            for src,label in [('n_sampled_bipolar_sites','sampled'),
                              ('n_confirmed_responsive_bipolar_sites','confirmed_responsive')]:
                row[task+'_'+label]=int(counts_index.loc[key,src]) if key in counts_index.index else None
        row['ALL_UNIQUE_sampled']=len(rows)
        row['ALL_UNIQUE_confirmed_responsive']=int(rows.confirmed_responsive.sum())
        wide.append(row)
    wide=pd.DataFrame(wide)
    for column in wide.columns[1:]:wide[column]=wide[column].astype('Int64')
    assert len(wide)==65

    tables={'task_summary.csv':summary,'subject_task_counts_and_lists.csv':counts,
        'subject_task_counts_wide.csv':wide,'sampled_sites_by_subject_task.csv':sampled,
        'responsive_sites_by_subject_task.csv':responsive,'sampled_sites_unique.csv':unique,
        'nmf_candidate_sites_not_final_inputs.csv':nmf_candidates}
    for name,frame in tables.items():frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    notes=pd.DataFrame([
        ('计数单位','一个满足当前中点及双端点组织／分区规则的岛叶双极位点；同一任务内不重复计阶段、条件或模态。'),
        ('采样位点','当前统一解剖规则纳入的岛叶位点；组织不明确、身份冲突和明确排除者均暂不计入。'),
        ('显著位点','上述采样位点中，现有且版本核实的 Repeat mask 在 Stimulus 0–0.6 s、Delay 0–1 s 或 Response −0.5–0.5 s 任一窗口显著；合并 sound/image/text。沿用开区间。'),
        ('显著计数状态','已确认数量，尚未作为完整队列的最终显著总数定稿。94 个跨任务去重位点仍缺完整证据；不能将全部未入显著名单者写成无响应。'),
        ('参与者×任务表','包括全部有现存坐标记录的参与者×任务组合。0 表示该任务记录中没有符合当前纳入条件的位点。'),
        ('参与者宽表','65 位有纳入采样位点的参与者；空白表示没有该参与者×任务的现存坐标记录，不代表已核实没有做该任务。'),
        ('跨任务去重','同一参与者的同一双极位点跨任务只计一次；逐任务数不能直接相加。'),
        ('NMF 名单','227 个已确认显著位点是新 NMF 的候选筛选名单；并非旧 255 位点，也不是已确认三阶段数据覆盖完整的最终拟合输入。本轮未读写 NMF 模型或生成图片。'),
    ],columns=['项目','定义与备注'])
    xlsx=OUT/'参与者任务_采样与显著位点清单.xlsx'
    cn_counts=counts.rename(columns={'subject':'参与者','task':'任务',
        'n_sampled_bipolar_sites':'采样双极位点数','n_confirmed_responsive_bipolar_sites':'已确认显著位点数',
        'sampled_site_ids':'采样位点名单','confirmed_responsive_site_ids':'显著位点名单'})
    with pd.ExcelWriter(xlsx,engine='openpyxl') as writer:
        sheets={'口径说明':notes,'任务汇总':summary,'参与者×任务':cn_counts,
            '65位参与者宽表':wide,'逐任务采样位点':sampled,'逐任务显著位点':responsive,
            '跨任务去重采样':unique,'新NMF候选_尚非最终输入':nmf_candidates}
        for name,frame in sheets.items():frame.to_excel(writer,sheet_name=name,index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes='C2' if sheet.title=='参与者×任务' else 'B2'
            sheet.auto_filter.ref=sheet.dimensions
            for cell in sheet[1]:
                cell.font=Font(bold=True,color='FFFFFF')
                cell.fill=PatternFill('solid',fgColor='36576E')
            for col in sheet.columns:
                width=max(len(str(c.value or '')) for c in list(col)[:70])+2
                sheet.column_dimensions[col[0].column_letter].width=min(max(width,15),70)
                for cell in col:cell.alignment=Alignment(vertical='top',wrap_text=True)
    validation=dict(passed=True,
        sampled_task_site_rows=len(sampled),responsive_task_site_rows=len(responsive),
        sampled_unique_sites=len(unique),confirmed_responsive_unique_sites=len(nmf_candidates),
        sampled_subjects=unique.subject.nunique(),responsive_subjects=nmf_candidates.subject.nunique(),
        all_recorded_subject_task_rows=len(counts),all_recorded_subjects=counts.subject.nunique(),
        subject_wide_rows=len(wide),inputs_unchanged=all(digest(Path(p))==h for p,h in inputs.items()),
        protected_unchanged=all(digest(Path(p))==h for p,h in protected.items()),
        figures_generated=False,nmf_fitted=False,nmf_input_finalized=False,
        significance_count_finalized=False)
    assert validation['inputs_unchanged'] and validation['protected_unchanged']
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (OUT/'input_and_protection_manifest.json').write_text(json.dumps(dict(inputs=inputs,protected=protected),indent=2)+'\n')
    for name in list(tables)+[xlsx.name,'validation.json','input_and_protection_manifest.json']:
        shutil.copy2(OUT/name,EVIDENCE/name)
    compact=summary.iloc[:,:5].rename(columns={'task':'任务','n_subjects_with_sampled_sites':'采样参与者数',
        'n_sampled_bipolar_sites':'采样位点数','n_subjects_with_responsive_sites':'显著参与者数',
        'n_confirmed_responsive_bipolar_sites':'已确认显著位点数'})
    report='# 参与者 × 任务：采样与显著位点清单\n\n'
    report+='计数单位为符合当前解剖纳入规则的岛叶双极位点。显著位点按已约定的 Repeat 三阶段窗口合并模态，取现有核实 mask 的阳性证据。\n\n'
    report+=markdown(compact)+'\n'
    report+='逐任务数量有跨任务重复；ALL_UNIQUE 按参与者＋双极名称取并集。\n\n'
    report+=f'完整长表覆盖 **{len(counts)} 条参与者×任务记录**；65 位纳入采样队列参与者另有宽表。每条计数均附对应采样和显著位点名称，任务汇总另列参与者 ID。\n\n'
    report+='## 文件\n\n'
    prefix='evidence/subject_task_site_lists_2026-10-17/'
    for label,name in [('完整工作簿',xlsx.name),('参与者×任务计数及名单','subject_task_counts_and_lists.csv'),
        ('65 位参与者宽表','subject_task_counts_wide.csv'),('逐任务采样位点','sampled_sites_by_subject_task.csv'),
        ('逐任务显著位点','responsive_sites_by_subject_task.csv'),('跨任务去重的新 NMF 候选名单','nmf_candidate_sites_not_final_inputs.csv')]:
        report+=f'- [{label}]({prefix+name})\n'
    report+='\n## 使用说明\n\n'
    report+='- 采样数量和名单已按当前解剖规则列齐。0 表示有该任务记录但没有当前纳入位点；宽表空白表示缺少该参与者×任务的现存坐标记录。\n'
    report+='- 显著列为当前**已确认**阳性数量。94 个位点尚缺完整响应证据，未据此判为阴性，因此该列暂不能作为完整队列的最终显著总数。工作簿保留此备注。\n'
    report+='- 227 个跨任务去重显著位点可用作新 NMF 的候选筛选名单；正式拟合还须检查拟采用任务／模态的三阶段波形覆盖。此处不把它写成已经确认的最终拟合人数。\n'
    report+='- 本轮只汇总现有审核结果。Figure 1、NMF 模型和旧输入均未改动；输入及受保护文件哈希核对通过。\n'
    (BASE/'docs/subject_task_site_lists_2026-10-17.md').write_text(report)
    print(compact.to_string(index=False),flush=True)
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':main()
