#!/usr/bin/env python3
"""Package the completed audit for user review; no figure generation."""
import json
from pathlib import Path
import shutil

import pandas as pd

from audit_four_task_selection import BASE,OUT,TASKS


def markdown_table(frame):
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join(
        '| '+' | '.join(str(v) for v in row)+' |\n' for row in frame.itertuples(index=False,name=None))


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    validation=json.loads((OUT/'validation.json').read_text())
    assert validation['status']=='passed'
    sites=pd.read_csv(OUT/'site_anatomy_audit.csv',keep_default_na=False)
    tasks=pd.read_csv(OUT/'task_site_anatomy_audit.csv',keep_default_na=False)
    response=pd.read_csv(OUT/'task_site_response_audit.csv',keep_default_na=False)
    review=pd.read_csv(OUT/'逐位点审核总表.csv',keep_default_na=False)
    stages=pd.read_csv(OUT/'stage_counts.csv')
    rates=pd.read_csv(OUT/'participant_coverage_counts.csv')
    rc=pd.read_csv(OUT/'response_counts.csv')
    reasons=sites.groupby(['status','primary_reason','reason_zh']).size().reset_index(name='n_sites')
    reasons.to_csv(OUT/'global_reason_counts.csv',index=False)
    pending_subjects=sites.loc[sites.status.eq('pending')].groupby(
        ['subject','first_stage','primary_reason','reason_zh']).size().reset_index(name='n_sites')
    pending_subjects.to_csv(OUT/'pending_by_subject_reason.csv',index=False)
    old=sites.loc[sites.historical_464]
    old_status=old.status.value_counts().to_dict()
    old.groupby(['status','primary_reason','reason_zh']).size().reset_index(name='n_sites').to_csv(
        OUT/'old464_reason_counts.csv',index=False)
    oldR=set(pd.read_csv(BASE/'results/audit/fig1_nmf_selection_2026-10-17/responsive_fig1.csv').channel)
    oldR_review=review.loc[review['位点'].isin(oldR)]
    oldR_review.to_csv(OUT/'旧348逐位点重审.csv',index=False,encoding='utf-8-sig')
    oldR_eligible=int((sites.channel.isin(oldR)&sites.status.eq('include')).sum())
    oldR_positive=int((oldR_review['解剖结论'].eq('纳入')&oldR_review['显著响应']).sum())
    per_task={}
    for name in TASKS:
        one=tasks.loc[tasks.task.eq(name),['task','channel','subject','status','first_stage','reason_zh','historical_464','old_manual26']]
        one=one.merge(response.loc[response.task.eq(name)].drop(columns=['subject','anatomy_status']),
                      on=['task','channel'],validate='one_to_one')
        one=one.rename(columns={'task':'任务','channel':'位点','subject':'参与者','status':'解剖结论',
            'first_stage':'首次未通过步骤','reason_zh':'原因','historical_464':'旧464内','old_manual26':'旧26名单内',
            'response_status':'响应结论','any_significant':'窗口内显著响应',
            'all_three_phases_available':'三阶段模态记录完整','missing_phase_modality_cells':'缺少的阶段模态'})
        one['解剖结论']=one['解剖结论'].map({'include':'纳入','exclude':'排除','pending':'待核对，暂不纳入'})
        one.to_csv(OUT/f'{name}_审核清单.csv',index=False,encoding='utf-8-sig')
        per_task[name]=one
    stage_pivot=stages.pivot(index='stage_zh',columns='task',values='remaining')
    stage_order=stages.loc[stages.task.eq('ALL_UNIQUE'),'stage_zh'].tolist()
    stage_pivot=stage_pivot.loc[stage_order,TASKS+['ALL_UNIQUE']].reset_index()
    stage_pivot=stage_pivot.rename(columns={'stage_zh':'筛选步骤','ALL_UNIQUE':'跨任务去重'})
    stage_pivot.to_csv(OUT/'四任务逐步数量.csv',index=False,encoding='utf-8-sig')
    xlsx=None
    try:
        import openpyxl
        from openpyxl.styles import Alignment,Font,PatternFill
        xlsx=OUT/'四任务筛选审核.xlsx'
        sheets={'逐步数量':stage_pivot,'解剖纳入323':review.loc[review['解剖结论'].eq('纳入')],
                '明确排除86':review.loc[review['解剖结论'].eq('排除')],
                '待核对197':review.loc[review['解剖结论'].eq('待核对，暂不纳入')],
                '显著性数量':rc,'旧26位点重审':review.loc[review['旧26手动名单内']],
                '旧348重审':oldR_review}
        sheets.update(per_task)
        with pd.ExcelWriter(xlsx,engine='openpyxl') as writer:
            for sheet,table in sheets.items():table.to_excel(writer,sheet_name=sheet,index=False)
            for ws in writer.book.worksheets:
                ws.freeze_panes='A2'
                ws.auto_filter.ref=ws.dimensions
                for cell in ws[1]:
                    cell.font=Font(bold=True,color='FFFFFF')
                    cell.fill=PatternFill('solid',fgColor='36576E')
                for column in ws.columns:
                    letter=column[0].column_letter
                    width=max(len(str(c.value or '')) for c in list(column)[:80])+2
                    ws.column_dimensions[letter].width=min(max(width,14),55)
                    for cell in column:cell.alignment=Alignment(vertical='top',wrap_text=True)
    except ImportError:
        pass
    evidence=BASE/'docs/evidence/four_task_selection_2026-10-17'
    evidence.mkdir(parents=True,exist_ok=True)
    selected=['summary.json','validation.json','inventory_summary.json','stage_counts.csv','四任务逐步数量.csv',
        'global_reason_counts.csv','exclusion_counts_by_task.csv','response_counts.csv','participant_coverage_counts.csv',
        '逐位点审核总表.csv','site_anatomy_audit.csv','included_anatomy_sites.csv','excluded_anatomy_sites.csv',
        'pending_anatomy_sites.csv','pending_by_subject_reason.csv','old464_reason_counts.csv',
        'historical_464_removed_or_pending.csv','newly_eligible_vs_464.csv','old_manual26_reassessment.csv',
        'D0125_LIA6_all_affected_pairs.csv','point_tissue_evidence.csv','candidate_coordinate_evidence.csv',
        'task_site_response_audit.csv','response_evidence.csv','response_mask_conflicts.csv',
        'response_lost_due_to_window_change.csv','stimulus_response_lost_due_to_window_change.csv',
        'outside_insula_candidate_pool.csv','旧348逐位点重审.csv',
        'coordinate_manifest.json','anatomy_input_manifest.json','response_input_manifest.json']
    selected += [f'{name}_审核清单.csv' for name in TASKS]
    if xlsx:selected.append(xlsx.name)
    for name in selected:shutil.copyfile(OUT/name,evidence/name)
    prefix='evidence/four_task_selection_2026-10-17/'
    body='# 四任务统一位点筛选审核\n\n日期：2026-10-17。审核版本，等待用户确认后才重新生成 Figure 1。**Figure 1、原始位点名单和 NMF 均未修改。**\n\n'
    body+='## 1 数量与逐步筛选\n\n'
    body+='现有四任务坐标表共 15,986 个参与者＋双极名称去重记录；606 个记录在任一位置、ROI 或用户指定 LIA6 排除范围内涉及岛叶候选。候选中 **323 个解剖合格、86 个明确排除、197 个待核对且暂不纳入**。合格位点来自 **65 位参与者，左侧 193、右侧 130**。\n\n'
    body+=markdown_table(stage_pivot)+'\n'
    body+='各任务按参与者＋双极名称去重；跨任务列再取并集，不能把四任务数量相加。同名位点的原生位置或植入版本不一致时，整组先待核对；没有取第一个任务的标签代表所有任务。待核对数量不是已确认的不合格组织数量。\n\n'
    body+='D0121 噪声排除规则已应用，本次岛叶候选中无该参与者，减少 0 个。含 LIA6 的两对 **D0125_LIA5-6、D0125_LIA6-7** 均排除；当前坐标记录中它们只出现在 PictureNaming，没有在其他任务虚构相应记录。\n\n'
    body+='## 2 明确排除与待核对原因\n\n'
    for status,title in [('exclude','明确排除：86 个'),('pending','待核对、暂不纳入：197 个')]:
        body+='### '+title+'\n\n'
        table=reasons.loc[reasons.status.eq(status),['reason_zh','n_sites']].rename(columns={'reason_zh':'首次未通过原因','n_sites':'位点数'})
        body+=markdown_table(table)+'\n'
    body+='每个位点按第一次未通过的步骤计数，所以上表可相加。主表中的附加原因是已经记录的证据，不表示对早期已排除位点完成了所有后续检查。所有原始标签、端点坐标及可用逐体素证据另存，便于复核。\n\n'
    body+='## 3 与旧 464／348／26 名单的关系\n\n'
    body+=f"旧 464 中：**{old_status.get('include',0)} 个保留、{old_status.get('exclude',0)} 个明确排除、{old_status.get('pending',0)} 个待核对**；另有旧 464 之外的 **24 个**满足当前规则，构成 **299＋24＝323**。旧规则的 AIC/PIC pure 限制不再使用；岛叶内部跨分区不自动排除，仍须满足中点及端点要求。\n\n"
    body+='旧 464 内这 2 个明确排除位点为 **D0125_LIA5-6**（含 LIA6）和 **D0125_LTMS3-4**（端点 CSF）。另一个含 LIA6 的 **D0125_LIA6-7** 本来就不属于旧 464，也已在新候选审核中明确排除。\n\n'
    body+=f'旧 348 个响应位点中，{oldR_eligible} 个通过本次解剖审核，其中 {oldR_positive} 个仍有新窗口内显著响应证据；逐条结果见旧 348 重审表。当前 227 不是由旧 348 简单减去一个统一数字得到的。\n\n'
    body+='旧 26 个手动排除位点均重新接受筛选：**21 个解剖合格、5 个待核对**；没有继续以旧手动名单本身作为排除理由。\n\n'
    body+='## 4 显著响应：227 个已确认，94 个证据不完整\n\n'
    body+='继续使用现存 Repeat 显著性 mask，并合并输入模态。窗口为 Stimulus **0–0.6 s**、Delay **0–1 s**、Response **−0.5–0.5 s**；端点采用原实现的开区间。未重算 p 值或显著性模型。\n\n'
    table=rc.rename(columns={'task':'任务','anatomy_eligible':'解剖合格','responsive':'任一阶段显著',
        'confirmed_not_responsive':'记录完整但未显著','response_incomplete':'记录不完整',
        'stimulus':'Stimulus','delay':'Delay','response':'Response'})
    body+=markdown_table(table)+'\n'
    body+='**323＝227 个有响应证据＋2 个所有相关任务记录完整且未显著＋94 个记录不完整。** 后 94 个不能计作已确认无响应，因此此时不能将 227/323 直接写成完整评估队列的响应率。其缺失阶段／模态逐条列在审核表中。\n\n'
    body+='按参与者×模态筛过的历史打包文件，不能提供所有采样位点的完整阴性证据。能找到匹配时间轴的现存全通道统计 mask 已用于补充；其余缺失保持未知，没有当作 False。现存全通道统计与打包 mask 未发现窗口内显著性冲突。\n\n'
    body+='在解剖合格名单内，Stimulus 从 0–1 s 缩为 0–0.6 s，减少 **21 个 Stimulus 响应位点**；其中多数仍在 Delay 或 Response 显著。现存三阶段响应并集减少 **2 个**：**D0084_RAI1-2、D0123_LAI1-2**。两者均因其他记录不完整，暂归入“响应证据不完整”，不能据此判为完全无响应。\n\n'
    body+='## 5 证据口径与适用范围\n\n'
    body+='- **Hammers 是主要分区 atlas**：中点必须同时满足岛叶分区和灰质组织；所有岛叶标签允许跨 AIC/PIC 内部分区。非零历史 Hammers 标签含原流程的灰质门控证据。\n'
    body+='- **0 不代表白质**：0 端点仅在原生 aparc、aseg、ribbon 三者一致为脑白质时放行；如果当前 MAPER 组织图可用，还须支持白质。缺失、Unknown、灰白质或版本冲突均待核对。\n'
    body+='- 原生 FS、aseg、ribbon 不是三种独立实验；这里要求它们内部一致，不把重复标签当作独立证据。对 Hammers 为 0、FS 却为岛叶灰质的端点，仍保留为待核对，没有替换成 Hammers 岛叶灰质。\n'
    body+='- 可用的新 MAPER（包括已重建的 D0125）优先用于实际组织和分区取样；其他参与者使用历史 Hammers 和可用原生 MRI。此次没有批量重建其他参与者的 MAPER。\n'
    body+='- 明确的原生非岛叶灰质／其他不允许组织端点排除。组织不明或档案不一致单列，暂不纳入。\n'
    body+='- 有效原生坐标用于解剖判断；模板显示坐标仅记录其可用性。未因缺少模板显示坐标将本来解剖合格的位点自动排除。\n\n'
    body+='## 6 审核文件\n\n'
    if xlsx:body+=f'- [四任务筛选审核工作簿]({prefix}{xlsx.name})：逐步数量、合格、明确排除、待核对、原 26／348 重审及四个任务逐位点表。\n'
    for label,name in [('中文逐位点审核总表','逐位点审核总表.csv'),('每一步数量','四任务逐步数量.csv'),
        ('明确排除名单','excluded_anatomy_sites.csv'),('待核对名单','pending_anatomy_sites.csv'),
        ('待核对原因按参与者细分','pending_by_subject_reason.csv'),('逐触点组织证据','point_tissue_evidence.csv'),
        ('旧464未保留的逐位点原因','historical_464_removed_or_pending.csv'),('新增24个合格位点','newly_eligible_vs_464.csv'),
        ('逐任务响应证据及缺失记录','task_site_response_audit.csv'),('全脑未进入岛叶候选池的记录','outside_insula_candidate_pool.csv'),
        ('一致性检查','validation.json')]:
        body+=f'- [{label}]({prefix}{name})\n'
    body+='\n各任务的独立审核 CSV：\n\n'+''.join(f'- [{name}]({prefix}{name}_审核清单.csv)\n' for name in TASKS)
    body+='\n## 7 运行与保留\n\n盘点作业 **58867025**、正式核对 **58867188**（3 分 16 秒）、最终一致性检查 **58867324**。核对了 646 个已纳入端点决策，2,892 个坐标／响应输入文件在核对期间未改变，Figure 1 Notebook 与 NMF 文件哈希未变。早期开发作业 58867117、58867160 已取消，不作为最终结果。\n\n'
    body+='审核脚本为 `scripts/audit_four_task_selection.py`，一致性核对为 `scripts/validate_four_task_selection.py`。详细输出保存在 `results/audit/four_task_selection_2026-10-17/`；本页链接的小型证据已固定到论文代码目录。等待用户确认名单和记录不完整的处理后，再重新生成 Figure 1。\n'
    report=BASE/'docs/four_task_selection_review_2026-10-17.md'
    report.write_text(body)
    print('Saved review:',report,flush=True)
    print('Old464 status:',old_status,flush=True)
    print('Old348 anatomy eligible:',oldR_eligible,'new responsive:',oldR_positive,flush=True)
    print('Workbook:',xlsx,flush=True)
    print('Fixed evidence files:',len(selected),flush=True)


if __name__=='__main__':main()
