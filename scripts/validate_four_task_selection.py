#!/usr/bin/env python3
"""Check rule invariants and export review tables, without regenerating figures."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_four_task_selection import BASE,OUT,TASKS,category


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    sites=pd.read_csv(OUT/'site_anatomy_audit.csv',keep_default_na=False)
    task=pd.read_csv(OUT/'task_site_anatomy_audit.csv',keep_default_na=False)
    point=pd.read_csv(OUT/'point_tissue_evidence.csv')
    response=pd.read_csv(OUT/'task_site_response_audit.csv',keep_default_na=False)
    evidence=pd.read_csv(OUT/'response_evidence.csv')
    stages=pd.read_csv(OUT/'stage_counts.csv')
    included=sites.loc[sites.status.eq('include')]
    names=set(included.channel)
    assert sites.channel.is_unique
    assert not task.duplicated(['task','channel']).any()
    assert len(sites)==sum(summary['anatomy_status'].values())
    assert sites.historical_464.sum()==464
    assert summary['old_464_included']+summary['old_464_removed_or_pending']==464
    assert set(task.loc[task.status.eq('include'),'channel'])==names
    assert task.groupby('channel').status.nunique().eq(1).all()
    assert included.identity_issues.eq('').all()
    assert not included.subject.eq('D0121').any()
    assert not (included.subject.eq('D0125') &
                (included.contact_1.eq('LIA6')|included.contact_2.eq('LIA6'))).any()
    assert included.endpoint_1_status.eq('include').all() and included.endpoint_2_status.eq('include').all()
    coords=pd.read_csv(OUT/'candidate_coordinate_evidence.csv')
    checked=0
    for site in included.itertuples():
        positions=point.loc[point.channel.eq(site.channel)].set_index('position')
        mid=positions.loc['midpoint']
        if pd.notna(mid.maper_tissue_code):
            assert int(mid.maper_tissue_code)==2 and category(mid.maper_raw_label)=='insula'
        else:
            assert all(category(label)=='insula' for label in coords.loc[coords.channel.eq(site.channel),'label'])
        for i in [1,2]:
            p=positions.loc[f'contact_{i}']
            kind=getattr(site,f'endpoint_{i}_evidence')
            if 'WM' in kind:
                assert p.native_FS_class=='white_matter'
                assert 'Cerebral-White-Matter' in p.native_aseg_label
                assert p.native_ribbon_code in {2,41}
                if pd.notna(p.maper_tissue_code):assert int(p.maper_tissue_code)==3
                assert not p.native_archived_FS_conflict
            else:
                if pd.notna(p.maper_tissue_code):
                    assert int(p.maper_tissue_code)==2 and category(p.maper_raw_label)=='insula'
                else:
                    assert all(category(label)=='insula' for label in
                        coords.loc[coords.channel.eq(site.channel),f'contact_{i}_label'])
            checked+=1
    for taskname,group in stages.groupby('task',sort=False):
        assert group.remaining.diff().dropna().le(0).all(),taskname
        assert int(group.loc[group.stage.eq('eligible_anatomy'),'remaining'].iloc[0])==(
            len(included) if taskname=='ALL_UNIQUE' else int((task.task.eq(taskname)&task.status.eq('include')).sum()))
    changed=[]
    manifests=json.loads((OUT/'coordinate_manifest.json').read_text())+json.loads((OUT/'response_input_manifest.json').read_text())
    for record in manifests:
        path=Path(record['path'])
        if (path.stat().st_size,path.stat().st_mtime_ns)!=(record['bytes'],record['mtime_ns']):changed.append(str(path))
    assert not changed,changed
    # Summaries for review: every candidate has a single anatomical decision
    # and a separate response decision. Unknown response does not become negative.
    output=[]
    windows_lost=[]
    response_sets={}
    for site in sites.itertuples():
        one=response.loc[response.channel.eq(site.channel)]
        positive=bool(one.any_significant.any())
        complete=bool(len(one) and one.all_three_phases_available.all())
        old_positive=bool(one.old_window_any_significant.any())
        reason='至少一个任务／阶段显著' if positive else '各任务三阶段记录完整，窗口内未显著' if complete else '显著性记录不完整，不能判为无响应'
        out=dict(位点=site.channel,参与者=site.subject,任务=site.tasks,
            解剖结论={'include':'纳入','exclude':'排除','pending':'待核对，暂不纳入'}[site.status],
            首次未通过步骤=site.first_stage,主要原因=site.reason_zh,
            附加已记录原因=site.all_reasons,旧464内=site.historical_464,旧26手动名单内=site.old_manual26,
            响应结论=reason,显著响应=positive,三阶段记录完整=complete,
            最终响应分析纳入=site.status=='include' and positive)
        for phase in ['stimulus','delay','response']:
            out[phase+'显著']=bool(one[f'{phase}_significant'].any())
        output.append(out)
        response_sets[site.channel]=positive
        if site.status=='include' and old_positive and not positive:
            windows_lost.append(dict(channel=site.channel,subject=site.subject,tasks=site.tasks,
                reason_zh='仅旧 Stimulus 0–1 s 窗口显著，缩为 0–0.6 s 后三阶段均无显著响应',
                response_status='not_responsive' if complete else 'incomplete'))
    review=pd.DataFrame(output)
    review.to_csv(OUT/'逐位点审核总表.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(windows_lost,columns=['channel','subject','tasks','reason_zh','response_status']).to_csv(
        OUT/'response_lost_due_to_window_change.csv',index=False)
    for taskname in TASKS:
        one=task.loc[task.task.eq(taskname)].copy()
        one.to_csv(OUT/f'{taskname}_逐位点解剖审核.csv',index=False,encoding='utf-8-sig')
        one.loc[~one.status.eq('include')].to_csv(OUT/f'{taskname}_排除及待核对.csv',index=False,encoding='utf-8-sig')
    rates=[]
    for taskname in TASKS+['ALL_UNIQUE']:
        one=included if taskname=='ALL_UNIQUE' else task.loc[task.task.eq(taskname)&task.status.eq('include')]
        ids=set(one.channel)
        task_response=response if taskname=='ALL_UNIQUE' else response.loc[response.task.eq(taskname)]
        positive_ids=set(task_response.loc[task_response.any_significant,'channel'])
        responsive=one.loc[one.channel.isin(positive_ids)]
        expected_response=next(r['responsive'] for r in summary['response_counts'] if r['task']==taskname)
        assert responsive.channel.nunique()==expected_response
        hemi=[]
        for channel in ids:
            x=point.loc[point.channel.eq(channel)&point.position.eq('midpoint'),'x_native'].iloc[0]
            hemi.append('L' if x<0 else 'R')
        rates.append(dict(task=taskname,anatomy_eligible_sites=len(ids),
            anatomy_eligible_subjects=one.subject.nunique(),left=hemi.count('L'),right=hemi.count('R'),
            responsive_sites=responsive.channel.nunique(),responsive_subjects=responsive.subject.nunique()))
    pd.DataFrame(rates).to_csv(OUT/'participant_coverage_counts.csv',index=False)
    # Stimulus-specific losses can remain responsive in Delay or Response.
    stim=evidence.loc[evidence.phase.eq('stimulus')&evidence.channel.isin(names)&~evidence.evidence_conflict]
    old_stim=set(stim.loc[stim.old_window_significant,'channel'])
    new_stim=set(stim.loc[stim.significant,'channel'])
    lost=sites.loc[sites.channel.isin(old_stim-new_stim)].copy()
    lost['still_responsive_in_other_phase']=lost.channel.map(response_sets)
    lost.to_csv(OUT/'stimulus_response_lost_due_to_window_change.csv',index=False)
    validation=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='passed',
        anatomical_sites_checked=len(included),endpoint_decisions_checked=checked,
        coordinate_and_response_input_files_unchanged=len(manifests),
        cross_task_eligibility_consistent=True,old464_fully_accounted_for=True,
        no_included_D0121_or_LIA6=True,
        confirmed_WM_has_concordant_native_aparc_aseg_ribbon=True,
        stimulus_only_window_losses=len(lost),all_phase_response_window_losses=len(windows_lost),
        participant_counts=rates,figures_generated=False,nmf_rerun=False)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(validation,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
