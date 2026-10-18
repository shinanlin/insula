#!/usr/bin/env python3
"""Verify and document the user-requested Unknown endpoint restoration."""
from pathlib import Path
import hashlib,json,sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.main_analysis import COHORT_DIR,NMF_WINDOWS

BEFORE=ROOT/'docs/evidence/before_unknown_endpoints_2026-10-17'
OUT=ROOT/'docs/evidence/unknown_endpoints_restored_2026-10-17'
ANATOMY=ROOT/'docs/evidence/four_task_selection_2026-10-17'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 archived=json.loads((BEFORE/'archive_manifest.json').read_text())
 assert all(sha(BEFORE/name)==digest for name,digest in archived.items())
 previous=pd.read_csv(BEFORE/'sound_only_cohort_2026-10-17/sampled_sites_unique.csv')
 previous_positive=pd.read_csv(BEFORE/'sound_only_cohort_2026-10-17/nmf_candidate_sites_not_final_inputs.csv')
 sampled=pd.read_csv(COHORT_DIR/'sampled_sites_unique.csv')
 positive=pd.read_csv(COHORT_DIR/'nmf_candidate_sites_not_final_inputs.csv')
 anatomy=pd.read_csv(COHORT_DIR/'site_anatomy_audit.csv').set_index('channel')
 restored=pd.read_csv(COHORT_DIR/'unknown_endpoint_restorations.csv')
 assert len(restored)==25
 assert set(sampled.site_id)-set(previous.site_id)==set(restored.channel)
 assert set(previous.site_id).issubset(sampled.site_id)
 assert set(previous_positive.site_id).issubset(positive.site_id)
 assert sampled.loc[sampled.site_id.isin(restored.channel),'has_accepted_unknown_endpoint'].all()
 legacy=pd.read_csv(ROOT/'docs/evidence/nmf_sensory_comparison_2026-10-17/old_sensory_sites_trace.csv')
 old_sensory_restored=set(legacy.channel)&set(restored.channel)
 assert len(old_sensory_restored)==8 and old_sensory_restored.issubset(positive.site_id)
 unresolved={'D0107_RTAS1-2','D0086_RTPS4-9'}
 assert anatomy.loc[sorted(unresolved),'status'].eq('pending').all()
 assert anatomy.loc['D0125_LTMS3-4','status']=='exclude'
 assert anatomy.loc['D0125_LIA5-6','status']=='exclude'
 gray_ids=['D0049_LAI1-2','D0085_RAI1-2','D0096_LTAS2-3','D0103_LTMS1-2','D0118_RTAS2-3']
 assert anatomy.loc[gray_ids,'status'].eq('pending').all()
 coords=pd.read_csv(ANATOMY/'candidate_coordinate_evidence.csv')
 points=pd.read_csv(ANATOMY/'point_tissue_evidence.csv').set_index(['channel','position'])
 detail=[]
 for channel in gray_ids:
  rows=coords.loc[coords.channel.eq(channel)]
  for endpoint in [1,2]:
   labels=sorted(set(rows[f'contact_{endpoint}_label'].astype(str)))
   point=points.loc[(channel,f'contact_{endpoint}')]
   detail.append(dict(channel=channel,contact=point.contact,
    hammers_endpoint_labels='|'.join(labels),
    freesurfer_aparc_label=point.native_FS_label,
    freesurfer_aseg_label=point.native_aseg_label,
    ribbon_code=int(point.native_ribbon_code),
    hammers_midpoint_labels='|'.join(sorted(set(rows.label))),
    status='pending',hammers_zero_endpoint=labels==['0']))
 detail=pd.DataFrame(detail)
 detail.to_csv(OUT/'five_gray_sites_endpoint_labels.csv',index=False)
 summary=json.loads((COHORT_DIR/'summary.json').read_text())
 model=json.loads((ROOT/'results/nmf/nmf_manifest.json').read_text())
 assignments=pd.read_csv(ROOT/'results/nmf/channel_assignments.csv')
 assert set(assignments.channel)==set(positive.site_id)
 assert model['modality']=='sound' and model['condition']=='Repeat' and model['k']==3
 assert model['fit_windows']=={p:list(w) for p,w in NMF_WINDOWS.items()}
 for name in ['fig1','fig2']:
  validation=json.loads((ROOT/f'results/{name}/execution_validation.json').read_text())
  assert validation['passed'] and validation['modality']=='sound'
  manifest=json.loads((ROOT/f'results/{name}/figure_manifest.json').read_text())
  for filename,data in manifest['figures'].items():
   assert sha(ROOT/f'img/{name}'/filename)==(data['sha256'] if isinstance(data,dict) else data)
 assert model['figure1_notebook_sha256']==sha(ROOT/'vizpub/fig1.ipynb')
 membership=pd.read_csv(ROOT/'results/fig1/average_hga_source_membership.csv')
 assert membership.modality.eq('sound').all()
 old_labels=pd.read_csv(BEFORE/'figure2_sound_only_2026-10-17/channel_assignments.csv')
 comparison=assignments[['channel','functional_cluster']].merge(
  old_labels[['channel','functional_cluster']],on='channel',how='outer',suffixes=('_current','_before'),indicator=True)
 comparison.to_csv(OUT/'nmf_label_comparison.csv',index=False)
 restored['sound_responsive']=restored.channel.isin(positive.site_id)
 restored['old_sensory_site']=restored.channel.isin(old_sensory_restored)
 restored=restored.merge(assignments[['channel','functional_cluster']],on='channel',how='left',validate='one_to_one')
 restored.to_csv(OUT/'restored_sites_and_response.csv',index=False)
 task_summary=pd.read_csv(COHORT_DIR/'task_summary.csv')
 result=dict(passed=True,previous_sampled_sites=len(previous),current_sampled_sites=len(sampled),
  previous_responsive_sites=len(previous_positive),current_responsive_sites=len(positive),
  restored_unknown_endpoint_sites=len(restored),restored_sound_responsive_sites=int(restored.sound_responsive.sum()),
  restored_old_sensory_sites=len(old_sensory_restored),gray_sites_still_pending=gray_ids,
  coordinate_conflicts_still_pending=sorted(unresolved),explicit_CSF_sites_still_excluded=True,
  previous_snapshot_unchanged=True,sound_only_selection_and_waveforms_verified=True,
  nmf_k=3,nmf_cluster_counts=model['cluster_counts'],figure1_and_figure2_validated=True)
 (OUT/'validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 descriptions={'ctx_lh_S_circular_insula_inf':'左岛叶下环沟',
  'ctx_rh_G_insular_short':'右岛叶短回',
  'ctx_rh_G_Ins_lg_and_S_cent_ins':'右岛叶长回及中央沟'}
 table='| 双极位点 | Hammers=0 的触点 | FreeSurfer Destrieux 标签 | 另一端触点 | 另一端 Hammers 标签 |\n|---|---|---|---|---|\n'
 for channel in gray_ids:
  pair=detail.loc[detail.channel.eq(channel)]
  zero=pair.loc[pair.hammers_zero_endpoint].iloc[0];other=pair.loc[~pair.hammers_zero_endpoint].iloc[0]
  table+=f'| {channel} | {zero.contact} | `{zero.freesurfer_aparc_label}`（{descriptions[zero.freesurfer_aparc_label]}） | {other.contact} | `{other.hammers_endpoint_labels}` |\n'
 task_table='| Task | 采样位点 | sound 显著位点 |\n|---|---:|---:|\n'
 for row in task_summary.itertuples():
  task_table+=f'| {row.task} | {row.n_sampled_bipolar_sites} | {row.n_confirmed_responsive_bipolar_sites} |\n'
 text=f'''# Unknown 端点恢复与 5 个灰质标签位点说明

## 本次决定与范围

用户要求把端点 Unknown 的位点恢复纳入。此规则统一应用四个任务，中心仍须满足岛叶要求，明确 CSF／非岛叶灰质与通道身份冲突沿用既有决定。Unknown 明确保留为 Unknown，不称作白质。本次仅恢复原生 aparc／aseg 均为 Unknown、ribbon=0，且无 MAPER 明确组织分类或身份／历史标签冲突的端点。

全队列恢复 **{len(restored)} 个采样位点**，其中 **{int(restored.sound_responsive.sum())} 个**已有 sound Repeat 显著响应证据；另 7 个仍缺完整 sound 统计。此前讨论的 8 个旧 sensory 位点均已恢复采样及显著名单。采样从 {len(previous)} 增至 **{len(sampled)} 位点／{summary['sampled_subjects']} 人**，显著从 {len(previous_positive)} 增至 **{len(positive)} 位点／{summary['responsive_subjects']} 人**。

{task_table}
各任务存在重复采样；ALL_UNIQUE 按参与者＋双极通道去重。当前采样数 348 与旧正文的“348 个显著位点”不是同一步统计。显著数仍是已确认阳性数，尚有 {summary['incomplete_response_sites']} 个采样位点缺完整 sound 响应证据，不能判为阴性。

## 5 个 Hammers=0、FreeSurfer 为岛叶灰质的具体标签

{table}
这些 0 触点的原生 aseg 均为对应侧 `Cerebral-Cortex`，ribbon 左侧为 3、右侧为 42。它们有 FreeSurfer 皮层／岛叶证据；Hammers 端点仍为 0。本轮未把它们归入 Unknown 恢复，仍单列等待纳入依据讨论。另一端一栏是 Hammers 标签，不代表各图谱在该触点必然一致；完整双端点 FreeSurfer 标签见 [逐端点表](evidence/unknown_endpoints_restored_2026-10-17/five_gray_sites_endpoint_labels.csv)。

## 输出同步

- Figure 1 和 Supplementary 参与者×任务清单采用同一份 {len(sampled)}／{len(positive)} 名单；平均曲线三阶段均为 {len(positive)} 个 sound 位点、{summary['responsive_subjects']} 位参与者。
- NMF 使用同一 sound 显著名单、三个阶段、k=3；拟合窗口仍为 Stimulus 0–0.6、Delay 0–1、Response −0.5–1 s，未改变。
- Figure 2 已基于更新后的分群重建，成分数目：{json.dumps(model['cluster_counts'],ensure_ascii=False)}。
- [旧版完整快照](evidence/before_unknown_endpoints_2026-10-17/archive_manifest.json)经过逐文件校验，未覆盖；原 insula-functional 输入保持只读。
- [恢复位点及响应／新分群](evidence/unknown_endpoints_restored_2026-10-17/restored_sites_and_response.csv)、[验证记录](evidence/unknown_endpoints_restored_2026-10-17/validation.json)。
'''
 (ROOT/'docs/unknown_endpoints_restored_2026-10-17.md').write_text(text)
 rank=pd.read_csv(ROOT/'results/nmf/rank_selection_metrics.csv').set_index('k')
 rank_three=rank.loc[3]
 highest=rank.cophenetic.idxmax() if 'cophenetic' in rank else None
 guide=f'''# Insula Paper：sound-only 三阶段 NMF

工作区 `/cwork/ns458/insula-paper`，分支 `codex/paper`。主分析遵循 [MAIN_ANALYSIS.md](MAIN_ANALYSIS.md)。

## 当前输入

- 采样 **{len(sampled)} 个位点／{summary['sampled_subjects']} 位参与者**；包括按用户最新决定保留并明确标记的 {len(restored)} 个 Unknown 端点位点。
- sound 阳性 **{len(positive)} 个位点／{summary['responsive_subjects']} 位参与者**；另 {summary['incomplete_response_sites']} 个缺完整 sound 统计、{summary['confirmed_nonresponsive_sites']} 个在完整记录下未显著。缺失不能判阴性。
- 筛选仅使用 sound Repeat 在 Stimulus 0–0.6、Delay 0–1、Response −0.5–0.5 s 开区间的显著 mask。
- Figure 1 保存的三个阶段 sound site means 完整覆盖全部 {len(positive)} 个阳性位点；没有 image/text 输入、缺段补零或旧 26 位点名单。
- 拟合窗口：Stimulus 0–0.6、Delay 0–1、Response −0.5–1 s，按闭区间取样，共399列（77、129、193）。
- 可用任务在位点内等权平均；负值截断后整行 L2 归一化。

## 重建顺序

1. `scripts/slurm/build_sound_only_cohort.sh`：重建 sound 采样／显著清单，明确标记保留的 Unknown 端点。首次本轮恢复使用 `scripts/slurm/restore_unknown_cohort.sh`，先保存旧版快照。
2. `scripts/slurm/execute_fig1_notebook.sh`：执行原 Figure 1 Notebook，更新 SVG 和 site means。
3. `scripts/slurm/nmf_three_phase_fit.sh`：同一 sound 名单，k=3，20 次初始化，归一化 H 后按 W 最大载荷赋标签。
4. `scripts/slurm/nmf_three_phase_rank.sh`：同一输入，k=2–6，每个 k 200 次 80% 子抽样；最高 cophenetic 不覆盖指定 k。
5. `scripts/slurm/execute_fig2_notebook.sh`：更新 Figure 2 并验证名单、模态、因子和标签。

## 当前结果

成分位点数：{json.dumps(model['cluster_counts'],ensure_ascii=False)}。20 次初始化中 {model['converged_starts']} 次收敛，平均初始化 ARI 为 {model['mean_initialization_ari']:.4f}。

k=3 cophenetic 为 **{rank_three.cophenetic:.4f}**、子抽样平均 ARI 为 **{rank_three.mean_ari:.4f}**；最高 cophenetic 对应 k={highest}。完整秩评估见 `results/nmf/rank_selection_metrics.csv`。本轮恢复、逐端点标签、旧版快照与验证见 [Unknown 恢复记录](unknown_endpoints_restored_2026-10-17.md)。先前 323／213 版本是恢复 Unknown 前的 sound-only 版本；227 位点混合模态及218位点诊断均为历史结果。
'''
 (ROOT/'docs/NMF_THREE_PHASE_WORKTREE.md').write_text(guide)
 print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
