# 参与者 × 任务：采样与显著位点清单

计数单位为符合当前解剖纳入规则的岛叶双极位点。显著位点按已约定的 Repeat 三阶段窗口合并模态，取现有核实 mask 的阳性证据。

| 任务 | 采样参与者数 | 采样位点数 | 显著参与者数 | 已确认显著位点数 |
| --- | --- | --- | --- | --- |
| PhonemeSequence | 36 | 161 | 32 | 98 |
| LexicalDelay | 38 | 199 | 36 | 115 |
| PictureNaming | 28 | 151 | 28 | 102 |
| SentenceRep | 16 | 53 | 13 | 26 |
| ALL_UNIQUE | 65 | 323 | 64 | 227 |

逐任务数量有跨任务重复；ALL_UNIQUE 按参与者＋双极名称取并集。

完整长表覆盖 **168 条参与者×任务记录**；65 位纳入采样队列参与者另有宽表。每条计数均附对应采样和显著位点名称，任务汇总另列参与者 ID。

## 文件

- [完整工作簿](evidence/subject_task_site_lists_2026-10-17/参与者任务_采样与显著位点清单.xlsx)
- [参与者×任务计数及名单](evidence/subject_task_site_lists_2026-10-17/subject_task_counts_and_lists.csv)
- [65 位参与者宽表](evidence/subject_task_site_lists_2026-10-17/subject_task_counts_wide.csv)
- [逐任务采样位点](evidence/subject_task_site_lists_2026-10-17/sampled_sites_by_subject_task.csv)
- [逐任务显著位点](evidence/subject_task_site_lists_2026-10-17/responsive_sites_by_subject_task.csv)
- [跨任务去重的新 NMF 候选名单](evidence/subject_task_site_lists_2026-10-17/nmf_candidate_sites_not_final_inputs.csv)

## 使用说明

- 采样数量和名单已按当前解剖规则列齐。0 表示有该任务记录但没有当前纳入位点；宽表空白表示缺少该参与者×任务的现存坐标记录。
- 显著列为当前**已确认**阳性数量。94 个位点尚缺完整响应证据，未据此判为阴性，因此该列暂不能作为完整队列的最终显著总数。工作簿保留此备注。
- 227 个跨任务去重显著位点可用作新 NMF 的候选筛选名单；正式拟合还须检查拟采用任务／模态的三阶段波形覆盖。此处不把它写成已经确认的最终拟合人数。
- 本轮只汇总现有审核结果。Figure 1、NMF 模型和旧输入均未改动；输入及受保护文件哈希核对通过。
