# Unknown 端点恢复与 5 个灰质标签位点说明

## 本次决定与范围

用户要求把端点 Unknown 的位点恢复纳入。此规则统一应用四个任务，中心仍须满足岛叶要求，明确 CSF／非岛叶灰质与通道身份冲突沿用既有决定。Unknown 明确保留为 Unknown，不称作白质。本次仅恢复原生 aparc／aseg 均为 Unknown、ribbon=0，且无 MAPER 明确组织分类或身份／历史标签冲突的端点。

全队列恢复 **25 个采样位点**，其中 **18 个**已有 sound Repeat 显著响应证据；另 7 个仍缺完整 sound 统计。此前讨论的 8 个旧 sensory 位点均已恢复采样及显著名单。采样从 323 增至 **348 位点／65 人**，显著从 213 增至 **231 位点／62 人**。

| Task | 采样位点 | sound 显著位点 |
|---|---:|---:|
| PhonemeSequence | 173 | 104 |
| LexicalDelay | 214 | 126 |
| PictureNaming | 162 | 91 |
| SentenceRep | 56 | 28 |
| ALL_UNIQUE | 348 | 231 |

各任务存在重复采样；ALL_UNIQUE 按参与者＋双极通道去重。当前采样数 348 与旧正文的“348 个显著位点”不是同一步统计。显著数仍是已确认阳性数，尚有 110 个采样位点缺完整 sound 响应证据，不能判为阴性。

## 5 个 Hammers=0、FreeSurfer 为岛叶灰质的具体标签

| 双极位点 | Hammers=0 的触点 | FreeSurfer Destrieux 标签 | 另一端触点 | 另一端 Hammers 标签 |
|---|---|---|---|---|
| D0049_LAI1-2 | LAI1 | `ctx_lh_S_circular_insula_inf`（左岛叶下环沟） | LAI2 | `insula anterior long gyrus L` |
| D0085_RAI1-2 | RAI1 | `ctx_rh_G_insular_short`（右岛叶短回） | RAI2 | `insula posterior short gyrus R` |
| D0096_LTAS2-3 | LTAS3 | `ctx_lh_S_circular_insula_inf`（左岛叶下环沟） | LTAS2 | `insula posterior long gyrus L` |
| D0103_LTMS1-2 | LTMS2 | `ctx_lh_S_circular_insula_inf`（左岛叶下环沟） | LTMS1 | `insula posterior long gyrus L` |
| D0118_RTAS2-3 | RTAS3 | `ctx_rh_G_Ins_lg_and_S_cent_ins`（右岛叶长回及中央沟） | RTAS2 | `insula posterior long gyrus R` |

这些 0 触点的原生 aseg 均为对应侧 `Cerebral-Cortex`，ribbon 左侧为 3、右侧为 42。它们有 FreeSurfer 皮层／岛叶证据；Hammers 端点仍为 0。本轮未把它们归入 Unknown 恢复，仍单列等待纳入依据讨论。另一端一栏是 Hammers 标签，不代表各图谱在该触点必然一致；完整双端点 FreeSurfer 标签见 [逐端点表](evidence/unknown_endpoints_restored_2026-10-17/five_gray_sites_endpoint_labels.csv)。

## 输出同步

- Figure 1 和 Supplementary 参与者×任务清单采用同一份 348／231 名单；平均曲线三阶段均为 231 个 sound 位点、62 位参与者。
- NMF 使用同一 sound 显著名单、三个阶段、k=3；拟合窗口仍为 Stimulus 0–0.6、Delay 0–1、Response −0.5–1 s，未改变。
- Figure 2 已基于更新后的分群重建，成分数目：{"motor": 129, "sustain": 67, "sensory": 35}。
- [旧版完整快照](evidence/before_unknown_endpoints_2026-10-17/archive_manifest.json)经过逐文件校验，未覆盖；原 insula-functional 输入保持只读。
- [恢复位点及响应／新分群](evidence/unknown_endpoints_restored_2026-10-17/restored_sites_and_response.csv)、[验证记录](evidence/unknown_endpoints_restored_2026-10-17/validation.json)。
