# 主分析口径：Repeat + sound-only

**用户已明确：主分析仅使用 sound 条件。该要求同时约束有效／显著位点筛选、Figure 1、NMF 输入和依赖这些分群的后续主分析。**

## 数据条件

- 使用 `description=Repeat` 且 `modality=sound` 的记录。不能将 image/picture、text 或其他条件的显著性并入主分析的有效位点名单，也不能把它们的波形平均到主 NMF 输入中。
- PictureNaming 是任务名称，其中也有 sound 试次；按记录的 modality 筛选，不能直接删除整个 PictureNaming 任务。
- image/picture 和 text 用于单独的条件比较，不改变 sound 主分析的入组资格。
- 采样分母是具有 sound 记录且符合已确认中点、双端点解剖规则的双极位点。相同参与者、相同双极位点跨任务去重；逐参与者×任务的名单同时保存。
- **端点 Unknown 的最新决定：恢复纳入。**当端点 Hammers 为 0，原生 FreeSurfer 的 aparc／aseg 均为 Unknown、ribbon 为 0，且没有 MAPER 明确组织分类或标签／身份冲突时，按用户要求保留；清单标记 `has_accepted_unknown_endpoint` 和具体触点，不能称为已确认白质。中心仍需满足岛叶要求。明确 CSF、非岛叶灰质和坐标／植入版本冲突不因本例外恢复。Hammers 为 0、FreeSurfer 为岛叶灰质的另一类位点继续单列，等待对其具体标签的讨论。
- 显著响应是 sound Repeat 在 Stimulus **0<t<0.6 s**、Delay **0<t<1 s** 或 Response **−0.5<t<0.5 s** 任一阶段／任务的确认阳性。缺失 sound 统计不能用 image/text 结果填补，也不能判为阴性。
- NMF 仅从该 sound 阳性名单出发，再核对三个阶段的 sound 波形覆盖。缺整段不能补零、不能用其他模态波形代替；额外损失及原因须列出。
- NMF 拟合窗口：Stimulus **0–0.6 s**，Delay **0–1 s**，Response **−0.5–1 s**。Response **−0.5–1.5 s**是曲线显示上下文，不是拟合窗口。
- 同一 site/task 的 sound 波形先平均现有试次，同一位点内对可用任务等权平均。群体曲线先平均参与者内位点，再取参与者均值和 SEM；这一显示权重需在方法中明确，与旧直接按记录行平均有区别。

## 颜色约定

| 解剖区域 | 固定色值 |
|---|---|
| Insula | `#6A5ACD` |
| STG | `#20B2AA` |
| SMC | **`#F39B35`** |
| IFG | `#2369BD` |
| Other | `#D3D3D3` |

这些值来自原 insula-functional Figure 1 Notebook 及实际导出 SVG。区域类别需固定映射到颜色，不能因当前类别缺失或排序变化而移动配色。
NMF 功能成分沿用 sensory `#2369BD`、sustained `#C4A35A`、motor `#A9373B`；它们不代表 SMC 等解剖区域。

## 版本与当前修正

此前 323/227 名单及 Figure 1、两版三阶段 NMF 使用了跨模态显著并集或混合模态波形，不能作为符合本口径的正式主分析结果。旧归档保留作为审计依据。
当前唯一参数入口为 `src/main_analysis.py`；正式 sound-only 名单位于 `docs/evidence/sound_only_cohort_2026-10-17/`。修正时同步更新 supplementary 清单与 Figure 1，再从同一名单重建 NMF 和 Figure 2。
