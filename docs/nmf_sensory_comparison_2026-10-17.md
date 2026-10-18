# 旧版强 sensory 位点去了哪里：对照核查

对照工作区：`/hpc/group/coganlab/nanlinshi/insula-functional`（旧 255 位点，四阶段）；
当前工作区：`/cwork/ns458/insula-paper`（227 位点，三阶段）。

当前 NMF 实际拟合的 Response 是 **−0.5–1 s**；**−0.5–1.5 s 是 HGA 显示范围**。
本次只生成诊断结果，现有 NMF、Figure 1、Figure 2 和旧工作区文件的哈希均未改变。
完整数据见 [evidence 目录](evidence/nmf_sensory_comparison_2026-10-17/)。

## 1. 结论

旧版 sensory 组的原始 HGA 均值峰值（Stimulus 0–0.6 s）为 **0.715 z**，当前为 **0.251 z**，下降确实存在。
追踪发现：旧 sensory 39 个位点中只有 19 个进入当前 NMF；当前 sensory 44 个位点里，只有 17 个来自旧 sensory，另有 20 个来自旧 sustained、2 个来自旧 motor、5 个不在旧 255 名单。新旧 sensory 的成员已大幅变化。

前次重建还把旧版 **sound-only** 输入改成 **sound/image/text 汇总**，并调整了平均方式。这些方法变更与解剖筛选、响应筛选、删除 Go 和裁窗同时发生，因此新旧结果差异不能只归因于 Response 窗口。

## 2. 旧 39 个 sensory 位点的具体去向

| 当前状态 | 位点数 | 解释 |
|---|---:|---|
| 仍是 sensory | **17** | 当前 44 个 sensory 位点的一部分 |
| 转为 motor | **2** | D0125_LTMS1-2、D0125_LTMS2-3 |
| 解剖待核，暂未纳入 | **15** | 13 个端点为 0、组织证据不足或冲突；2 个跨任务原生坐标冲突 |
| 解剖排除 | **1** | D0125_LTMS3-4：端点 CSF |
| 解剖合格，未通过当前响应纳入 | **4** | 3 个响应资料不完整；1 个当前 Repeat 三窗口未显著 |
| 合计 | **39** | 当前纳入 19，未纳入 20 |

逐一清单：[旧 sensory 位点核对表](evidence/nmf_sensory_comparison_2026-10-17/旧sensory位点逐一核对.csv)。
包含未四舍五入数据、当前输入阶段能量和旧标签的技术表：[old_sensory_sites_trace.csv](evidence/nmf_sensory_comparison_2026-10-17/old_sensory_sites_trace.csv)。

### 旧 sensory 里峰值最强的 8 个位点

以下数值均为旧 sound 波形在 0–0.6 s 内的峰值，不是 NMF H 的系数。

| 位点 | 原 sound 峰值（z） | 当前去向 |
|---|---:|---|
| D0125_LTMS3-4 | **4.242** | 端点 CSF，排除 |
| D0125_LTMS2-3 | **3.370** | 纳入，分到 motor |
| D0107_RTAS1-2 | **2.943** | 跨任务原生位置冲突，待核 |
| D0132_LTMS3-4 | **2.305** | 端点 0，组织待核 |
| D0086_RTPS4-9 | **1.990** | 跨任务原生位置冲突，待核 |
| D0049_LAI1-2 | **1.706** | 端点 0，组织待核 |
| D0115_LTMS1-2 | **1.185** | 端点 0，组织待核 |
| D0125_LTMS1-2 | **1.154** | 纳入，分到 motor |

这 8 个目前均不在当前 sensory 组：5 个待核、1 个解剖排除、2 个转群。它们的原始 sound 响应并未因 NMF 重跑而被删除。

响应标准未通过的 4 个位点为：D0119_RIP1-2、D0090_RIP2-3、D0103_LAI1-2（资料不完整），以及 D0143_RTMS1-2（现有 Repeat 三窗口未显著）。旧 NMF 中存在某个位点，并不能证明它符合当前 Repeat 三窗口的阳性标准。

## 3. 固定名单，单独改变输入模态

从当前 227 个位点中，找到 sound 在三个拟合阶段都完整的 **218 个共同位点**。在这同一批位点上比较：

- sound-only；
- sound/image/text 按当前方式先平均模态、再平均任务。

两次拟合的裁窗、特征网格（399 列）、负值截断、L2 归一化、k=3、20 次初始化、种子和命名规则完全相同。

| 输入 | sensory | sustained | motor |
|---|---:|---:|---:|
| sound-only | 31 | 63 | 124 |
| 合并模态 | 29 | 60 | 129 |

**17/218 个位点换群，ARI=0.8153。** 这直接说明模态汇总本身足以改变标签。

两个具体例子：

| 位点 | sound 原始峰值 | 合并模态后峰值 | sound-only 对照标签 | 合并模态对照标签 |
|---|---:|---:|---|---|
| D0125_LTMS2-3 | **3.370 z** | **1.326 z** | sensory | motor |
| D0125_LTMS1-2 | **1.154 z** | **0.435 z** | sensory | motor |

这两次受控拟合中，19 个旧 sensory 位点仍在共同名单里：sound-only 下 18 个为 sensory、1 个为 sustained；合并模态下 17 个为 sensory、2 个为 motor。

对照名单及载荷：[sound_only_assignments.csv](evidence/nmf_sensory_comparison_2026-10-17/sound_only_assignments.csv)、[all_modalities_assignments.csv](evidence/nmf_sensory_comparison_2026-10-17/all_modalities_assignments.csv)。
逐位点标签变化：[modality_control_label_comparison.csv](evidence/nmf_sensory_comparison_2026-10-17/modality_control_label_comparison.csv)。
这两个 218 位点模型是诊断用对照，没有替换当前 227 位点主模型。

## 4. 样本组成也存在真实敏感性

把 227 位点的合并模态模型与 218 位点的合并模态模型比较；原始共有样本、模态、窗口、预处理、求解器、初始化和命名均相同，仅移除 9 个无完整 sound 输入的位点：

- 218 个共同位点中，**21 个换群**；ARI=**0.7817**。
- sensory 总数由全体模型 44 个变为 218 位点模型 29 个；这个总数变化也包含被移除位点，不能直接解释为 15 个共同位点换群。

因此，20 次初始化标签一致并不能代表对样本组成稳健。它与前次报告的子抽样稳定性下降是一致的现象。
详见 [cohort_control_summary.json](evidence/nmf_sensory_comparison_2026-10-17/cohort_control_summary.json)。

## 5. 平均方式与 H 的数值尺度

在相同的旧 39 个 sensory 位点、相同 sound 数据上：

| 原始 HGA 的平均方式 | Stimulus 均值峰值 |
|---|---:|
| 旧 Notebook：按记录行平均 | 0.7147 z |
| 位点等权 | 0.7280 z |
| 先在参与者内平均，再参与者等权 | 0.6889 z |

平均方式有影响，但这里的差异远小于整体从 0.715 降到 0.251 的变化。
[顺序比较图](evidence/nmf_sensory_comparison_2026-10-17/sensory_peak_averaging_comparison.svg)和[曲线数据](evidence/nmf_sensory_comparison_2026-10-17/averaging_comparison_curves.csv)供核对。
图中由旧 39 位点切到当前保留位点的步骤，同时包含新的源数据补充方式；这些顺序曲线不是独立效应的可相加分解。

**NMF 的 H 与原始 HGA z 值需分开解读：**旧、新 H 都被完整时间轴 L2 归一化至 1。旧 sensory H 的 Stimulus 峰值为 0.1731，当前为 0.1371；相同 0–0.6 s 区间的 H 能量占比从 93.2% 变成 73.7%，表明当前 sensory 模板还分配了更多能量到其他阶段。H 的高度不是电极 HGA 的绝对强度。

此外，旧拟合为四阶段、444 列（Stimulus 127、Delay 127、Go 127、Response 63），当前为三阶段、399 列（77、129、193）。Response 时间点占比从 14.2% 增为 48.4%，同时 Stimulus 由 0–1 s 缩为 0–0.6 s、Go 被删除。全行归一化没有把各阶段额外等权，这也是模型定义的变化；本次并未以额外拟合把这些因素逐一分解。

## 6. 后续处理建议

1. **先对齐 NMF 的发现输入。** 如果目标是更新原来的听觉重复分析，建议恢复 sound-only 定义；当前具备三阶段完整 sound 输入的 218 位点对照已可审阅。另 9 个只在其他模态有完整输入的位点需要明确列为覆盖差异，而不是默默补零或合并。
2. **补核旧 sensory 的 15 个解剖待定位点。** 高峰值位点优先关注 D0132、D0049、D0115 的端点组织，以及 D0107、D0086 的跨任务位置/植入身份冲突。组织待定不等于信号质量差，也不等于已经证明属于岛叶；CSF 排除的位点单独处理。
3. **固定位点、模态、汇总方式和成分尺度，再比较裁窗。** 主模型要回答什么条件下的时间组织应先明确，避免将名单变化、模态变化和窗口变化一起解释成 NMF 的随机不稳定。

## 7. 执行与证据

- 位点、原始峰值、平均方式追踪：Slurm **58870361**，`scripts/slurm/diagnose_nmf_sensory_change.sh`。
- 同 218 位点的模态受控拟合：Slurm **58870384**，`scripts/slurm/compare_nmf_modality_inputs.sh`。
- 样本组成比较使用上述已保存的模型和当前模型，不增加 NMF 拟合。
- 当前完整三阶段原始输入由模态源波形重建后，与主模型逐点一致（绝对容差 `1e-12`）。当前/旧 NMF 及 Figure 1、Figure 2 文件哈希未变。
- 本轮只提交诊断脚本和证据；主分析仍需以上述输入定义和解剖核对为依据推进。
