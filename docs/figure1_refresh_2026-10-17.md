# Figure 1 Notebook 更新

> **历史混合模态版本，不能作为正式主分析。** 用户已明确主分析筛选及波形均限 sound Repeat。当前口径见 [MAIN_ANALYSIS.md](MAIN_ANALYSIS.md)，修正结果见 [sound-only 修正记录](sound_only_correction_2026-10-17.md)。原 evidence 目录保留用于追溯。

主入口：[vizpub/fig1.ipynb](../vizpub/fig1.ipynb)。图中内容和生成代码均保留在 Notebook 中；辅助脚本只负责执行与输出核对。

## 已更新的内容

| 内容 | 当前来源／口径 | 图件 |
|---|---|---|
| 任务、参与者与位点数量 | 与 supplementary information 共用参与者×任务清单；跨任务去重 323 个采样位点／65 人，227 个已确认显著位点／64 人 | `fig1_subject_task_counts.svg` |
| 全脑采样覆盖中的岛叶标记 | 按已审核的中点、双端点规则标记岛叶；排除噪声参与者 D0121 | `fig1_all_electrodes_projected.svg` |
| 双侧岛叶位点分布 | 显示全部 323 个采样位点，标记其中 227 个已确认响应位点；取消旧的显示排除名单；灰点不解释为已确认阴性 | `fig1_insula_electrodes_gyrus_lines.svg` |
| 左右侧数量 | 左侧 193 个采样／146 个已确认响应；右侧 130／81；采用计数图，删除将缺失证据当阴性的 Fisher 检验 | `fig1_insula_laterality_counts.svg` |
| 分阶段响应和 ROI 组成 | Stimulus 0–0.6 s：115 个；Delay 0–1 s：132 个；Response −0.5–0.5 s：182 个；三阶段并集 227 个 | 分阶段饼图、ROI 饼图和比率图，共 5 个 SVG |
| 群体 HGA 均值 | 直接选择 Figure 1 的已确认响应位点，独立于 NMF 标签；三阶段均覆盖 227 位点／64 人 | `fig1_insula_average_traces.svg` |
| 原生电极示意 | D0064 LAI，标记当前纳入的 LAI3-4、LAI4-5、LAI5-6、LAI6-7；灰色物理触点与紫色虚拟双极中点区分显示 | 两个 `fig1_insula_electrode_schematic*.svg` |
| TFR 示例 | 原 D0088_LAI3-4 中点为 0，当前待核对；换成通过解剖审核且有响应证据的 D0064_LAI3-4，使用已有三阶段 TFR，不重做统计 | 两个 `fig1_example_electrode_tfr*.svg` |

## 均值和统计解释

- 群体均值先合并同一位点内现有模态及任务，再在参与者内平均位点，最后显示参与者均值的 mean ± SEM。每个阶段使用共同时间网格，避免仅较长 epoch 提供末端样本。均值使用完整已有波形，不截断为逐时刻显著片段。
- 旧打包遗漏的部分已有波形从现存完整 zscore epochs 读取并平均；补充来源和逐位点／任务／模态名单已保存。三阶段均没有缺少波形的已确认响应位点。
- 显著性仍是当前可确认的阳性证据：94 个位点的响应证据不完整，另有 2 个完整评估但未响应。因此图件使用 “confirmed responsive”，不将其余采样点全部称为不显著。
- ROI 饼图描述当前识别出的响应；响应数／采样数图采用可用证据的描述性比率，不据此推断最终响应概率。其他 ROI 保留原区域 QC，并排除 D0121 及未纳入的岛叶候选，避免把待核对岛叶位点改称其他皮层。
- 原生脑表面的 Destrieux 标注只提供显示背景；位点是否纳入始终由统一 Hammers 清单决定。没有用别的 atlas 替代已缺失的个体 MAPER 轮廓。

## 输出与复核

- [14 个 SVG 图件](evidence/figure1_refresh_2026-10-17/figures/)
- [图件及输入清单](evidence/figure1_refresh_2026-10-17/figure_manifest.json)
- [实际运行核对](evidence/figure1_refresh_2026-10-17/execution_validation.json)
- [群体均值位点数量](evidence/figure1_refresh_2026-10-17/average_hga_counts.csv)
- [左右侧计数](evidence/figure1_refresh_2026-10-17/insula_laterality_counts.csv)
- [ROI 采样及已确认响应数量](evidence/figure1_refresh_2026-10-17/roi_sampled_and_confirmed_counts.csv)

Notebook 的 22 个代码单元完整执行，生成 14 个 SVG，无执行错误；已检查计数表、岛叶图、均值和原生示意图的显示。旧工作区的 Notebook 和 NMF 文件哈希不变。本轮没有拟合 NMF，也没有重算显著性／TFR。

运行脚本：`scripts/slurm/execute_fig1_notebook.sh`。D64 的 6 个表面／标注文件已通过 Globus 恢复。首次执行修正了保存 epochs 的读取方式，随后修正图例和脚注遮挡；最终执行作业为 **58868840**。正文中的旧样本数及占比需与最终确认的分析队列同步，本轮图件保留上述证据范围说明。
