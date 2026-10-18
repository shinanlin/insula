# Sound-only 主分析纠正：Figure 1、NMF 与 Figure 2

**历史版本说明：**本文记录恢复 Unknown 端点之前的 323／213 sound-only 版本。用户随后要求恢复 Unknown；最新名单和结果见 [Unknown 恢复记录](unknown_endpoints_restored_2026-10-17.md)，恢复前文件见 `evidence/before_unknown_endpoints_2026-10-17/`。下文数量和原文件哈希用于追溯该历史版本。

用户明确主分析及有效位点筛选仅使用 **sound**。前次把 image/text 并入有效位点、波形及主 NMF 是方法错误，不能作为正式主分析。
已将该约束写入 [MAIN_ANALYSIS.md](MAIN_ANALYSIS.md)、根目录 `AGENTS.md`、项目文档入口及共享参数 `src/main_analysis.py`，并在原 Figure 1、Figure 2 Notebook 中落实。

## 1. 统一后的条件与完整筛选链

1. 选取四个任务中的 **Repeat + sound** 记录；PictureNaming 中的 sound 试次保留，image/text 试次只用于单独比较。
2. 应用此前约定的解剖规则：中点岛叶灰质；端点岛叶灰质或确认白质；组织／身份不明暂列。D0121 排除，不自动沿用旧 26 位点排除名单。
3. sound Repeat 显著性使用开区间 Stimulus 0–0.6 s、Delay 0–1 s、Response −0.5–0.5 s，取阶段／任务并集。
4. sound 阳性名单再检查三个阶段的 sound 波形覆盖；全部 **213** 个都完整，未发生额外缺段排除。
5. NMF 直接读取 Figure 1 保存的 sound site means；拟合窗口为 0–0.6 s、0–1 s、−0.5–1 s，共 399 列。显示曲线的 Response 上下文为 −0.5–1.5 s。

| 项目 | 前次本工作区混合模态版本 | 正式 sound-only 版本 |
|---|---:|---:|
| 解剖合格、有 sound 采样的位点 | 323 | **323** |
| 采样参与者 | 65 | **65** |
| 确认显著响应位点 | 227 | **213** |
| 确认响应参与者 | 64 | **61** |
| Stimulus 显著位点 | 115 | **104** |
| Delay 显著位点 | 132 | **120** |
| Response 显著位点 | 182 | **167** |
| 正式 NMF 输入 | 227（混合模态，作废） | **213 sound 阳性位点** |

全部 323 个采样位点在 sound 坐标记录中都存在，因此采样分母没有变化；这不代表此前混合模态的显著筛选正确。
当前剩余 **103 个位点缺完整 sound 统计证据，7 个在现有记录下完整未显著**。213 是已确认的阳性数量，不能把余下 110 个全部叫作阴性。

前次 227 个中有 **14 个没有确认的 sound 阳性**，只因其他模态进入了原名单，现已移出主分析；其中 11 个 sound 资料不完整、3 个 sound 未显著。名单及具体状态：[excluded_multimodal_only_sites.csv](evidence/sound_only_correction_2026-10-17/excluded_multimodal_only_sites.csv)。

此前诊断中的 **218** 是“当前旧名单里有完整 sound 波形”的数量；它没有重新按 sound 显著性入组。这 218 中只有 213 符合主分析阳性标准，不能把 218 直接作为正式 NMF 人数。

## 2. 每个任务、参与者和位点的清单

| 任务（均仅 sound Repeat） | 采样参与者 | 采样位点 | 响应参与者 | 确认响应位点 |
|---|---:|---:|---:|---:|
| PhonemeSequence | 36 | 161 | 32 | 98 |
| LexicalDelay | 38 | 199 | 36 | 115 |
| PictureNaming | 28 | 151 | 25 | 82 |
| SentenceRep | 16 | 53 | 13 | 26 |
| 跨任务去重 | **65** | **323** | **61** | **213** |

逐任务位点和参与者有重叠，不能直接相加。

- [参与者×任务计数及完整名单](evidence/sound_only_cohort_2026-10-17/subject_task_counts_and_lists.csv)
- [参与者宽表](evidence/sound_only_cohort_2026-10-17/subject_task_counts_wide.csv)
- [逐任务采样名单](evidence/sound_only_cohort_2026-10-17/sampled_sites_by_subject_task.csv)
- [逐任务 sound 阳性名单](evidence/sound_only_cohort_2026-10-17/responsive_sites_by_subject_task.csv)
- [跨任务去重的 213 位点阳性名单](evidence/sound_only_cohort_2026-10-17/nmf_candidate_sites_not_final_inputs.csv)
- [完整 supplementary 工作簿](evidence/sound_only_cohort_2026-10-17/参与者任务_采样与显著位点清单.xlsx)

名单生成阶段仍将这 213 个称为 NMF 候选；后续覆盖验证及正式模型确认全部 213 个进入拟合。

## 3. Figure 1 哪些地方错了，如何修正

| 内容 | 前次问题 | 当前处理 |
|---|---|---|
| 显著位点名单、图中响应点标记 | 纳入了其他模态的阳性 | 改为 sound 213 位点 |
| 阶段及 ROI 响应数量、组成和比率 | raw mask 及岛叶清单含其他模态阳性 | 全部按 sound 重建 |
| 群体平均 HGA | 合并 sound/image/text 波形 | 只读取 sound Repeat，三阶段各 213 位点／61 人 |
| 任务与 supplementary 计数表 | 共享的阳性清单是跨模态并集 | 改用同一 sound-only 清单 |
| 采样数、坐标和原生示意 | 解剖队列本身并非该模态错误来源 | 确认全部 323 有 sound 采样，保留既定解剖规则 |
| TFR 示例 | D0064 PhonemeSequence 已属 sound | 保留原统计产品，重新核对该位点在 sound 阳性清单中 |

平均 HGA 仍先在位点内平均可用 sound 任务，再平均参与者内位点，最后显示参与者均值和 SEM；不再平均模态。原始 sound 波形不按逐点 mask 截断。

### 指定的区域响应比例图

文件：`img/fig1/fig1_roi_significant_yield.svg`。

| 区域 | 采样位点 | 前次混合模态确认响应 | 当前 sound 确认响应 | 当前描述比例 |
|---|---:|---:|---:|---:|
| Insula | 323 | 227 | **213** | 65.94% |
| STG | 491 | 391 | **383** | 78.00% |
| SMC | 448 | 327 | **309** | 68.97% |
| IFG | 237 | 147 | **136** | 57.38% |
| Other | 3652 | 1921 | **1769** | 48.44% |

这些是基于可用阳性证据的描述比例，不是缺失统计均被确认阴性后的最终响应概率。完整表见 [ROI 数量](evidence/figure1_sound_only_2026-10-17/roi_sampled_and_confirmed_counts.csv)。

## 4. SMC 颜色核查

对照了原 insula-functional 工作区和当前工作区的**实际指定 SVG**，SMC 柱子填色均为 **`#F39B35`**，透明度／边线样式也一致；没有在这两个文件中找到 SMC 色值不同的证据。核对记录：[SMC_export_color_audit.json](evidence/sound_only_correction_2026-10-17/SMC_export_color_audit.json)。

但原 `docs/PLOTTING_STYLE.md` 的两个示例错误地把 SMC 写成 `#6A5ACD`（紫色）、insula 写成金色，与 Figure 1 不一致。现已纠正并明确以 `src/main_analysis.py::ROI_COLORS` 为单一入口，按区域名映射：

- Insula `#6A5ACD`；STG `#20B2AA`；**SMC `#F39B35`**；IFG `#2369BD`；Other `#D3D3D3`。
- NMF 的 sensory／sustained／motor 是功能成分，使用另一套已约定颜色，不以 SMC 的区域色代替 motor 成分色。
- Figure 1 生成时额外检查指定 SVG 的实际 SMC 填色，不能只检查变量声明。

## 5. 同步修正 NMF 与 Figure 2

主模型现在是 **213×399 的纯 sound 输入**，对应 61 位参与者。三阶段波形与 Figure 1 使用同一来源，参与者均值曲线数值核对一致。

| 成分 | 位点 | 参与者 |
|---|---:|---:|
| sensory | **29** | 19 |
| sustained | **61** | 32 |
| motor | **123** | 53 |
| 合计 | **213** | **61（去重）** |

20 次初始化均收敛且标签一致。重建误差 7.481539，解释能量 73.72%；子抽样下 k=3 cophenetic 0.981514、平均 ARI 0.807973。最高 cophenetic 仍是 k=2（0.987056），不能写成选秩过程选择了 k=3。

用户关心的 D0125_LTMS1-2、D0125_LTMS2-3 在本次正式 sound-only 模型中均为 sensory。解剖待定、CSF 及缺少 sound 显著性证据的其他位点未因恢复 sound 输入而自动恢复入组。

Figure 2 的全部载荷、波形、热图、PCA/k-means、空间／密度图、Hammers 构成及左右覆盖汇总均从同一新模型重新生成。

## 6. 验证与版本

| 阶段 | 作业 | 验证结果 |
|---|---|---|
| sound-only 名单与 supplementary 导出 | 58870696 | 条件严格为 sound Repeat；323→213；参与者／任务／位点清单一致 |
| Figure 1 完整执行 | 58870704 | **22 个代码单元，14 张 SVG**；source membership 全 sound；SMC 实际填色验证 |
| NMF | 58870730 | 213 位点、61 人、399 列；Figure 1 输入一致；W/H 与最大载荷标签吻合 |
| 秩检验 | 58870738 | 同一 sound 矩阵，k=2–6，每个 k 200 次 80% 子抽样 |
| Figure 2 完整执行 | 58870776 | **19 个代码单元，16 张 SVG**；sound 入组与波形来源严格验证 |

Figure 1 的说明文字另已同步到 213 位点及 sound-only 口径；执行核对中的 Notebook 哈希随此文档同步更新，代码单元及数值输出不变。

- [Figure 1 执行核对](evidence/figure1_sound_only_2026-10-17/execution_validation.json)
- [Figure 2 执行核对](evidence/figure2_sound_only_2026-10-17/execution_validation.json)
- [Figure 1 完整输出](evidence/figure1_sound_only_2026-10-17/)
- [正式 sound-only NMF 与 Figure 2 输出](evidence/figure2_sound_only_2026-10-17/)

原工作区文件及旧混合模态 evidence 归档保留；历史报告已明确标为不适用正式主分析。Figure 3 以后的分析还没有重新生成，不能直接沿用旧标签作为本次 sound-only 结果。
