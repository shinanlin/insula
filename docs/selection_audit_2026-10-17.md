# Insula 采样位点与 NMF 筛选核对

日期：2026-10-17。数据来自原工作区的现有 HGA 打包文件与已保存 NMF 表；结果保存于新工作区。**本次没有拟合 NMF，也没有修改原始结果、Notebook 或 manuscript。**

## 结论

- Figure 1 的 **464 个采样位点**、**348 个响应位点**均可从现有数据精确复现。
- 旧 NMF 的 **255 个位点**也可精确复现，名单与已保存 assignments、rank-selection metadata 完全一致，输入尺寸为 **255 × 444**。
- **255 不是 348 的子集**。实际关系是：**255 = 238 个 Figure 1 响应位点 + 14 个在 464 内但不满足 Figure 1 响应定义的位点 + 3 个在 464 外的跨任务解剖冲突位点**。

## 1 采样口径与正文的 348

### 464 的定义

按 PhonemeSequence → LexicalDelay → PictureNaming → SentenceRep 的顺序读取坐标表，保留具有 x/y/z 坐标的记录；相同 `channel` 保留第一条记录；再按原规则筛选 `roi ∈ {AIC, PIC}` 且 `mix=False`。端点标签为 0 不直接导致排除；此处也不额外排除中心 `label=0`。

结果：**464 个双极位点，67 位参与者；左侧 285，右侧 179**。跨任务重复记录按带被试编号的完整双极位点名去重。

这复现的是现有 Figure 1 的“先保留第一条坐标记录”规则。若改成“任一任务记录符合岛叶规则就保留”，会得到 472；本次没有改动此规则。

### 348 的定义

在上述 464 中，保留 `description=Repeat`，合并各输入模态（当前文件也包括 image、text）；任意一个以下开区间中存在 `mask=True` 的采样点，即记为响应位点：

| 阶段 | Figure 1 相对事件时间窗 | 响应位点数 |
|---|---|---:|
| Stimulus | 0 < t < 1 s | 217 |
| Delay | 0 < t < 1 s | 199 |
| Response | −0.5 < t < 0.5 s | 273 |
| 三阶段去重并集 | 任一阶段满足 | **348** |

三个阶段的名单有重叠，不能相加。348 个位点来自 66 位参与者，左侧 222、右侧 126，占 464 的 **75%**。348 个位点均至少包含一个正值的显著 HGA 采样点。若仅计 sound 模态的显著响应，得到 **323**；这与“有 sound 波形可供 NMF 使用”是不同条件。

因此，Google Doc 正文中的 “Of the 464 insular sites, 348 (75%) … during at least one task phase” 对应的是 **Figure 1 的 Repeat 三阶段响应并集**。正文目前没有把条件、合并模态及具体时间窗写全。348 不对应旧 NMF 的预筛选池，也不对应 Figure 1 另一条使用中心标签非零规则的 ROI 统计路径。

464 → 348 共排除 116 个：

- **82 个**没有进入现有 Repeat 打包 HGA 候选池；这一项记录文件层面的缺失，不能单凭它进一步断定原始数据从未显著。
- **34 个**有 Repeat 打包波形，但 Figure 1 指定的条件与三个窗口中没有显著 mask。

## 2 从 348 个响应位点追踪旧 NMF 资格

下表逐步累积筛选，每个位点只计入其第一次被排除的步骤。

| 步骤 | 本步排除 | 剩余 | 原因 |
|---|---:|---:|---|
| Figure 1 三阶段响应并集 | — | **348** | 来自 464 个采样位点 |
| 有 Repeat、sound 打包 HGA | 15 | **333** | 这 15 个只有其他模态的 Repeat 打包数据 |
| sound 记录仍满足岛叶且 mix=False | 6 | **327** | 坐标表与 sound 波形所附解剖记录的资格不同 |
| 中心 label 非 0 | 69 | **258** | 旧 NMF 额外执行了 Figure 1 采样口径没有的中心标签过滤 |
| 删除旧 26 位点排除名单 | 20 | **238** | 26 个中有 20 个属于上述 348；另 6 个不属于 348 |
| 四阶段存在、每阶段覆盖率 ≥95%、整流后范数非零 | 0 | **238** | 这些后续条件没有再减少位点 |

六个因 sound 解剖记录不符合规则而被排除的响应位点：

`D0086_LTPI1-2`、`D0086_LTPI2-3`、`D0086_RTMM5-6`、`D0086_RTPM4-5`、`D0086_RTPM5-6`、`D0107_RIA6-7`。

**这里的 69 是中心标签为 0；端点标签为 0 本身并没有被 NMF 单独排除。** 若今后统一执行 Figure 1 的原始 mix=False 规则，这一额外过滤需要明确处理。本次只记录差异，未更改纳入规则。

## 3 旧 NMF 实际使用的完整入口

NMF 直接从打包 HGA 中寻找 Repeat、sound、岛叶且 mix=False 的记录。它没有先与 Figure 1 的 348 名单相交。打包流程的入口是每个任务内、按参与者 × 模态合并**各条件、各阶段显著位点的并集**；随后从 z-score 数据中提取这些位点的各阶段波形。波形载入时不再要求其 Repeat 三阶段窗口内显著。

### 所有任务记录中的旧 NMF 入口

| 步骤 | 位点数 | 本步排除 |
|---|---:|---:|
| Repeat、sound、岛叶且 mix=False | **355** | — |
| 排除参与者 D0121 | **355** | **0**：本次入口中没有该被试的合格位点 |
| 排除中心 label=0 的记录 | **281** | **74** |
| 删除旧 26 位点名单 | **255** | **26** |
| 按旧四阶段裁窗、取完整阶段交集、覆盖率检查、整流后非零 | **255** | **0** |

355 中有 349 个在 Figure 1 的 464 内；281 中有 278 个在 464 内；最终 255 中有 252 个在 464 内。与历史日志 `nmf_concat_51537208.out` 记录的 **281 → 255** 完全一致。

旧 NMF 裁窗：Stimulus、Delay、Go 均为 0 < t < 1 s，Response 为 0 < t < 0.5 s。按 channel × time 对跨任务记录取均值，四阶段分别检查有效时间覆盖率 ≥95%，插值后拼接，再截断负值并按整行 L2 范数归一化。本次仅构建输入和检查资格，未调用拟合。

### 仅在 464 内追踪旧 NMF 入口

| 步骤 | 剩余 | 本步排除 |
|---|---:|---:|
| Figure 1 采样位点 | 464 | — |
| 进入现有 Repeat 打包候选池 | 382 | 82 |
| 有 sound 数据 | 356 | 26 |
| sound 记录满足岛叶且 mix=False | 349 | 7 |
| 中心 label 非 0 | 278 | 71 |
| 删除旧 26 位点名单 | 252 | 26 |
| 四阶段、覆盖率、非零检查 | 252 | 0 |

上表 7 个解剖资格差异位点中，除前述 6 个响应位点，还包括 `D0086_LTMM3-4`。这条入口合计排除 212 个位点；最终再加上 3 个 464 外的旧 NMF 位点，构成历史 255。

## 4 NMF 中额外的 14 个与 3 个

### 14 个位于 464 内但不满足 Figure 1 响应定义的位点

对这 14 个位点涉及的 306 份各条件 HGA 文件进行了补充核对。**14 个均有其他条件下的 sound 显著响应**，其中 5 个也在 Repeat 的 Go 阶段显著。另 9 个在现有 Repeat/sound 表的所有阶段、完整保存时间范围内均没有显著 mask。它们可通过旧的跨条件显著并集进入打包候选池。

| 位点 | 在 Repeat/sound 的 Go 阶段显著 | 存在显著记录的条件，合并模态 |
|---|---|---|
| D0022_LMIF3-4 | 否 | Passive |
| D0028_LPI1-2 | 是 | Mime、Passive、Repeat |
| D0032_LAI1-2 | 是 | Mime、Repeat |
| D0032_LPI6-7 | 是 | Decision、Mime、Repeat |
| D0035_LAI2-3 | 是 | Decision、Repeat |
| D0079_LAI1-2 | 是 | Decision、Repeat |
| D0090_RIP2-3 | 否 | Decision |
| D0093_LIA2-3 | 否 | Passive |
| D0103_LAI1-2 | 否 | Decision |
| D0107_RIA1-2 | 否 | Passive |
| D0119_RIP1-2 | 否 | Passive |
| D0137_LAI1-2 | 否 | Decision、Repeat；Repeat 显著记录在 text 模态 |
| D0138_LTAS2-3 | 否 | Passive |
| D0143_RTMS1-2 | 否 | Decision |

逐条条件、模态、阶段、显著时间范围及源文件见 [14 位点显著性证据](../results/audit/fig1_nmf_selection_2026-10-17/nmf_14_significance_evidence.csv)。

### 3 个不在 464 内的旧 NMF 位点

| 位点 | Figure 1 先保留的 PhonemeSequence 解剖记录 | LexicalDelay / PictureNaming 记录 |
|---|---|---|
| D0086_RPI2-3 | Unknown，中心 0，左侧 | PIC，中心为右侧 PLG，mix=False |
| D0086_RPI3-4 | Hipp–pTL，mix=True，左侧 | PIC，中心为右侧 ALG，mix=False |
| D0086_RTAM6-7 | Hipp，左侧 | PIC，中心为右侧 PLG，mix=False |

Figure 1 的先去重规则排除了它们；旧 NMF 保留其他任务中的合格记录。这属于历史解剖数据不一致，不能用显著性筛选解释。完整来源见 [跨任务解剖冲突表](../results/audit/fig1_nmf_selection_2026-10-17/cross_task_anatomy_conflicts.csv)。本次没有判定哪份解剖记录正确，也没有更改标签。

## 5 26 位点排除名单与尚缺的理由

26 个位点全部实际影响了旧 NMF 的 **281 → 255** 这一步。当前代码、冻结名单及可追溯日志能够确认执行结果；**未找到每个位点为何被排除的独立记录，不能据此把它们全部解释为伪迹、白质或信号泄露。** 历史 `exlude_insula.csv` 是另一份不同名单，不能拿来替代这 26 项的理由。

```text
D0028_LAI3-4    D0028_LPI3-4    D0032_LPI2-3    D0032_LPI3-4
D0055_RAI7-8    D0057_RAI1-2    D0057_RPI1-2    D0059_LAT10-11
D0059_LAT11-12  D0059_LAT7-8    D0061_LAI2-3    D0061_LPI1-2
D0063_RAI1-2    D0067_LPII1-2   D0081_LTAS2-3   D0085_LAI2-3
D0090_RIP3-4    D0092_LIA1-2    D0095_LIA3-4    D0096_LIA3-4
D0096_LIA4-5    D0096_LIA5-6    D0101_RIA3-4    D0101_RIA5-6
D0125_LTAS3-4   D0140_LAI3-4
```

旧配置也明确排除 D0121，但这一次对计数没有实际影响；没有将它列为本轮减少位点的原因。

## 6 名单与复核证据

完整表含 472 个至少在某个任务记录中符合原解剖规则的位点；用“属于 Figure1 的 464”列即可得到采样主名单。每个位点列出三个阶段响应、是否进入旧 255、首次排除原因与解剖标签。

- [中文逐位点筛选表](../results/audit/fig1_nmf_selection_2026-10-17/逐位点筛选核对.csv)
- [全部筛选标志与原因](../results/audit/fig1_nmf_selection_2026-10-17/site_selection_audit.csv)
- [348 响应位点](../results/audit/fig1_nmf_selection_2026-10-17/responsive_fig1.csv)
- [116 个未满足 Figure 1 响应定义的位点](../results/audit/fig1_nmf_selection_2026-10-17/not_responsive_fig1.csv)
- [348 中没有进入旧 NMF 的 110 个位点](../results/audit/fig1_nmf_selection_2026-10-17/responsive_not_saved_nmf.csv)
- [中心标签过滤删除的 71 个采样位点，其中 69 个响应](../results/audit/fig1_nmf_selection_2026-10-17/名单_midpoint_label_zero.csv)
- [数值汇总与八项一致性检查](../results/audit/fig1_nmf_selection_2026-10-17/summary.json)
- [阶段有效数据覆盖率](../results/audit/fig1_nmf_selection_2026-10-17/phase_coverage.csv)
- [Google Doc 计数段落及版本记录](../results/audit/fig1_nmf_selection_2026-10-17/manuscript_count_excerpt.json)

输入共 1,834 份坐标文件、968 份 Repeat 时间序列文件；补充检查 306 份相关被试的各条件文件。八项检查全部通过：464、348、255 的计数正确，rank 与 assignments 名单一致，重建输入的名单与维度一致，关键源文件哈希及打包输入的大小/修改时间在核对期间未变。

| 核对任务 | Slurm 作业 | 实际用时 | 脚本 | 日志 |
|---|---|---|---|---|
| 主筛选链 | 58864900 | 3 分 40 秒 | [主核对脚本](../scripts/slurm/audit_fig1_nmf_selection.sh) | [主日志](../logs/slurm/selection_audit_58864900.out) |
| 14 位点显著来源 | 58865149 | 19 秒 | [补充核对脚本](../scripts/slurm/audit_nmf_response_exceptions.sh) | [补充日志](../logs/slurm/exception_audit_58865149.out) |

正文来源：[manuscript](https://docs.google.com/document/d/1dZGhk1ID4weFt_eGvOKpWGZnmoHEq43pRB_2kttJKA0/edit)。Figure 1 来源为原工作区 `vizpub/fig1.ipynb`；旧 NMF 入口为 `src/nmf/waveform_analysis.py` 与 `scripts/plot_nmf_concat_phases.py`；预选池来源为 `src/hga/package_highgamma.py`。具体绝对路径、输入清单与关键文件哈希均保存在本次输出目录。

## 7 统一口径时需要明确的三项规则

1. 固定跨任务唯一解剖主表，解决 3 个已进入 NMF 的冲突位点及其他受影响位点。
2. 明确中心标签为 0 是否也遵循已选定的 mix=False 原规则；旧 NMF 目前额外删除了它们。
3. 明确 NMF 入口是否限定为 Figure 1 的 Repeat 三阶段响应名单，还是继续采用跨条件、跨阶段显著并集。两者回答的筛选问题不同，不能在样本链中省略。

本次核对保留上述差异，等待统一规则后再建立三阶段分析输入。
