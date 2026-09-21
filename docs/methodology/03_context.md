# 03 任务情境、单变量条件对比与 Figure 3

入口：[Figure 3](../../vizpub/fig3.ipynb)。统计生成：[contrasts_mean.py](../../src/univariate/contrasts_mean.py)。主要启动文件：[PN P75](../../scripts/slurm/univariate_p75_repeat_vs_passive_picture_naming.sh)、[LD P75](../../scripts/slurm/univariate_p75_decision_vs_repeat_delay.sh)、[Mean runner](../../scripts/run_univariate_mean.sh)。

## 1. 三类图来自不同分析

| 图中内容 | 数据处理 | 主要含义 |
|---|---|---|
| Passive / Repeat / Decision 波形 | 各任务单电极试次平均，再按功能群和条件绘制 | 跨任务池的描述性响应 |
| Delay 空间显著点 | LD Decision−Repeat，窗内 **Mean**，单电极置换及 FDR | 正向条件差异的位置 |
| 成对半小提琴/散点 | PN Repeat−Passive 或 LD Decision−Repeat，窗内 **P75** | 单电极条件差异及功能群内方向性 |

不能把空间图写成 P75 分析，也不能把所有波形差异当成同一批被试的配对对照。

## 2. 波形怎样形成

从原始 `epoch(band)(zscore)` 读取，按任务、被试取所有条件/阶段 sig 电极并集，再与 NMF 名单相交。只取 sound，排 D0121，不纳入 Sentence。每个 channel×task×condition 先跨 trials 做 `nanmean`；随后 seaborn 按条件、功能群、阶段求平均曲线及默认 95% bootstrap CI（1000 次，未固定 seed）。没有先把每位被试平均成同等权重。

基础条件池：Passive=LND+PN；Repeat=LD+LND+PN+PS；Decision=LD+LND。另有下列显式展示规则：

- Passive 的 Response 只用 PN，不用 LND，并从 PN 数据补入这部分响应波形。
- sustain 群的 Stimulus/Passive 只用 LND，不用 PN。
- Delay 中 Repeat/Decision 只用 LD，Passive 只用 PN；Delay 只显示 t≥0。
- 全部绘图曲线去掉 `D0125_LTMS1-2`、`D0125_LTMS2-3`、`D0125_LTMS3-4`、`D0129_LFAI2-3`。虽然附近注释提到 Passive/Response，代码的排除作用于整个波形表。
- 上限采用 `time<1.5`；各 panel 的进一步坐标裁切属于显示。

“跨条件使用同一 sig 并集”只保证每个任务/被试候选的一致性，不保证所有任务、所有条件有完全相同的最终电极×被试集合。条件曲线之间还包含任务构成差异，不能直接视作严格的同被试配对情境效应。上述四个绘图排除也没有自动传播到后面的统计 CSV。

## 3. 每个 trial 的标量响应

对每个 subject×channel×phase，裁如下固定窗；MNE crop 包含可匹配到的端点采样，其行为与 NMF 的开区间不同。

| phase | 窗口 |
|---|---|
| Stimulus、Delay、Go | `[0,0.5]` s |
| Response | `[-0.5,0.5]` s |

Mean 分支：`s_i = nanmean_t Z[i,c,t]`。

P75 分支：`s_i = nanpercentile_t(Z[i,c,t],75)`，NumPy 默认 linear 插值。然后每个条件的 `mean_a/mean_b` 是这些**单试次 P75 的均值**。不是将所有试次和时间一起求第 75 百分位，也不是先得到平均波形再求其 P75。

准确性过滤使用 slash token：有准确性标签的事件保留 CORRECT，没有标签的 PN 事件保留。两条件只比较共同通道，每条件每电极至少 3 个有限 trial 标量。

## 4. 电极内条件置换

令条件 a、b 的 trial 标量为 s_a、s_b，观察统计量 `D=mean(s_a)−mean(s_b)`。将两组 trial 合并，随机重排条件归属，同时保持两组原始大小，得到 5000 个 D_perm。双侧 p 值：

`p = (1 + count(|D_perm| >= |D|)) / 5001`。

每个单电极检验用可重复种子：对字符串 `"{subject}|{channel}|{phase}|{contrast}"` 求 SHA-256，将最前 4 bytes 按 little-endian 转为整数。没有按 trial 配对差值翻转符号，也没有按词分块重排；是独立试次的两条件置换。

BH-FDR 在**同一 subject×phase×contrast 的全部被检验通道**内实施，然后才在绘图端限制到 insula/NMF。显著定义 `p_fdr<0.05`。没有跨全部被试、阶段和 contrast 合成一个全局 family。若新项目希望这些维度也共同控制误差，需要另外定义 family。

结果 CSV 保存 `mean_a`、`mean_b`、`mean_diff`、`p`、`p_fdr`、`significant` 等；通常没有 window/n_perm 的完整属性。具体参数由当前生成代码及启动脚本确定，历史参数不完整的限制需保留。

## 5. Figure 3 实际如何使用 p 值

**空间 panel**：LD、Delay、`DecisionVsRepeatMean`，保留 `p_fdr<0.05` 且 `Decision−Repeat>0`。当前加载路径解析到相邻 `insula/results`；Notebook 缓存为 22 个点，sensory=3、sustain=16、motor=3。此处是 Mean 指标。

**上排 PN 配对 panel**：Delay 的 `RepeatVsPassiveP75`，每个电极以两条件 mean P75 配对。着色依据 **未校正 `p<0.05`** 且 Repeat>Passive。缓存有 116 个电极：sensory 15、sustain 46、motor 55。

**下排 LD 配对 panel**：Delay 的 `DecisionVsRepeatP75`，着色依据 FDR significant 且 Decision>Repeat。缓存 154 个电极：sensory 21、sustain 52、motor 81。虽然图横轴按 Repeat→Decision 排列，统计的 a−b 仍为 Decision−Repeat。

同一排、同一功能群还做单侧 paired Wilcoxon：输入电极的 `mean_a` 与 `mean_b`，`alternative="greater"`，`zero_method="wilcox"`，至少 8 个电极才运行。这里用全部有效配对电极，而非仅着色显著者。群间六项 Wilcoxon 没有额外多重比较校正，也未按被试聚类校正；同被试多个电极不能因此当成多个独立被试。

半小提琴的 Gaussian KDE 使用默认 Scott 带宽，网格为第 2–98 百分位内 200 个 y 点，是显示设置，不参与上述 p 值。

## 6. 迁移规范

固定每个 contrast 的任务、条件 a/b、准确性规则、trial 标量定义、phase 窗、入选通道、置换单位与 FDR family。预先规定着色使用 raw p 还是 adjusted p，图例写明，不从两个阈值中挑选更好看的结果。

如目标是总体被试层面的条件效应，建议先汇总到 subject，或另建含 subject 层级的模型；若同词重复呈现构成重要依赖，应考虑按 item 设计交换单位。若要隔离“任务情境”的因果解释，应以任务内、刺激匹配的条件对照为主。以上是新项目设计建议，现有五图没有自动具备这些额外控制。
