# 01 数据、HGA、解剖与 Figure 1

## 1. 来源与数据契约

本地绘图入口：[Figure 1](../../vizpub/fig1.ipynb)，cell 3、11；cell 编号均从 0 开始。打包实现：[package_highgamma.py](../../src/hga/package_highgamma.py)、[package_ave_cord.py](../../src/hga/package_ave_cord.py)。路径解析：[paths.py](../../src/paths.py)。

预处理位于仓库外的 `/hpc/group/coganlab/nanlinshi/seeg-preprocessing/`，以及 `seeg-preprocessing-worktrees/{lexical_delay,lexical_nodelay,phoneme_seq,picture_naming,sentence_rep}/task/`。主要入口为 `extract_ieeg_epochs.py`、`time_perm_bands.py`；共享实现为 `lib/ieeg_epochs.py`、`lib/time_perm_stats.py`、`common/save_clean_derivatives.py`、`common/parcellation.py`。文件指纹见 [source_fingerprints.csv](evidence/source_fingerprints.csv)。这里精确描述已检查的当前实现，未从原始 EDF 重建全部历史处理记录。

单试次 epoch 的核心维度为 `trials × channels × time`。通道名如 `D0057_RAI3-4` 表示被试及双极通道，不能将其中两触点当成两条独立观测。至少保存事件字符串、原始事件 sample、事件所属 recording、词/概念 ID、准确性标记、通道顺序和时间向量。

## 2. 清理、参考与 HGA

| 项目 | 本项目实现/参数 | 证据范围 |
|---|---|---|
| 参考 | bipolar；后续模块不重复重参考 | 预处理及结果属性 |
| 工频处理 | 当前 clean derivatives 使用 `[60,120,240]` Hz notch | 当前源码，不能据此补全所有历史清理步骤 |
| 双极几何筛选 | shaft bipolar 默认相邻候选距离上限 20 mm、转向上限 60° | `lib/shaft_bipolar.py` 默认；历史逐通道决定需名单 |
| highgamma 频带 | **70–200 Hz** | `DEFAULT_BANDS`；覆盖 `gamma.extract` 库默认的 70–150 Hz |
| 包络构造 | filterbank + Hilbert amplitude；子频带按约 1/7 octave 排列，最后将包络相加 | 安装的 `ieeg/timefreq/gamma.py` |
| 分析采样率 | 128 Hz | 源码及已保存 decoding/RT 输入、结果 |
| 常见 epoch | `[-1,1.5)` s，共 320 点，最后一点 1.4921875 s | 已保存输入；MNE crop 的闭端点行为需单独核对 |
| 边缘处理 | 提取时前后额外 0.5 s padding，随后裁回目标 epoch | 预处理实现 |

这里的 HGA 是频带 Hilbert 振幅组合。虽然中间目录可能叫 `(power)`，不能仅凭文件名写成“振幅平方功率”，也没有在 z-score 前统一 log-transform 的步骤。滤波器细节应随已安装 IEEG 版本/源码冻结，不能用一个任意 Butterworth 参数替代。

Sentence 的任务脚本存在 `tmax=3` 的默认值与 epoch 配置/启动参数采用 1.5 s 的区别；应以具体生成文件的时间向量为准，而不是从一个函数默认值推断所有任务。

## 3. 基线与 z-score

基线取条件起始事件：LD 为 Cue（区分 Yes_No/Repeat），PS 为 Start，PN 为 StartCue（按条件组织）。其初始提取窗为 `[-1,0.5]` s，计算基线的时间窗裁为 `[-0.5,0]` s。**这不是分别对 Stimulus、Delay、Go、Response 的事件前 0.5 s 各自重做基线。** 同一条件下的各阶段使用该条件的起始基线。PN 后续可分别保存 text/sound/image，但不能据此假定基线也按每个 modality 独立估计。

对通道 c，用该基线的试次和时间两个轴一起计算均值、标准差：

`Z[i,c,t] = (A[i,c,t] − mean_baseline[c]) / std_baseline[c]`

安装实现的 `rescale(mode="zscore")` 使用 axes `(0,2)`，标准差 `ddof=1`。不按每个试次单独做 z-score；不同分析分支后续的 StandardScaler 也不能与这一步混淆。

异常值处理调用 `outliers_to_nan(outliers=15)`。实际规则是：逐 trial×channel 计算整段时间上的最大绝对值，与该通道跨试次、跨时间的绝对值均值加 15 倍标准差比较。超过则将**整个 trial-channel 时间序列**设为 NaN；不是仅截断单个异常采样点，也不是删除该试次所有通道。

## 4. 相对基线的时序显著性与响应电极

入口是共享 `lib/time_perm_stats.py::run_stats_for_condition_phase`。检验使用尚未做上述 z-score 的包络：任务 epoch 与 baseline，默认统计函数为 IEEG 实现的 t-test，通过试次重排构建 null。

| 参数 | 实际调用/生效值 |
|---|---|
| `n_perm` | 5000 |
| `tails` | 1，增加方向 |
| cluster-forming `p_thresh` | 0.1 |
| `p_cluster` | 调用未显式提供；所检查依赖默认 0.1 |
| `ignore_adjacency` | 1；按通道递归处理，保留时间邻接 |
| 并行 | `n_jobs=-1`，应在分配的计算资源内运行 |
| 随机种子 | 调用未固定，默认 None |

时间簇使用连续时间点的**长度**，不是 t 值总和的 cluster mass；null 为最长时间簇。必须冻结 `ieeg.calc.stats.time_perm_cluster/time_cluster` 的具体版本：其经验比较及边界规则不应被静默替换成另一套“标准 cluster permutation”。这里不是跨全部通道和时间联合校正。

保存两类量：`mask`（簇判定）与逐点 `pvals`。电极进入 `sig_ch_names` 还要求在以下阶段窗内至少一个点同时满足 `mask=True` 和 `pvals<0.05`：

| 阶段 | 筛选窗 |
|---|---|
| Stimulus / Delay / Go | `[0,0.5)` s（实现中的索引 slice） |
| Response | `[-0.5,0.5)` s |

因此，不能把“响应电极筛选”简单写成一次 `p<0.05` 检验，也不能因为最终额外用了 0.05 就宣称簇阈值也是 0.05。历史 stats H5 主要存 mask、pvals 和通道名，没有完整统计参数属性；上述详细设置的证据来自当前实现及依赖。

## 5. 打包成后续图使用的表

`package_highgamma.py` 在每位被试、每种 modality 内，从所有条件、所有阶段的 `epoch(band)(sig)` 取显著电极并集（不含 baseline）。随后从完整的 `epoch(band)(zscore)` 中提取这些电极，并对试次做 `nanmean`。输出逐 electrode×condition×phase×time 的 `value`，并附对应 `mask`、解剖和坐标信息。

这里有两个不同概念：

- 电极进入打包表：因为它至少在一个条件/阶段被选为显著。
- 电极某一阶段某一时点显著：由该行的 mask 决定。

NMF 使用进入表的完整波形，并没有把每个非显著时间点清零。热图则可以单独依据 mask 遮掉非显著点。缺失的 mask 被警告并置 False，不能当成有统计支持。

同一电极可以在多个任务中出现。后续 NMF 在 electrode×time 内平均这些任务行，不按各任务原始 trial 数加权；曲线显示分支未必做同样聚合。坐标覆盖表则来自采样电极，不应只取显著电极作为覆盖分母。

## 6. 解剖归属与坐标

解剖分类使用个体空间中的 Hammers/MAPER 体积分区。双极通道依据两端点和中点的标签形成 gross ROI 共识：忽略 WM、Unknown、hypointensity 等非目标组织；有效组织属于一个 gross ROI 时为 pure，多个 gross ROI 时拼成混合标签并标记 `mix=True`。中点与两端几何关系另有一致性检查。具体取标签/边界行为以冻结的 `common/parcellation.py` 为准。

NMF 的 pure insula 要求 gross ROI 为 AIC/PIC、`mix=False`、有效 label 非 0。Hammers 细分用 AP、ASG、MSG、PSG、ALG、PLG；前四归前部、后两归后部。后续用于 surface 展示的 `aparc.a2009s` insula 标签不是这套数据筛选依据，不能混写成“依据 Destrieux atlas 定义全部岛叶电极”。

必须保存个体空间坐标与模板空间坐标的名称/变换。`x/y/z` 与 `x_t/y_t/z_t` 不可互换；模板 `cvs_avg35_inMNI152` 的 surface/tkRAS 与 scanner/MNI RAS 也不是仅换列名即可等价。绘图时吸附至最近 pial 顶点用于可视化，不重新决定组织 ROI。

## 7. Figure 1 具体怎样计数

任务集合为 PS、LD、LND、PN，不包含 Sentence。Notebook 使用旧相对路径 `../results/{Task}(bipolar)(hammers)`；本次检查四个目录在当前工作区不存在，相关旧数据存在于相邻 `insula/results`。以下数字来自 Notebook 缓存，未伪装成此次重算。

覆盖图先要求坐标非空并按通道去重；缓存输出为 13,933 个覆盖通道，其中 insula 442。ROI 饼图另做 pure/mix、label 和 ROI 排除筛选，分母因此不同。ROI 合并规则为 STGp/STGa/HG→STG，PrG/PoG/Subcentral→SMC，IFG 保留，其余保留为 Other。排除列表为 Unknown、BrainStem、Thal、CC、Caud、Put、Amyg、Hipp、LinG、Cun、mOccG、PhG、FuG、GRect、LatV、GSubcallosal、Intersection、BrainStem–Thal。完整 cell 来源可由 [notebook_cells.csv](evidence/notebook_cells.csv) 定位。

响应数在 Repeat 中要求 `mask=True` 且 **严格** `0<t<1`（Stimulus/Delay），或 `−0.5<t<0.5`（Response），随后对任务和阶段取唯一通道并集。此处不是预处理 sig 名单的 `[0,0.5)` 规则；也没有统一限定为 sound-only。

| 分组 | 响应/采样电极（缓存） | 比例 |
|---|---:|---:|
| Insula | 201 / 360 | 55.8% |
| STG | 331 / 456 | 72.6% |
| SMC | 242 / 406 | 59.6% |
| IFG | 128 / 214 | 59.8% |
| Other | 1348 / 3060 | 44.1% |
| 合计 | 2250 / 4496 | 50.0% |

饼图半径按 `sqrt(total/max_total)` 缩放，面积表达样本量。这是描述性覆盖/响应比例，没有因为画了比例就额外做 ROI 间显著性检验。覆盖 insula 442、筛选分母 360、NMF 的 255 是不同入选规则下的集合，不应混用。

Figure 1 的 TFR 部分仍是占位说明，没有可追溯的已实现时频分析链；本文没有替它补写 wavelet 参数。

## 8. 新项目复用规范

先冻结 baseline 事件与时间窗、滤波包络定义、采样率、坏通道名单和双极定义，再生成 z-score 与显著性结果。保留全通道数据、sig 名单、逐点 p 与 mask，避免下游只有筛选后的数据可用。将“全采样”“pure insula”“有任务响应”“进入 NMF”“某一阶段可解码”分成独立名单并记录筛选原因。

建议新增结果属性记录频带、滤波实现版本、baseline event/window、z-score axes/ddof、异常值规则、置换种子、cluster-forming/cluster-level 阈值及校正 family。以上属于迁移时的记录规范；现有历史文件并未全部具备这些属性。
