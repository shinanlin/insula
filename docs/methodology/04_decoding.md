# 04 Decoding 与 Figure 4

入口：[Figure 4](../../vizpub/fig4.ipynb)，固定窗部分及 cell 14/16/18 的 LDA 与跨条件图。结果审计覆盖 68 个 score H5 和 72 个 prepared-input H5；其中部分文件包含未展示的 Go 或反向跨条件结果，逐文件信息见 [结果清单](evidence/decoding_results.csv)、[输入清单](evidence/decoding_inputs.csv)。

## 1. 三条分支必须分别定义

| 设置 | PS 固定窗 articulator | LD 滑窗 lexicality | LD 跨条件 lexicality |
|---|---|---|---|
| 分类器 | PCA + LinearSVC | shrinkage LDA | shrinkage LDA |
| 输入窗口 | 按阶段固定 | 约 300 ms，约 30 ms 步长 | 训练窗×测试窗 |
| 时间特征 | 窗内全部采样点 | 分 5 个连续 bins 取均值 | 同左 |
| PCA | 功能群 80%；解剖 ROI 95% | 无 | 无 |
| CV | 分层 5-fold；重复 30 次 | 按词分组的分层 5-fold | 按共同词分组的分层 5-fold |
| 评分 | balanced accuracy | pooled OOF macro OVR AUC | pooled OOF macro OVR AUC |
| 置换 | 每 fold 打乱训练 trial 标签，200 次 | 全局打乱词级标签，5000 次 | 两条件同步打乱词级标签，5000 次 |
| 校正 | 每 panel 单独 p，无跨 panel 校正 | 每条曲线内时间簇长度 | 每张二维图内簇质量 |
| chance | 0.25（4 类） | 0.5（2 类） | 0.5（2 类） |

这些不是“同一种 decoding 换几个画法”。不能统一写成 SVM、PCA95%、trial shuffle 或 time-generalization accuracy。

## 2. 单试次伪总体数据怎样生成

核心：[prepare_functional_decoding_dataset.py](../../src/decoding/prepare_functional_decoding_dataset.py)。解剖对照的 preparation 在外部 task worktree 的 `prepare_decoding_dataset.py`，同样需要冻结源码。

### 2.1 特征与试次对齐

输入为单试次 HGA z-score，保存 `X[trial,channel,time]`。Sensory/Sustain/Motor 是跨被试、双侧电极组成的 pseudo-subject，STGl/SMCl/MFGl 是左侧解剖对照。一个 pooled trial 的不同通道可能来自不同被试的不同真实试次，不是这些人被同时记录的一次试次。

部分旧 prepared H5 的 `roi` 属性仍写作 `Sustained` 或 `Intermediate`，对应当前路径中的 Sustain 或 Motor。这是名称沿革，不能据这些旧字符串额外推断另一套功能群；应联合路径、通道名单和 assignment hash 确认。

对词的第 n 次呈现构造 `{word}_{occurrence}`，用于跨电极/被试对齐；occurrence 在正确性筛选之前生成，避免漏掉一个错误 trial 后使后续呈现序号整体移位。bare word 另存作 item/group ID。按 trial×channel×time 聚合后外连接，缺测组合为 NaN。不能将 occurrence trial ID 当成词级 CV group。

### 2.2 功能群与响应筛选

- PS：当前阶段 Repeat sig 电极 ∩ 冻结的功能标签名单 ∩ 可用 z-score 通道。
- LD：当前阶段 Decision/Repeat sig **并集** ∩ 功能标签名单 ∩ 可用通道，两条件用对称通道 QC。
- 输入仍是单电极特征，不是 NMF 的 W，也不是 NNLS 投影后的三个 component 特征。
- 解剖 ROI：STG 合并 HG/STGa/STGp，SMC 合并 PrG/PoG/Subcentral，另有 MFG。现有 PS 解剖 H5 明确记录输入为全 `epoch(band)(zscore)`；不能套用 functional same-phase-sig 规则。LD 解剖 H5 记录 same-phase Decision/Repeat sig union。

LD 准确性以 CORRECT 标记筛选。PS 还排除 RT<50 ms，RT 由原始事件序列及对应音节事件配对获得。PS preparation 中另有固定 13 个通道排除：

`D0040_L1IF3-4`, `D0084_RFAI1-2`, `D0086_LTPI2-3`, `D0032_LAI4-5`, `D0090_RIA4-5`, `D0096_LFAI2-3`, `D0096_LFAI4-5`, `D0102_RFAI2-3`, `D0106_LTAS2-3`, `D0121_LFMI3-4`, `D0122_LFAI3-4`, `D0125_LIA4-5`, `D0125_LIA7-8`。

### 2.3 缺测 QC

定义一个 trial-channel 单元只要任一时间点 NaN，就算该单元缺测。剔除缺测 trial 比例大于 0.5 的 channel，剔除缺测 channel 比例大于 0.5 的 trial；functional preparation 迭代至稳定。LD 两条件保持共同的最终通道集合，但 trial 列表允许不同。它不是“只要少于一半时间采样缺失就保留”的规则。

PS 外部解剖 preparation 当前实现为 channel QC 一遍后 trial QC 一遍，不能假定与 functional 的迭代过程完全一致。prepared H5 记录 retained/candidate channels、trial 数、selection 说明和（功能群）assignment hash，可逐项验收。

## 3. 标签的精确定义

PS 解码第一音素的 articulatory group，包含特殊的 `a e` digraph 处理：a/ae→low_vowel，i/u→high_vowel，b/p/v→labial，g/k→dorsal，其他→other。实际被分析 H5 只有四类，编码顺序 `dorsal=0, high_vowel=1, labial=2, low_vowel=3`。完整数据常为 24/28/28/24 次，缺测 QC 后个别类别少一个或数个 trial。该标签不是词 identity，也不是每个音素各为一类。

LD 为 lexicality：Word=0、Nonword=1。bare word 用作 group，多个呈现的 lexicality 一致。二分类 AUC 的 chance 是 0.5。

## 4. 分支 A：PS 固定窗 PCA–LinearSVC

实现：[run_decoding.py](../../src/decoding/run_decoding.py)、[decoder.py](../../src/decoding/decoder.py)；实际 functional worker：[functional_decoding_worker.sh](../../scripts/functional_decoding_worker.sh)。

### 4.1 裁窗和 pipeline

| phase | 功能群 | 解剖对照 | 功能群采样点数 |
|---|---|---|---:|
| Stimulus | `[0,0.5)` | 同左 | 64 |
| Delay | `[0,0.7)` | 同左 | 89 |
| Go（存在结果，主 bar 不展示） | `[0,0.5)` | 同左 | 64 |
| Response | **`[0,1)`** | **`[-0.5,0.5)`** | 128 |

每个 fold 依次执行 `Vectorizer → StandardScaler → PCA → LinearSVC`。Vectorizer 展平 channels×time，StandardScaler 按训练 trial 对每个特征居中及单位方差缩放，PCA 与分类器也仅用训练集拟合。

functional：PCA `n_components=0.80`（累计方差阈值），LinearSVC `C=1`、`random_state=42`、`max_iter=5000`；所检查 sklearn 其余默认为 L2 penalty、squared_hinge loss、dual="auto"、tol=1e-4、class_weight=None、one-vs-rest。解剖结果 H5 直接证明 PCA 阈值为 **0.95**；旧 runner 用 `LinearSVC(random_state=42)`，max_iter 等未写入 H5，不能将当前 5000 的参数当作历史已证实的解剖参数。

PCA 保留的是每个训练 fold 内计算的累计方差比例，不是固定 80/95 个 PC。代码显式设 PCA `random_state=42`，其余采用所检查的 `svd_solver="auto"`、`whiten=False` 默认；auto 按训练矩阵尺寸等决定求解路径。输出未记录每个 fold 的实际 PC 数与最终 solver。

### 4.2 CV 与缺失值

`MinimumNaNSplit` 基于重复分层 K-fold。每个外部 repeat 内 5 折，`n_repeats=1`，总共手动重复 30 次，种子 `42+repeat_index`；NaN 约束 `min_non_nan=2`，所检查实现约束训练部分。

这里没有按 word 分组，也没有 leave-one-subject-out；同一词的不同呈现可能同时进入 train/test，因此不能解释成严格的陌生词泛化，更不能解释成跨被试泛化。

`sample_fold` 在训练侧按类别把 NaN 先置零，再调用 mixup（seed=42）。安装的 mixup 只针对仍有缺失的行工作，因此这个顺序通常不产生注释所暗示的真实 mixup 增广。测试 NaN 以 seed=`42+fold_index` 的 N(0,1) 噪声填充；训练侧残留 NaN 也有噪声兜底。这个步骤发生在 fold 内，不能简写成“删除所有缺失 trial”或“均值插补”。

### 4.3 观察值与置换

每折计算 balanced accuracy，即各类别 recall 的平均。重复 0 的 5 个观察分数用于 200 次置换：每个 fold 只打乱训练 trial 的 y，测试 y 保持真实；每次重拟合 pipeline，再对折分数取均值。

`p=(1+count(mean_fold(score_perm) >= mean_fold(score_observed)))/201`。

最小 p 为 1/201≈0.004975。后面 29 轮增加观察 CV 的稳定性，没有各自重复生成 200 次 null。

置换种子由 `RandomState(42).randint(0,2**31−1,size=200)` 产生；每个 fold 重新从同一个 root seed 生成该序列。每次独立 permutation seed 用于打乱该 fold 训练标签，不是全体 trial 共用的一张标签重排表。输入填充的 fold seed 为 `42+fold_index`。

H5 区别：`accuracy` 为第一轮 5 个分数；`accuracy_repeats` 为 30×5；`accuracy_stable` 为这 150 个分数的均值；`confusion/confusion_norm` 来自第一轮 OOF 预测。

**Figure 4 bar 实际读取 `accuracy`**，显示第一轮均值±5 folds 的标准误；不是 30 轮均值±被试误差。星号使用该 panel 的未校正 p<0.05；18 个主要 bar 比较之间未再校正。混淆矩阵按真实类行归一化。

### 4.4 已保存输入的实际规模

下表为 `channels / pooled trials`，不是独立被试数：

| ROI | Stimulus | Delay | Response |
|---|---:|---:|---:|
| Sensory | 9 / 103 | 7 / 103 | 2 / 93 |
| Sustain | 9 / 101 | 16 / 104 | 8 / 104 |
| Motor | 8 / 104 | 14 / 104 | 54 / 104 |
| STGl | 195 / 104 | 195 / 104 | 195 / 104 |
| SMCl | 206 / 104 | 206 / 104 | 206 / 104 |
| MFGl | 210 / 104 | 210 / 104 | 210 / 104 |

不同 ROI 的通道数、入选规则、PCA 阈值，以及 Response 窗并不匹配。原图的性能高低不能单独归因为脑区信息含量差异。

空间 panel 对三阶段 prepared functional channels 取并集，得到 83 个通道；当前赋色标签与输入冻结标签间有 4 个原 motor 通道改为 sustain。训练输入没有因为 Notebook 重读新 assignment 而自动更新，见 [标签差异清单](evidence/decoding_assignment_mismatches.csv)。

## 5. 分支 B：LD 滑窗 shrinkage LDA

实现：[run_decoding_resolved_lda.py](../../src/decoding/run_decoding_resolved_lda.py)、`decoder.py::decode_permutation_auc_pooled`。主要 worker：[decoding_resolved_lda_worker.sh](../../scripts/decoding_resolved_lda_worker.sh)。

### 5.1 时间与特征

预先裁数据到 `[-0.5,1.5)`，采样率 128 Hz。请求 `window=0.3` s，按 `int(0.3*128)` 得 **38 点**；请求步长 0.03 s，按时间端点序列换算索引，实际相邻窗移 3 或 4 点，并非恒定 3 点。

窗终点名义序列从 −0.2 s 起，边界检查后保留 57 个终点至 1.48 s；`end_index=int((end_time−tmin)*128)+1`，`start_index=end_index−38`。**输出 time 与图横轴表示窗口终点**；若要表达中心，应另行转换并更新图注。

TimeBin 用 `np.array_split` 将 38 点分成 5 个连续 bins，大小为 8、8、8、7、7，每 bin 求均值；Vectorizer 得到 `5×n_channels` 特征，StandardScaler 按训练 fold 拟合。分类器为 `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`，priors=None，按训练类别频率估计；没有 PCA 或 SVM。参数语义见 [LDA 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)。

### 5.2 按词分组 CV 和评分

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`，group 是 bare word。一个词的所有呈现都在同一 fold，测试词不会以另一呈现在训练中出现。这是**留出词**的验证，不是留出被试。分组和分层的含义见 [官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)。

缺失值处理沿用 fold 内 `sample_fold` 规则。每折产生测试 trial 的 decision values，再在**该测试 fold 内**对每个类别的 decision-value 列做均值/标准差归一化，最后合并所有 OOF trials 算 macro one-vs-rest AUC。二分类时构造 `[-v,v]` 两列。

注意这包含两种归一化：输入 StandardScaler 用训练数据；decision value 的 fold 间尺度对齐用测试 batch 的分布（不使用测试标签）。因此评分依赖测试 batch 的组成，不能描述成完全逐个未知样本独立校准。也不能把 pooled OOF AUC 写成五折 AUC 的平均。

### 5.3 词级置换与一维时间簇

5000 次 null 将每个唯一词对应的类别整体打乱，再扩展至该词所有 trials。保留原始 CV splits，不对每次置换重新分层。每次用 permuted 训练标签重拟合 LDA，并对 permuted 测试标签评分。无监督预处理可以按原 fold 缓存。所有时间窗使用一致的 seed/schedule，保留 null 时间相关结构。

缺少训练类别等导致的无效置换得到 NaN；逐点 p 只用有效 null：`(1+count(null>=observed))/(1+n_valid)`。在送入时间簇校正前，null NaN 替换为 chance=0.5。

时间簇以逐点 p<0.05 形成，使用连续时间点**簇长度**及每次置换的最大长度，按安装的 IEEG `time_cluster` 经验比较规则产生 corrected mask。校正 family 是一条 ROI×phase×condition 曲线，不含其他 35 条曲线。保存 `p_values` 是逐点 p，并非每个点的 cluster-adjusted p；显著性图要读 mask。

绘图可用 `gaussian_filter1d(sigma=2,mode="nearest")` 平滑 AUC（名义约 60 ms），不重新用平滑结果计算 p/mask。

### 5.4 已保存样本规模

共 6 ROI×3 phase×2 condition=36 份结果。通道数如下，两条件经过对称 QC：

| ROI | Stimulus | Delay | Response |
|---|---:|---:|---:|
| Sensory | 16 | 14 | 10 |
| Sustain | 30 | 30 | 28 |
| Motor | 18 | 20 | 26 |
| STGl | 85 | 92 | 94 |
| SMCl | 33 | 32 | 47 |
| MFGl | 58 | 68 | 58 |

通常 168 pooled trials、84 个词；少数输入为 166/167 trials，具体见逐文件清单，不能把 168 写成所有条件固定不变。

## 6. 分支 C：跨条件、跨时间 LDA

实现：[run_cross_condition_resolved_lda.py](../../src/decoding/run_cross_condition_resolved_lda.py)、[pooled_io.py](../../src/decoding/pooled_io.py) 的 `pair_pooled_conditions`、`decoder.py::decode_cross_permutation_auc_pooled`。

### 6.1 两条件配对与 CV

按共同 `{word}_{occurrence}` 对齐两条件，检查标签字符串相同，并取共同 channels（顺序依 source）。无 trial key 时有 occurrence 合成回退，迁移时宜显式保存 key。

用共同 bare word 建立 5-fold StratifiedGroupKFold，seed=42。在某 fold，以 source 条件训练词的 trial 拟合，以 target 条件**留出词**的 trial 测试。不是在一个条件训练全部词、再在另一个条件测试相同全部词。

每个训练窗分别在 source train 上 fit 5-bin 特征变换、StandardScaler 和 LDA，再应用到 target 的每个测试窗。decision value 按 target test fold 归一化，计算 pooled OOF AUC。形成 57×57 的训练时间×测试时间矩阵。

已保存 Delay 的 Sensory/Sustain/Motor/STGl 四 ROI、两个方向，共 8 份结果；Figure 4 展示 Repeat→Decision。共同通道为 14/30/20/92。Sustain 为 166 个配对 trials（Word/Nonword 各 83，84 个词），其余通常 168。

### 6.2 二维簇校正

5000 次置换采用同一个词级随机标签映射，同时应用两条件，并贯穿所有训练×测试时间点。

1. 每个 cell 计算单侧逐点 p=`(1+#null>=AUC)/5001`。
2. 用 p<0.05 形成二值图；采用 3×3 全 1 邻接，即 **8-connected**（含对角邻居）。
3. 一个观察簇的 mass=`sum(AUC−0.5)`，不是簇面积/长度。
4. 每个 cell 对 null 按 `rankdata(method="max")` 转为上尾比例 `(B−rank+1)/B`，阈值同为 <0.05；每次置换保留全图最大簇 mass。
5. 簇 p=`(1+#null_max_mass>=observed_mass)/5001`，p<0.05 保留。

family 只含一 ROI、一个方向的一张二维图；不同 ROI/方向间没有共同 max-null。存储矩阵行=train、列=test；Notebook 为画 x=train/y=test 做转置，不能反读方向。显著的离对角区域说明表征可跨条件/时间泛化，不能解释成两个时间点之间的因果信息流。

## 7. 迁移时必须明确的决策

首先选择泛化目标：新 trial、新 item 还是新 subject。按目标选 CV group，保存每折 ID。现有 PS trial-CV 与 LD item-CV 支持的结论不同。若要比较 ROI，建议匹配窗口、PCA、通道数量/抽样方案及筛选规则；这是新设计，不应静默改写原图的方法。

NMF 和 sig 选区在当前固定 cohort 上预先完成，没有嵌套到每个 decoding fold。可将当前结果表述为“在预先固定的功能群/响应电极集合上评估解码”。如果要估计从发现功能群到预测新数据的完整流程性能，需要将数据驱动的发现/筛选纳入外层训练集，或使用独立发现数据。

所有发布文件建议直接保存 `split_ids`、group/trial 名单、feature/channel 顺序、窗口实际 indices/time、每折 PC 数、观察及 null 分数、置换映射、cluster statistic/family、assignment hash、完整 pipeline 参数与依赖版本。图应显式选择 `accuracy` 或 `accuracy_stable`；若选择后者，应相应生成匹配的推断结果，不能只换 bar 而保留第一轮的 p。

复现当前 functional 固定窗应以 worker 的 `variance=.80, n_folds=5, n_repeats=30, n_perm=200, C=1` 为准；CLI 某些默认 `.85/10 folds/2 perms` 是开发默认，不代表 Figure 4 的运行配置。
