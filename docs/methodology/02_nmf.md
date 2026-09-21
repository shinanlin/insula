# 02 NMF、功能群与 Figure 2

入口：[Figure 2](../../vizpub/fig2.ipynb)。核心实现：[waveform_analysis.py](../../src/nmf/waveform_analysis.py)、[rank_selection.py](../../src/nmf/rank_selection.py)。启动参数：[nmf_concat.sh](../../scripts/slurm/nmf_concat.sh)、[nmf_rank_bootstrap.sh](../../scripts/slurm/nmf_rank_bootstrap.sh)。发布结果位于 `results/nmf/`，已有简版说明见 [NMF.md](../NMF.md)。

注意 `waveform_analysis.py` 顶部还保留“只用 Stimulus 发现、其他阶段留出”的另一条分析说明；Figure 2 的 canonical manifest 和 concat 启动脚本明确采用四阶段拼接。本文按实际调用链和结果描述 concat 方法，不能仅根据模块开头的 docstring 宣称 Delay/Go/Response 在此图中均为 held-out 数据。

## 1. NMF 回答什么问题

这里把每个电极在多个事件对齐阶段的平均 HGA 形状表示为几个非负时间模板的加权组合。输入行是电极，不是试次或被试；列是按既定顺序拼接的时间点。模型得到非负载荷 W 与非负时间模板 H：

`X[N,T] ≈ W[N,K] × H[K,T]`

功能群来自电极的主导载荷；不是先按 AIC/PIC 定义群，再分别拟合 NMF。该分解描述所选电极的响应形状，并非 trial-level 潜变量模型。

## 2. 输入集合与拼接矩阵

1. 读取 PS、LD、PN、Sentence 的已打包 HGA。
2. 保留 `description=Repeat`、`modality=sound`、gross ROI 为 AIC/PIC、`mix=False`、有效 label 非 0。
3. 排除被试 D0121，另排除 [26 个通道的冻结名单](evidence/nmf_exclude_channels.txt)。这是本项目的 QC 记录；新项目需自行制定名单，不能机械复制被试号。
4. 电极候选已由预处理跨条件/阶段的显著性并集限制。输入是 `epoch(band)(zscore)` 的试次平均；不再次用逐点 mask 对波形清零。
5. 在每个阶段内，对相同 channel×time 的多个任务/文件行取均值。不是按 trial 数加权，也不是先将每位被试平均成一行。
6. 使用下表的**开区间**裁切；每阶段要求有效时间覆盖率至少 0.95，对剩余缺失沿时间线性插值，`limit_direction="both"` 补端点。
7. 只保留四阶段都存在的电极，即电极集合取交集；缺整段的电极不会用全零阶段补齐。
8. 按 Stimulus→Delay→Go→Response 拼接。

| 阶段 | NMF 实际裁窗 | 128 Hz 时列数 |
|---|---|---:|
| Stimulus | `0<t<1` s | 127 |
| Delay | `0<t<1` s | 127 |
| Go | `0<t<1` s | 127 |
| Response | `0<t<0.5` s | 63 |
| 合计 | 四段拼接 | 444 |

已保存分析对应 **255 个电极、63 位被试，255×444 的矩阵**。四个区间分别相对其事件零点，不表示连续 3.5 s 的真实时间；没有把不同试次 Delay 时长做时间扭曲，也没有给每个阶段额外等权归一化。较长阶段自然贡献更多列。

## 3. 非负化和幅值归一化

先对完整拼接行做半波整流：`X_plus[i,t]=max(X_raw[i,t],0)`。再按整行 L2 范数缩放：

`X[i,:] = X_plus[i,:] / sqrt(sum_t X_plus[i,t]^2)`。

范数不大于浮点 epsilon 的行删除。没有减全局最小值、没有逐阶段去均值、没有将负向 HGA 当作另一个非负通道，也没有 StandardScaler。这使拟合主要比较**非负响应形状**，而非让绝对振幅大的电极支配目标函数。负向响应信息在 NMF 输入中被舍弃，但原始 HGA 曲线显示可以保留负值。

## 4. 拟合参数与重复初始化

优化目标为 `0.5 × ||X−WH||_F²`，约束 W、H 非负。

| 参数 | canonical 拟合 |
|---|---|
| `n_components` | 最终 K=3 |
| `init` | `nndsvdar` |
| `solver` | `cd`，coordinate descent |
| `beta_loss` | `frobenius`，所检查 sklearn 默认 |
| `tol` | `1e-4` |
| `max_iter` | 5000 |
| regularization | `alpha_W=0`，`alpha_H="same"` 即 0，`l1_ratio=0` |
| `shuffle` | False |
| 根随机种子 | 42 |
| 每个 K 的独立拟合 | 20 次；脚本手动生成各 fit 的 seed |
| 拟合选择 | 选最小 `reconstruction_err_` 的 W/H |

canonical 脚本对 K=1…3 分别跑 20 次；因此“seed=42”是外层随机序列种子，不表示每一轮都给 NMF 同一个 42。保存结果显示 K=3 的 20 次均收敛。历史软件版本未完整嵌入结果；本次检查的 sklearn 为 1.7.1。

NMF 的参数语义可参见 [scikit-learn 官方 NMF 文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)；在线 stable 版本会更新，复现以本项目冻结环境与源码为准。

## 5. 如何选 K：80% 子抽样的共识聚类

入口 `rank_selection.py`。这里虽然命名为 bootstrap，实际是**无放回子抽样**：每次取 80% 电极，255 个电极对应 204 个。

| 参数 | 值 |
|---|---|
| 候选 K | 2、3、4、5、6 |
| 每个 K 的子抽样次数 | 200 |
| 抽样比例 | 0.8，无放回 |
| 种子 | 每 K 使用 `42+1000*K` 派生随机序列，再生成拟合 seed |
| 每次分解 | 相同 NMF 初始化/solver/tol/max_iter；重新拟合子矩阵 |
| 聚类标签 | 对尺度规范化后的 W 取 argmax |
| 层次聚类 | average linkage，距离 `1−consensus` |
| 主选秩指标 | cophenetic correlation 最大；精确相同取较小 K |
| near-tie 报告 | 与最优值差小于 0.02，仅标记，不自动改为近似并列中最小 K |

共识矩阵元素 `C[i,j]` 是两电极在“都被抽到”的轮次里分到同群的比例。对角设 1；从未共同观察的 pair 置 0。距离矩阵对称化后，以 average-linkage 的 cophenetic distance 与原始 pairwise distance 计算相关。

辅助稳定性：最多抽取 500 对子抽样结果，种子 `42+K`；在共同电极数至少 `max(K+1,3)` 时比较聚类，代码先 Hungarian 对齐标签再计算 ARI。ARI 本身对标签置换不敏感，这一步不改变其定义。

另外每个 K 做一轮 full-data fit（seed=42）报告重建误差、`1−||X−WH||²/||X||²` 及 cosine silhouette。这里的 explained energy 未减去 X 的列均值，不能写成 PCA 的 centered explained variance。

| K | Cophenetic | 平均 ARI | Explained energy | Cosine silhouette |
|---:|---:|---:|---:|---:|
| 2 | 0.998463 | 0.974957 | 0.676260 | 0.336291 |
| 3 | **0.998703** | **0.980255** | 0.726678 | 0.318636 |
| 4 | 0.988076 | 0.890054 | 0.749972 | 0.153605 |
| 5 | 0.953993 | 0.690107 | 0.765010 | 0.111489 |
| 6 | 0.965996 | 0.710678 | 0.779496 | 0.097783 |

选择 K=3 的直接规则是 cophenetic 最大，而不是 silhouette 最大；后者在 K=2 更高。K=2 与 K=3 的 cophenetic 仅相差约 0.0002403，2/3/4 被列为 near-tie。应报告三群是此规则下的选择，不能称为“所有指标一致证明唯一最优三群”。Figure 2 cell 14 的 silhouette 图标出来自 `chosen_k.json` 的 K=3，不是根据该曲线峰值选三。

## 6. W/H 尺度规范、硬标签与命名

NMF 存在任意的组件尺度自由度。先对每个组件计算 `s[k]=||H[k,:]||₂`，再设 `H_norm[k,:]=H[k,:]/s[k]`、`W_scaled[:,k]=W[:,k]*s[k]`，保持重建 WH 不变。

电极硬标签为 `argmax_k W_scaled[i,k]`。另存 dominance=`max(W_scaled[i,:])/sum(W_scaled[i,:])`，但没有设置例如 dominance>0.5 的入群门槛，也没有剔除混合载荷电极。H 的范数是在整个拼接时间轴计算，不是每个阶段单独按峰值归一化。

三组件命名仅基于 Stimulus 模板：早期 `[0,0.30]` 均值减去晚期 `[0.50,0.90]` 均值。该分数最低者叫 sustain，中间叫 motor，最高叫 sensory。命名未使用解剖 ROI，也不是通过另一次“运动选择性检验”赋予 motor 标签。名称应作为描述性解释，而非独立证明。

### 当前发布标签的版本差异

当前 `channel_assignments.csv` 有 255 行，sensory=39、sustain=80、motor=136。但下列 6 个电极的保存权重最大值仍对应 motor，`pc_scores.csv` 也保留 motor，而当前标签为 sustain：

`D0057_RAI3-4`、`D0057_RAI4-5`、`D0057_RAI5-6`、`D0084_RAI4-5`、`D0090_RIA4-5`、`D0102_RFAI1-2`。

因此仅执行上述 argmax 规则会得到旧的 39/74/142，不能精确重现当前标签。差异及当前表快照见 [nmf_label_differences.csv](evidence/nmf_label_differences.csv)、[nmf_channel_assignments_snapshot.csv](evidence/nmf_channel_assignments_snapshot.csv)。没有找到这六项变化的完整理由记录，不能自行解释为专家复核或新一轮 NMF。

当前 assignment SHA-256 为 `d5ef9cb1cb9a135e76e1b19caa679a900ec3871adbbb82176b6405a1aecb7ac7`；现有 functional decoding 输入记录的为 `0b28f2105a7b08d0c91cfdb94686e6c6f266c9e7e5acc1e32971936695b00b97`。需把“模型原始标签”和“发布标签/覆盖规则”分别冻结。

## 7. PCA 与 K-means 的独立对照

实现：[waveform_pca.py](../../src/nmf/waveform_pca.py)、[pc_clustering.py](../../src/nmf/pc_clustering.py)，Figure 2 cell 16。PCA 对相同预处理的矩阵使用 full SVD，sklearn 内部按特征中心化；没有额外 StandardScaler 或 whitening。

保存 20 个 PC 作 scree，选**各自解释比例严格大于 5%** 的 PC，当前为前三个：27.3578%、11.4688%、5.5574%，合计约 44.3840%。不是选累计达到 95% 的 PCs。

Figure 2 中 KMeans 在这三个 PC score 上用 `n_clusters=3`、`n_init=50`、`random_state=42`；其余所检查默认为 k-means++、Lloyd、max_iter=300、tol=1e-4。Hungarian 对齐用于显示类别名称，与保存的 PC 表内 NMF 标签比较，缓存 ARI≈0.749。

这是对表示/分群一致性的辅助检查，没有取代 NMF 的 canonical 标签，也不是选秩的主规则。PC 表使用旧标签，需先解决版本差异再把这个 ARI 声称为与当前标签的精确一致率。

## 8. Figure 2 的波形、热图和空间统计

**HGA 均值曲线**：对原始 z-score 表按功能群显示，不将负值截断；常见 seaborn 默认 95% bootstrap CI、1000 次、seed 未固定。观测行可能是 electrode×task，而不是先按被试平均；该区间不能称为被试层级 95% CI。

**显示窗与拟合窗不同**：曲线常显示 Stimulus `[-0.5,0.6]`、Delay `[0,1]`、Go `[-1,0]`、Response `[-0.5,1.5]`。NMF 的 Go 拟合段却为 `(0,1)`。H 模板只在真实拟合数据存在的区间有值，扩大坐标轴不会增加模型支持的时间段。

**热图**：按 channel×time×phase×cluster 聚合 mean value、any mask；非显著点置 NaN。每个阶段按该阶段显示窗内首次显著时间独立排序，同一横向位置不保证是同一电极。显示色阶 vmin=0、vmax=1，属于可视化裁色。

**解剖饼图**：细分六回 AP/ASG/MSG/PSG/ALG/PLG；合并三组时为 ASG、AP+MSG+PSG、ALG+PLG。通道解剖标签从打包表取众数，匹配不足时使用被试 parcellation 文件回退。这里主要是组成描述，没有每张饼图的独立显著性检验。

**左右侧比例**：分母是四任务 parcellation 中去重的 implanted pure insula（subject×channel），排 D0121，但未同步排除那 26 个 NMF 通道；分子来自 NMF assignment，已排这 26 个。因此是当前实现的分母规则，不是统一 QC 后所有电极的入群率。缓存 L=281、R=179；对每群做 2×2（assigned/not assigned × L/R）双侧 Fisher，并对三群已入群计数做 3×2 χ²；三个 Fisher 之间未额外校正。

**空间 KDE**：在脑表面截图投影的二维像素坐标上高斯平滑，sigma=25 pixels；每群以自身最大密度的 0.18 作轮廓阈值，填充 alpha=0.12。图像尺寸约 800 px；该 sigma 不是 25 mm。每群峰值各自归一化后不能比较群间绝对密度。此图不提供空间聚集的 p 值，也不是体积内三维概率图。

## 9. 可复用步骤与验收

新项目保持“通道×拼接时间”的数据契约，显式设置事件窗、覆盖阈值、插值、非负化、归一化、候选 K、抽样单位及命名规则。先保存 `X_raw/X_processed`、行顺序和阶段列索引，再做选秩与最终拟合，最后冻结 H、W、component-name map、原始 argmax 标签及任何人工覆盖。

最低检查：所有输入有限且非负；每行 L2≈1；阶段列数与采样点吻合；X 行名单与 W 一致；规范化前后 WH 不变；hard label 可由 W 重算或有逐项覆盖记录；子抽样次数、收敛数和完整选秩曲线可追溯。

若希望推断到新被试，建议另做 subject-level 留出/重抽样稳定性，而不是只重抽电极；若希望将同一功能模板迁移到新项目，可冻结 H 后对新数据拟合非负 W，并另行验证适配性。这两项是扩展建议，**不是 Figure 2 已完成的主分析**。NNLS projection、whole-brain projection 与 connectivity 分支虽在仓库中存在，未被混入这里的五图主方法。
