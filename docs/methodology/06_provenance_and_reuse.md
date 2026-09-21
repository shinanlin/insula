# 06 证据、复现差异与跨项目操作规范

## 1. 本次检查的边界

检查了五个 Notebook 的所有 cell 源码及保存的文本输出，沿读取路径反查本仓库和外部预处理 worktree 的生产代码；另读取 68 个 decoding result H5、72 个 decoding input H5、300 个 RT H5 的属性/维度及少量标签、通道、时间信息，并核对 NMF CSV/JSON。

没有重跑模型或置换，没有将缓存图当成此次生成。证据附件不复制原始神经时序或巨型 permutation arrays。源码指纹只证明本次审计时的文件内容；当前环境版本也不自动等于每个历史运行版本。

本次仓库 HEAD 为 `33659ba50e2ddce2159e182de2e2d650331ee12f`，工作树含审计前已存在的修改/未跟踪文件。因此只记 HEAD 不足以复现；还保存了关键文件 SHA-256 和 Notebook 每个 cell 的 source SHA-256。

## 2. 必须保留的复现差异

| 项目 | 直接证据/现象 | 对解释和复现的影响 |
|---|---|---|
| NMF 六个标签 | 当前 assignment 与 loading argmax/PC 表不同 | 仅凭 canonical 算法不能再现当前 39/80/136；需另存覆盖记录 |
| functional decoding 标签 | prepared H5 的 assignment hash 与当前表不同 | 新标签着色不代表旧特征集已重新训练 |
| 固定窗解剖 vs 功能 | H5 的 PCA 阈值 .95 vs .80；Response 裁窗不同 | 性能比较包含输入/降维配置差异 |
| fixed-window 重复 | 同存 accuracy[5] 与 accuracy_repeats[30,5] | 图只读前者；bar、p、confusion 都属第一轮 |
| Figure 1 旧相对目录 | 当前四个 `results/{Task}(bipolar)(hammers)` 不存在 | 当前 Notebook 不能仅靠此 checkout 原样重跑；缓存数保留缓存身份 |
| 旧结果根目录 | Figure 3/4 部分文件来自相邻 `insula/results` | 发布包还依赖外部结果，详见逐文件路径 |
| RT summary 版本 | 旧预测汇总 42 行 electrode、11 行 cluster 的标签与当前 assignment 不同；新 encoding 电极表一致 | 涉及同 6 个电极；不能仅因同目录就认定同一冻结版本 |
| RT 正确性匹配 | 源码 contains(CORRECT) 也会匹配 INCORRECT；已存 target 标签无 INCORRECT | 属迁移实现风险，不能推断现有结果已污染 |
| 历史参数缺项 | 预处理 stats、univariate CSV、旧 decoding H5 等记录不完整 | 当前代码可以补充算法描述，不能补造历史运行证明 |

精确差异文件：[NMF](evidence/nmf_label_differences.csv)、[decoding](evidence/decoding_assignment_mismatches.csv)、[RT 汇总标签](evidence/rt_summary_assignment_mismatches.csv)。文件为空（仅表头）表示该项检查没有差异，不表示未检查。

## 3. 统计方法与校正 family 总表

| 分支 | 置换/检验单位 | 原假设方向 | 多重比较范围 |
|---|---|---|---|
| HGA vs baseline | trial；每通道内时间簇 | HGA 增加 | 当前 IEEG 实现的每通道时间簇；forming/cluster 默认 .1，另叠逐点 .05 电极筛选 |
| NMF rank | 无放回抽电极；不是显著性检验 | 不适用 | cophenetic/ARI 等描述稳定性，不输出“群存在”的 p |
| Mean/P75 单电极条件对比 | 合并两条件的 trials 重排 | 双侧 | BH：subject×phase×contrast 全 tested channels |
| Figure 3 群内配对 | 电极 paired Wilcoxon | a>b | 六项群检验未再校正；同被试电极依赖未建模 |
| Figure 2 laterality | electrode 2×2 Fisher；3×2 χ² | 双侧/总体关联 | 三个 Fisher 未再校正 |
| PS 固定窗 SVM | 每 fold 训练 trial 标签置换 | balanced accuracy 增加 | 单 ROI×phase p；跨 panels 未校正 |
| LD 滑窗 LDA | 全局词标签映射 | AUC 增加 | 一 ROI×phase×condition 的一维时间簇长度 |
| LD 跨条件 LDA | 两域共用词标签映射 | AUC 增加 | 一 ROI×direction 的二维 8 邻接簇质量 |
| RT Ridge | 外层训练 trial 的 y 重排 | prediction r 增加 | 每 subject×task 内所有电极/时间，Delay+Go 联合；Response 另批 |
| Figure 5 振幅方向 | 汇总到被试后 Wilcoxon | amplitude–logRT r<0 | 三群未再校正；选择后描述性检验 |

## 4. 跨项目可直接遵循的步骤

### 步骤 A：先定义数据与科学问题

写清任务、条件、事件零点、baseline、准确性/RT 筛选、单位转换，以及目标是描述既有样本还是泛化到新词/新被试。建立 subject/task/recording/trial/item/channel 字典，验证 ID 唯一性和事件对齐。

### 步骤 B：冻结预处理与多个电极集合

保存全采样通道、解剖 pure 集合、响应显著集合、NMF 集合、每个 decoding phase 的可用集合。保留筛选前后计数和原因。使用明确的处理配置和依赖版本，不从文件名猜测 power/amplitude、baseline 或统计阈值。

### 步骤 C：拟合和发布 NMF

按固定阶段列索引构造矩阵，记录插值/整流/L2 顺序；做完整选秩曲线及近似并列报告，再重复初始化拟合选定 K。冻结 W/H、组件缩放、名称映射、argmax 标签、dominance 和任何覆盖表。不要直接在唯一的 assignment 文件中静默改标签。

### 步骤 D：独立定义每个下游分支

条件对比固定 a/b、Mean/P75 及 FDR family。Decoding 明确窗、特征、CV group、预处理拟合范围、评分和置换映射。RT 明确事件配对、log 单位、item folds、alpha 网格与 joint phase family。不要借用另一分支的默认配置。

### 步骤 E：将新项目改进与历史复现分成版本

可选的改进包括：按 subject 重抽样稳定性、外层训练内发现 NMF/响应电极、ROI 匹配特征数、严格嵌套 imputer/scaler、统一多个图的校正 family、独立验证所选 RT 窗口。为每项改进记录新版本、原因与需要重算的结果，保留原复现版本。

### 步骤 F：只从冻结结果绘图

绘图代码声明读哪个 score、哪个标签 hash、按什么单位平均、误差条定义及哪些筛选仅为显示。保存实际窗 start/end/center，不能只用名义 ms。依项目要求，新输出图使用 SVG。运行置换、全体模型、Notebook 执行等重计算时使用 Slurm；登录节点只做轻量检查和编辑。

## 5. 推荐的输出目录与记录字段

以下是迁移模板，**不是声称现有文件已经全部采用这些字段**。

```text
analysis_release/<version>/
  config/parameters.json
  provenance/{source_hashes,environment,input_manifest}.json
  cohorts/{sampled,pure_roi,responsive,nmf,decoding_by_phase}.csv
  nmf/{X_raw,X_processed,W,H}.npz
  nmf/{feature_index,component_names,raw_assignments,label_overrides}.csv
  nmf/{rank_metrics,fit_metrics}.csv
  contrasts/<task>/<subject>/<contrast>.csv
  decoding/<branch>/<roi>/<phase>/<condition>.h5
  rt/<task>/<subject>/<phase>.h5
  summaries/
  figures/*.svg
```

每个模型结果至少存：`run_id, source_sha256, config_sha256, input_sha256/manifest_id, assignment_sha256, subject/task/phase/condition, channels, trial_ids, item_ids, labels/target, sfreq, actual_window_indices, fold_ids, estimator_params, seeds, scores, permutation_scheme, n_perm, point_alpha, cluster_alpha, cluster_statistic, correction_family`。

对于 NMF，另存 component permutation/scale；对于 CV，另存每折训练/测试 ID；对于 RT，另存 Go/Response 原始事件和 alpha 选择；对于图，另存 source files、聚合单位、误差类型、显示筛选和标签版本。

## 6. 不需重分析即可进行的验收

- 逐文件检查 shape、class counts、通道和 trial 顺序；按 item 的 CV train/test 交集必须为空。
- 用保存的 NMF 载荷重算 argmax，与发布标签比较；差异必须能由 override 表逐项解释。
- 检查同一图的 prepared input 和赋色标签 hash 是否一致，sample counts 是否使用正确分母。
- 检查 fixed-window score 是否为第一轮或重复均值；p/null 必须匹配该观察统计量。
- 检查时间向量是否使用实际采样点，以及端点规则、window end/center 是否标注一致。
- 检查每个统计输出的 correction family 是否与图注一致；不能仅写一个没有范围的“校正后 p<0.05”。

这些检查用于发现发布包内部不一致，不能替代重跑或独立验证模型。当前文档与附件完成的是源码/元数据层面的核对。
