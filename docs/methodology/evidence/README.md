# 方法审计证据附件

这些文件是 2026-09-13 的只读核对快照。它们支持方法文档中的参数和版本判断，未复制原始神经时序，未重新运行模型。路径保留实际解析位置，可能包含当前仓库以外的 BIDS、预处理 worktree 和旧 `insula/results`。

| 文件 | 含义 |
|---|---|
| [audit_summary.json](audit_summary.json) | 审计范围：92 cells、68 decoding results、72 prepared inputs、300 RT files；差异计数 |
| [verification.json](verification.json) | 已通过的文档链接、JSON、源码/cell 指纹、样本数与参数一致性检查；没有重跑模型 |
| [figure_panel_map.csv](figure_panel_map.csv) | 按 22 组图/panel 对应具体 cell、输入、生产模块及方法章节 |
| [notebook_cells.csv](notebook_cells.csv) | 五个 Notebook 的 0-based cell 索引、cell ID、执行编号、源码哈希、输出存在性 |
| [source_fingerprints.csv](source_fingerprints.csv) | 94 个关键源码/小型结果/Notebook 的 SHA-256、路径、大小和修改时间 |
| [source_symbols.csv](source_symbols.csv) | Python function/class 的具体行号，辅助定位方法实现 |
| [environment_versions.json](environment_versions.json) | 本次检查环境版本；不保证每次历史运行版本相同 |
| [decoding_results.csv](decoding_results.csv) | 每份结果的分支、ROI、phase、窗口、PCA/CV/置换设置、shape、实际文件路径 |
| [decoding_results_metadata.json](decoding_results_metadata.json) | 上述 H5 的完整属性、dataset shapes、少量时间/通道/类别数组；无 score 大数组 |
| [decoding_inputs.csv](decoding_inputs.csv) | 实际 trial/channel/item 数、类别计数、输入类型、通道列表与 assignment hash |
| [decoding_inputs_metadata.json](decoding_inputs_metadata.json) | prepared H5 选择/QC 属性及通道名单；不含 X 神经数组 |
| [decoding_assignment_mismatches.csv](decoding_assignment_mismatches.csv) | 34 条 input-file×channel 标签差异；重复条件/阶段不去重 |
| [rt_results.csv](rt_results.csv) | 300 个 RT 文件的任务/被试/阶段、trial/item/fold/channel 数与实际时间窗 |
| [rt_results_metadata.json](rt_results_metadata.json) | RT attrs、shapes、fold/recording 计数，共享时间向量保存一次；无 permutation/神经数组 |
| [rt_summary_assignment_mismatches.csv](rt_summary_assignment_mismatches.csv) | 53 条 summary-row 标签差异；42 个 electrode 行、11 个 cluster 行 |
| [nmf_channel_assignments_snapshot.csv](nmf_channel_assignments_snapshot.csv) | 当前 255 行发布表的原样快照，包括载荷和空间信息 |
| [nmf_label_differences.csv](nmf_label_differences.csv) | 当前标签、loading argmax 与 PC 保存标签不同的 6 个电极 |
| [nmf_exclude_channels.txt](nmf_exclude_channels.txt) | canonical NMF 的 26 通道排除名单 |
| [nmf_nmf_manifest.json](nmf_nmf_manifest.json) | canonical 发布信息原样快照 |
| [nmf_rank_selection_meta.json](nmf_rank_selection_meta.json) | 选秩配置、255 行名单与输入维度 |
| [nmf_rank_selection_metrics.csv](nmf_rank_selection_metrics.csv) | K=2…6 的完整选秩指标 |
| [nmf_chosen_k.json](nmf_chosen_k.json) | K=3 的选择和 near-tie 信息 |
| [nmf_model_selection_metrics.csv](nmf_model_selection_metrics.csv) | canonical 多初始化拟合质量/收敛摘要 |
| [nmf_pc_clustering_meta.json](nmf_pc_clustering_meta.json) | PCA 输入、解释比例及另存的聚类扫描元数据 |

`source_fingerprints.csv` 对源码及小型结果文件计算完整内容哈希；对数 GB 的 H5 只检查元数据，不声称完成全文件内容一致性验证。CSV 中复杂字段以 JSON 字符串保存。`*_mismatches.csv` 的行数是文件/记录层级，不能直接当成独立电极数。

Notebook cell 索引是 0-based；源码行号和 CSV 行号是 1-based。Notebook 的 execution_count 可以不连续或逆序，不能据其推断完整分析生成顺序。源码首行常是 import 或注释，需要用 cell ID 与相邻 markdown 一起定位。

NMF/PCA 的部分 metadata 还记录了未被五张主图全部使用的辅助扫描。方法正文明确区分 Figure 2 实际 KMeans 对照和这些额外分析；文件存在不等于它就是图的输入。

环境快照：NumPy 1.26.4、SciPy 1.16.1、scikit-learn 1.7.1、pandas 2.2.3、MNE 1.10.1、MNE-BIDS 0.16.0、h5py 3.13.0、IEEG 0.7.0、Himalaya 0.4.8、seaborn 0.13.2、Matplotlib 3.10.1。IEEG 为外部 editable checkout，其文件哈希比单一版本号更有辨识力。
