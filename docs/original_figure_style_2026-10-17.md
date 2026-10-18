# Figure 1／2 原版绘图约定恢复

参考原工作区 `/hpc/group/coganlab/nanlinshi/insula-functional/vizpub/fig1.ipynb` 和 `fig2.ipynb`。当前采样／显著名单仍为 348／231，主分析为 sound Repeat，NMF 仍是当前三阶段 k=3。此次恢复绘图呈现，不重新拟合 NMF。

## 热图

- 恢复原版 `agg.loc[~agg['mask'], 'value'] = np.nan`，非显著时间点留白。
- 恢复每个阶段、每个 cluster 内按首次显著时间排序；没有显著时间的电极排在后面。
- 保持原版 sensory／sustained／motor 行组、按电极数分配行高、颜色范围 0–1、画布尺寸、阶段列宽及坐标轴。
- 使用保存的 sound Repeat mask，同一位点跨可用任务取 any。原打包 CSV 的逐点 mask 优先，完整统计文件仅补打包时遗漏的位点／时间。
- 每阶段窗口内显著位点数必须精确重现当前筛选清单。缺少 mask 的位置保持无证据标记并留白，不从 HGA 或 NMF loading 推断显著性。

## 其他图

- 群体 HGA 曲线恢复原版画布、边距、固定纵轴、0.5-z 比例尺、虚线、字体、线宽和配色。用户后续明确要求：直接在 electrode 层面平均，保留 SEM，均值和 SEM 均以电极为单位。
- NMF 成分图恢复原版 renderer，使用当前三个阶段的数据；显示上下文与拟合窗口分别记录。
- 秩评估面板恢复原版 cosine silhouette 图，突出实际拟合的 k=3。cophenetic 最大值仍是独立的秩评估记录。
- PCA 恢复原版边际直方图边界和刻度；嵌入维度按单个 PC 解释方差 >5% 的旧规则选择，取消累计 90% 规则。载荷分布恢复 dominance 横轴。空间图恢复原版尺寸、图例位置和留白。
- 左右侧图恢复原版线图及其电极计数统计呈现，分母来自当前核对后的采样名单。统计单位为位点；未确认响应和缺响应证据仍须在清单中区分，不能将这些计数统计当作参与者层面的推断。
- 原版没有对应面板的新增参与者×任务计数表保留；Figure 1 平均曲线采用原版 Figure 2 群体曲线的画布与坐标样式。

## 固定 SEEG 示例

原版物理触点示意图包含 D0064 的 3D 视图和 D0075 的 MRI／MAPER 切片。当前 D0075 的原 MAPER 文件不在原路径，因此直接保留原版 `fig1_insula_electrode_schematic_seeg.svg`，逐字节校验，不替换参与者、切片分割或触点画法。该图是固定解剖示例，不用于本轮总体采样／显著数量计算。

## 证据

- [恢复的绘图单元及来源哈希](evidence/original_figure_style_2026-10-17/restored_cells.json)
- Figure 2 的 `site_significance_masks.csv.gz`、`site_significance_manifest.json`：逐点 mask 与来源。
- Figure 2 的 `heatmap_row_order.csv`、`heatmap_plot_validation.json`：实际行序与留白核对。
- Figure 1／2 的 `execution_validation.json`：完整 Notebook 执行、当前名单、sound-only 条件及 SVG 导出验证。

重绘入口：先执行 `scripts/slurm/execute_fig1_notebook.sh`，再执行 `scripts/slurm/redraw_fig2_notebook.sh`；后者核对 NMF 输入未改变、恢复现有 mask，并执行 Figure 2。
