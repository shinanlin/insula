# Figure 3–6 更新记录（2026-10-18）

## 当前交付状态

| Figure | 已完成 | 待完成与原因 |
|---|---|---|
| 3 | 5 张原版空间／配对统计 SVG，已换成当前 231 位点及新功能群；真实执行输出已写回 publication notebook 的相应单元 | 全部原始波形、热图和波形 SI 需要原单试次 HGA；该部分未生成 |
| 4 | 隔离的输入重建和正式解码入口、逐群输入／试次名单记录、旧分析核心核对 | PS、LD 输入准备在拟合分类器之前失败；没有正式新解码结果 |
| 5 | 已落实批准的按任务解剖名单、Repeat+sound、electrode mean±SEM、保留 Go；针对性检查通过 | 缺原单试次 HGA，尚未拟合正式 RT 模型或生成 Figure 5 |
| 6 | 三阶段固定 H 全脑投影、表面缓存、完整 notebook 和 3 张 SVG | 已完成本次更新 |

Figure 1/2 和现有 NMF 输入、因子、标签、秩评估等文件已通过运行前后 SHA-256 核对，保持不变。工作区为 `/cwork/ns458/insula-paper`；旧 `insula-functional` 的工作文件只读。

## Figure 3：已生成的输出

- `img/fig3/fig3_decision_gt_repeat_spatial_delay.svg`
- `img/fig3/fig3_delay_decision_gt_repeat_spatial.svg`
- `img/fig3/fig3_output_repeat_gt_passive_spatial.svg`
- `img/fig3/fig3_paired_delay_p75.svg`
- `img/fig3/fig3_pn_repeat_sound_vs_image_spatial.svg`

使用原 notebook 的绘图单元和原保存统计；原 Mean/P75 指标、原单电极 p 值／FDR 范围、原群体 Wilcoxon 和显示设置保留。没有以缩小后的位点名单重新定义 FDR family。

更新后的 Delay P75 配对位点数：

| 比较 | sensory | sustained | motor | 总数 |
|---|---:|---:|---:|---:|
| PN Repeat–Passive | 12 | 33 | 49 | 94 |
| LD Decision–Repeat | 18 | 42 | 77 | 137 |

PN stimulus 空间图：sound 48 个（12/20/16），image 23 个（1/12/10），顺序均为 sensory/sustained/motor。

逐位点表和统计位于 `results/fig3/available_*.csv`；部分执行记录为 `results/fig3/available_execution_validation.json`。`vizpub/fig3.ipynb` 标记为部分完成，未执行的波形单元保持未执行状态。

## Figure 6：模型和范围

- 使用 `results/nmf/H_by_phase.csv`，Stimulus/Delay/Response 共 399 个时间特征；不重新拟合岛叶 NMF。
- 94 位参与者、8,477 个位点完成投影；沿用原显示筛选后，表面图候选集为 93 位参与者、4,440 个位点。
- 固定 H 的 NNLS、原空间参数、subject-balanced 表面估计和 top 50% 显示方式保持原实现。解释为波形相似性。
- 原投影图的筛选排除的是 discovery 位点，未排除所有其他 AIC/PIC；候选图集中仍有 107 个其他岛叶位点。本轮沿用这条原规则，未新增解剖排除。
- 当前 231 个 discovery 位点中，230 个在全脑 packaged 输入中可重建；`D0060_LI1-2` 缺该投影输入，已单列。其岛叶 NMF 输入仍保留，未改变 Figure 2。
- CSV 为 trial-averaged 波形，未从该源虚构 trial 数。

输出：

- `img/fig6/fig6_nmf_templates_nogo.svg`
- `img/fig6/wholebrain_surface_motifs_focused_combined_pial_top50.svg`
- `img/fig6/wholebrain_surface_motifs_focused_aparc_share.svg`

输入名单、参与者×区域数量和检查位于 `results/fig6/`；完整执行后的 notebook 为 `vizpub/fig6.ipynb`。

## Figure 4/5 已固定的运行规则

- Figure 4 原分类器、解码专用 PCA、CV、置换、簇校正均保留。原结果中已存在的功能群分析对应 33 个拟合项：PS articulator 固定窗 9 项、LD lexicality 时间曲线 18 项、LD Delay 双向跨条件 6 项。没有增加旧版未计算的 Stimulus/Response 跨条件检验。解剖对照仅在确认输入／方法未变后复制并保存校验记录。
- Figure 4 控制区保留本图原 palette，包括原 SMC/IFG 色值；未借本次数据刷新修改样式。
- Figure 5 使用三个相关任务的当前解剖合格位点，独立于 HGA 响应显著性；模型以 Repeat+sound 为输入，NMF 分群汇总仅使用具有当前标签的子集。其余建模位点单列。
- Figure 5 HGA 为单电极先平均试次、同一电极内等权平均可用任务，再计算电极均值±SEM。原 subject/cluster 的快慢 RT 划分和有限试次筛选保留；RT 相关统计继续原被试级汇总与检验。
- Go 保留。Delay/Go 联合校正，Response 单独校正，CV、窗口、置换次数与原版一致。
- `approved_method_checks.json` 通过代码编译、解剖名单含非显著位点、原 RT 划分保留、电极权重与 SEM 的针对性检查。该检查不代表已完成真实数据的 RT／解码拟合。

## 输入缺失及备份检查

本批参与者的原路径检查结果：

- PhonemeSequence：`epoch(band)(zscore)`／`epoch(band)(sig)` 均未找到。
- LexicalDelay：任务目录仅剩其他参与者的部分处理文件；本批名单中的对应文件均未找到。
- LexicalNoDelay：未找到对应 epoch 文件。
- PictureNaming：本批目前只找到 sound Passive Response 的部分文件；Repeat、其他所需阶段／条件未齐。

逐 subject×task×phase×condition×modality 的探查列表为 `docs/evidence/figures3_6_refresh_2026-10-17/original_epoch_availability.csv`；缺失位置为 `missing_original_inputs.csv`。这是一份探查组合表，含部分可选的阶段／条件组合，不应将其总行数解释成所有原本必然存在的文件数。

已通过当前 Globus 登录检查 Windows/Box 下相关任务目录及 PictureNaming_derivatives。找到的是其他版本（例如 PS 的 perception/production `.fif` 和 PN 的 CAR 文件），尚未找到完整同版的 bipolar zscore/sig `.h5`。没有将这些其他版本代入本次分析，也未重新定义预处理。

需恢复／确认的目录形状：

`<BIDS>/derivatives/epoch(bipolar)/sub-<ID>/epoch(band)(zscore)/`

`<BIDS>/derivatives/epoch(bipolar)/sub-<ID>/epoch(band)(sig)/`

已经向用户询问其他备份位置，待该信息补齐后继续。Figure 4 未运行的依赖数组已取消，不留无法启动的占位作业。Figure 4/5 的代码和启动入口保留，尚未声称完成正式拟合。

## 运行记录

| 作业 | 作用 | 结果 |
|---|---|---|
| 58872392 | Figure 3 完整 notebook | 原始 epoch 缺失，停止 |
| 58872400 | Figure 4 PS/LD 输入准备 | 原始 epoch 缺失，停止于拟合前 |
| 58872436 | Figure 6 全脑投影、缓存与绘图 | 完成，7 分 5 秒 |
| 58872530 | Figure 3 可用统计／空间单元 | 完成，1 分 30 秒 |
| 58872564 | Figure 6 notebook 单独复核 | 完成，26 秒 |
| 58881008 | Figure 5 已批准方法的针对性检查 | 完成，通过 |
| 58881078 | 最终输入／输出与冻结基线核对 | 完成，通过 |

日志位于 `logs/slurm/`。最终机器可读状态为 `docs/evidence/figures3_6_refresh_2026-10-17/delivery_status.json`，其中明确区分完成和待补输入的部分。
