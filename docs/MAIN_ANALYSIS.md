# 主分析口径：Repeat + sound-only

## Figure 3 正文与 supplementary 统一入口（2026-11-01 用户批准）

- 非空间 weight 构成草图（2026-11-02 用户要求 manifold 风格美化）：仍使用 `vizpub/diagnostics/fig3_weight_distributions.ipynb` 的两张三角坐标图，固定 PN21／14人、LD27／16人的三个冻结 loading 比例。保持原始点位；按用户后续要求去掉组平均菱形及对应图例，数值均值仍保留在导出表中，弱化框架，增加低饱和度密度云和等高线。密度采用共同σ=0.09（三角边长1）的等权高斯显示核，各组除以电极数，再共享峰值色阶；只显示有效三元权重范围，等高线为共同峰值的20／40／60／80%。此为描述性密度美化，不代表新embedding、置信区间或显著性；不改变坐标、权重、NMF或正文／SI原图。原始三元坐标和密度网格均保存用于溯源。 用户随后要求加入component颜色：各电极按三个权重比例连续混合 Sensory蓝#2369BD／Sustained金#C4A35A／Motor红#A9373B，保留纯成分颜色图例；密度背景改为中性灰。重着色前后已逐数组核对密度网格，并核对权重、点位和均值表完全一致。 用户随后要求同样展示显著Sound > Image：在同一诊断notebook追加正文响应池的单向18位点／12人独立SVG，双向显著1位点仍单列；沿用原权重、颜色、σ和前两图共同密度峰值，正文／SI不改。用户最新要求同步更新三张诊断图的辅助线：删除竖直虚线及50%倒三角连线，改为等边三角形中心（各成分1/3）分别连接三个顶点的三条实线；保留边缘50%刻度。这三个几何扇区不等同于最大weight功能分类；点位、权重、颜色与密度数组保持不变。 用户批准新增叠加图：同一三元坐标中以灰色表示未检出差异21位点／14人，蓝色表示单向Sound > Image18位点／12人；复用原点位与密度数组，共用σ、峰值及等高线阈值。仅此叠加图中，点色表示条件分组，点大小14、透明度0.60；两组密度云及对应色条透明度均0.40。保留三色顶点名称和中心连顶点的辅助线，不显示均值菱形或顶点100%。同一notebook追加独立SVG，原有图和结果保持不变。

- 正文分类图外观修正（用户要求）：恢复两个圆自然相交的常规 Venn 外观，取消区域外移、人工间隙及各区域单独描边，恢复原有两个整圆的轮廓。作者自行在 Adobe Illustrator 中分开区域；保留不改变外观的填色／数字／标签分组。形状、面积比例、数量与统计不变。正文 notebook 的两张输出同步更新；曲线、脑图、SI 及 NMF 保持原样。更新入口仍为 `scripts/refresh_figure3_venn.py`。

- 修改前已确认旧正文 notebook 与 Git 内容完全一致，并建立检查点 `5924930849762b209838fadf4f991833597086ad`，保存旧正文及当时的 Figure3 代码和 SI。
- 正文唯一入口：`vizpub/fig3.ipynb`。保留条件差异分类示意、功能着色空间图、PN21 位点两条件波形及平均 HGA heatmap、LD27 位点两条件波形及差值 heatmap。仍为独立子图，保持已批准的颜色、窗口、行高、mask 和统计结果。输出统一到 `img/fig3/`、`results/fig3/`，共享入口为 `src/figure3.py`。
- SI 统一入口（用户要求合并）：`vizpub/supplementary/fig3_supplementary.ipynb`。按 Picture Naming、Lexical Delay、Repeat–Passive 组织，前两组各自的采样／响应信息和完整 HGA 放在一起。原采样与完整 HGA 两个 notebook 已合并移除。采样与 baseline 响应部分：先列完整解剖采样 PN162／28 人、LD214／38 人；再列 native QC 通过且配对统计完整128／27、187／35；功能分类后81／24、135／35；目标阶段响应池44／21、71／27。完整采样是覆盖元数据，功能统计排除 Unassigned。Venn 中央表示两条件均相对 baseline 响应：PN Sound35、Image21、交集12；LD Decision61、Repeat58、交集48。与正文中央“未检出条件差异”的分类图严格区分。
- 同一 SI notebook 的完整 HGA 部分：保留 Stimulus／Delay／Response × Sensory／Sustained／Motor 的两条件波形与原始 HGA heatmap，沿用原版函数体、baseline mask 空白规则、P95 色阶及条件配色。Sound–Image81 位点、Decision–Repeat135 位点；Repeat–Passive 保留原缓存与当前功能样本的64 位点。Passive Response 沿用 nominal 对齐，不视为实际发声。
- 所有四任务采样与任务交叠图继续保留在独立 `vizpub/supplementary/task_sampling.ipynb`。不混入正文 Figure3。
- 已淘汰的 weight 混色／高斯展示及重复入口从当前工作流移除。必要的原始统计、波形、冻结 NMF 和排除记录保留；旧代码可从 Git 检查点恢复。正文 Figure1／2／4／5／6 与 NMF 不改。

**用户已明确：主分析仅使用 sound 条件。该要求同时约束有效／显著位点筛选、Figure 1、NMF 输入和依赖这些分群的后续主分析。**

## 数据条件

- 使用 `description=Repeat` 且 `modality=sound` 的记录。不能将 image/picture、text 或其他条件的显著性并入主分析的有效位点名单，也不能把它们的波形平均到主 NMF 输入中。
- PictureNaming 是任务名称，其中也有 sound 试次；按记录的 modality 筛选，不能直接删除整个 PictureNaming 任务。
- image/picture 和 text 用于单独的条件比较，不改变 sound 主分析的入组资格。
- 采样分母是具有 sound 记录且符合已确认中点、双端点解剖规则的双极位点。相同参与者、相同双极位点跨任务去重；逐参与者×任务的名单同时保存。
- **端点 Unknown 的最新决定：恢复纳入。**当端点 Hammers 为 0，原生 FreeSurfer 的 aparc／aseg 均为 Unknown、ribbon 为 0，且没有 MAPER 明确组织分类或标签／身份冲突时，按用户要求保留；清单标记 `has_accepted_unknown_endpoint` 和具体触点，不能称为已确认白质。中心仍需满足岛叶要求。明确 CSF、非岛叶灰质和坐标／植入版本冲突不因本例外恢复。Hammers 为 0、FreeSurfer 为岛叶灰质的另一类位点继续单列，等待对其具体标签的讨论。
- 显著响应是 sound Repeat 在 Stimulus **0<t<0.6 s**、Delay **0<t<1 s** 或 Response **−0.5<t<0.5 s** 任一阶段／任务的确认阳性。缺失 sound 统计不能用 image/text 结果填补，也不能判为阴性。
- NMF 仅从该 sound 阳性名单出发，再核对三个阶段的 sound 波形覆盖。缺整段不能补零、不能用其他模态波形代替；额外损失及原因须列出。
- NMF 拟合窗口：Stimulus **0–0.6 s**，Delay **0–1 s**，Response **−0.5–1 s**。Response **−0.5–1.5 s**是曲线显示上下文，不是拟合窗口。
- 同一 site/task 的 sound 波形先平均现有试次，同一位点内对可用任务等权平均。**群体曲线直接在 electrode 层面等权平均，每个双极位点贡献一条波形；不先平均参与者内的位点。误差带保留 SEM，以电极为计算单位，即位点间样本标准差除以位点数平方根。**这是用户在新旧差异审阅后明确要求的规则。
- **PCA／k-means 恢复旧规则：按单个 PC 解释方差 >5% 选择嵌入维度，调用旧版 `n_pcs_above_threshold`；不用累计解释方差 90%。**保留原函数至少两个 PC 的兜底逻辑，不把维度硬编码为 3。

## 颜色约定

| 解剖区域 | 固定色值 |
|---|---|
| Insula | `#6A5ACD` |
| STG | `#20B2AA` |
| SMC | **`#F39B35`** |
| IFG | `#2369BD` |
| Other | `#D3D3D3` |

这些值来自原 insula-functional Figure 1 Notebook 及实际导出 SVG。区域类别需固定映射到颜色，不能因当前类别缺失或排序变化而移动配色。
NMF 功能成分沿用 sensory `#2369BD`、sustained `#C4A35A`、motor `#A9373B`；它们不代表 SMC 等解剖区域。

## 原版绘图约定

- Figure 1／2 的画法以原 insula-functional Notebook 为依据，刷新名单和数据时不更改画布、坐标轴、比例尺、配色、图例和布局。
- Figure 2 HGA 热图将保存的 `mask=False` 时间点设为 NaN 留白，同一位点跨 sound Repeat 任务的 mask 取 any；各阶段内按首次显著时间独立排列。没有 mask 的位置保持缺证据标记并留白，不能从 HGA 数值推断显著性。
- 原秩评估面板呈现 cosine silhouette，突出实际拟合的 k=3；cophenetic 最大值 k=2 继续保存在统计结果中，不能改称已拟合 k=2。
- 曲线均值和 SEM 来自已保存的 electrode 波形；恢复原画布、0.5-z 比例尺及固定坐标范围，不改变每个位点的 NMF 输入。原 SEEG 固定解剖示例直接保留原 SVG，并记录其来源。

## 变更必须先审阅

- 用户明确要求：**以后任何相对旧版或当前已批准实现的改动，即使细微，也必须先交用户审阅，再实施。**这适用于分析方法、平均权重、误差带、PCA 维度、成分命名、标签覆盖、缺点处理、随机设置以及绘图细节。
- 更新数据、重画图或修复其他问题，不代表已经批准额外的方法或样式变化。先列出具体差异、影响范围和依据，取得审阅意见后再修改。
- 本轮已经批准的内容：electrode 等权平均；SEM；PCA 单个 PC >5% 的旧规则；当前三阶段及时间窗。时间窗已由用户审核，无需再次变更。
- 已有的成分命名策略、旧标签例外、缺点处理等其他差异，不因本轮修改而自动获得额外批准。后续涉及这些内容时仍需单独提交审阅。

## 版本与当前修正

### Figure 3–6 更新：2026-10-17 用户已批准

- 冻结当前 Figure 1/2、348 个采样位点、231 个 sound-responsive 位点和三阶段 k=3 模型。Figure 3/4 优先，Figure 5/6 同时更新。
- Figure 3 保留 PN sound–image、LD Decision–Repeat、PN Repeat–Passive，沿用原窗口、逐电极统计和原 FDR family；新名单只在原规则对应的位置相交。核对 Passive 的虚拟 Response 对齐及旧绘图排除，保存缺失和配对名单。
- Figure 4 重新准备并拟合功能群解码。原分类器、解码 PCA、CV、置换次数及时间/二维簇校正保持原样；Figure 2 的单-PC >5% 规则不替代解码 PCA。未改变输入和方法的解剖对照可校验后复用。
- Figure 5：RT 模型按任务使用当前解剖合格且有有效 Repeat+sound 试次的位点，不要求预先 HGA 显著。功能群汇总只取其中具有当前 NMF 标签的位点，其他位点单列。快慢反应 HGA 改为 electrode 等权均值±SEM；RT 相关统计继续原被试级汇总/检验。保留 Delay、Go、Response，Delay/Go 联合校正与 Response 单独校正沿用原实现。
- Figure 6：冻结当前三阶段 H，重新 NNLS 投影、计算空间缓存及区域计数，保留原空间参数和 top 50% 显示规则；不复用旧四阶段投影或旧 no-Go 测试模板。
- 本次路径迁移只隔离输入/输出，不改变方法。所有逐图参与者、位点、试次和缺失清单随结果保存。额外方法/画法变化必须先审阅。

此前 323/227 名单及 Figure 1、两版三阶段 NMF 使用了跨模态显著并集或混合模态波形，不能作为符合本口径的正式主分析结果。旧归档保留作为审计依据。
当前唯一参数入口为 `src/main_analysis.py`；正式 sound-only 名单位于 `docs/evidence/sound_only_cohort_2026-10-17/`。修正时同步更新 supplementary 清单与 Figure 1，再从同一名单重建 NMF 和 Figure 2。
