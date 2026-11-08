# 主分析口径：Repeat + sound-only

## Figure5／6按224位点模型更新（2026-11-07，作者要求）

- Figure5、Figure6显式使用 `results/insula_manual14_2026-11-07/nmf/` 的224位点／60人三成分模型；重算数据、输入来源、旧版快照和验收均在 `results/figure56_manual14_2026-11-07/`。新版分别由 `FIG5_RT_ROOT`、`FIG6_PROJECTION_DIR` 读取；旧 `results/rt/` 和 `results/nmf/whole_brain_projection/` 保留，不覆盖旧模型。Figure1–4及所有NMF版本继续保护。本节覆盖下方Figure5／6的旧版本／冻结说明。RT计划为99组受试者×任务：79组匹配后复用，20组重拟，均保留原三阶段校正族。
- Figure5保留原解剖候选建模规则，不额外要求HGA显著；使用原生任务坏道排除、Repeat＋sound、200-ms窗／20-ms步长、10×5折Ridge、1000次置换、seed42。Delay＋Go仍联合做原FWER，Response单独校正。新增解剖候选影响整个受试者×任务的校正族，因此涉及的三阶段一并重拟；其他原模型只在电极顺序、试次、RT、原生来源及参数匹配后复用。功能汇总、显著簇、HGA快慢RT曲线和编码零分布均重新取当前224个标签，不报告14个分析排除或Unassigned；原解剖校正族不因报告筛选而删除这14个。
- Figure5恢复D0086的LexicalDelay／PhonemeSequence及D0139的PictureNaming试次级预处理，沿用原预处理入口，输出写入本次独立目录。原生power epoch之后，保留原 `lib/time_perm_stats.py` 的同一baseline z-score和15阈值outlier-to-NaN步骤，只为RT生成全通道zscore epoch；没有另做或伪造HGA显著性检验。D0139少量缺失触点坐标在独立副本中由同名既有ACPC米单位表补齐，来源与填补记录保留；没有改变原EDF、原生QC或通用数据。D0107依作者要求继续暂缓，不启动预处理。HGA仍按位点内部可用任务等权后画电极均值±SEM；RT的受试者层面推断及5000次编码零分布方法保留。Go图原0.55／0.93秒竖线保留为固定视觉参照，不能称作新版总体平均RT。
- Figure6依作者进一步确认，统一限定Figure1的67位岛叶采样受试者，保留他们可用的全脑接点。用新版三阶段H重新做固定模板NNLS（399特征、mask-any、coverage≥.95、至少1个任务、ridge0），不重新拟合NMF。`in_discovery` 只标识224个建模位点的成员；其余合格全脑位点可保留连续投影，不赋予discovery标签。14个分析排除不等于删除这67人的全脑采样位置。
- Figure6重新计算原空间缓存：8-mm带宽、15-mm最远距离、普通／focused至少5／3名受试者，focused正值比例.5、分位数.5、最多5簇、至少20顶点，图仍显示top50%。这是波形模板投影，不能解释为电极间连接。已完成67人／6689个位点投影，其中202个为当前discovery，22个建模位点没有进入现有投影来源，逐项列于 `fig6/discovery_sites_not_in_projection.csv`；原QC后独立投影图3623个位点／67人，原patch内非岛叶统计1116个位点／67人。
- 已完成：Figure5正文10个代码单元及8张SVG全部执行通过，当前224个位点／60人均有至少一个任务的可用RT模型；原三阶段任一显著簇78个位点／43人，原Delay／Go主窗口命中24个位点／18人。99组原解剖受试者×任务模型中79组验证后复用、20组重拟；功能汇总保留当前224标签，不把原解剖校正族与功能报告范围混为一谈。Figure6正文7个代码单元及3张SVG通过模板／名单／缓存核对（Slurm61372881）。验收分别见 `results/fig5/execution_validation.json`、`results/fig6/execution_validation.json`，版本及来源登记到当前主分析registry。
- 最终视觉核对发现新Sensory均值＋SEM超出旧固定纵轴；仅将三类HGA面板的共用上限由2扩到4.5（实际最大mean＋SEM约3.9324），raster上方OOF-r的共用上限由.6扩到.8（实际最大约.7147），下限、数据、SEM、检验、颜色和布局不变。初次完整新版图和notebook保留在 `fig5/before_display_limit_expansion/`；最终绘图Slurm61386650。原Go固定0.55／0.93-s参考线继续保留，不能解释为新版均值。
- 原数值引擎未改：元数据读取只增加已审核来源和明确的boolean mix类型转换；真实D0103数据的4与12个worker预测／评分／全部1000次置换逐项一致。新增三组的原生标准化来源与参数逐项保存，297个H5的参数、任务、受试者、三阶段和SHA全部核对。统一入口 `scripts/execute_paper_notebook.py` 显式读取新版本，历史全脑刷新脚本也路由当前缓存，避免未来误覆盖旧NMF。最终Slurm61386661（2026-11-08）在显示修正后再次验证931个其他图／旧模型／诊断文件以及669个原生／投影输入未变；主分析registry已更新，最终视觉预览通过。完成记录 `results/figure56_manual14_2026-11-07/completed.json`。

## Figure3采用新名单与224位点模型（2026-11-07，作者要求）

- 正文 `vizpub/fig3.ipynb` 的全部既有面板按新名单重画：两张分类示意、七张功能脑图、两张时间画像，以及六张成分权重／配对／密度叠加图。采样基础为378个位点／67人；功能分析只使用作者已采用的224位点三成分模型及其当前W、标签，保留14个分析排除，不把旧231位点标签搬到新模型。
- 原生任务QC及成对条件数据完整性后，PN为84个位点／24人，LD为136／35。Figure3继续使用其已批准的原生mask且点水平p<.05、目标阶段闭区间[0,1]规则：PN Stimulus声音或图像响应池49／21人，互斥Sound>Image20、未检出差异24、Image>Sound4、双向1单列；LD Delay响应池73／29人，Decision>Repeat26、未检出差异46、Repeat>Decision1。不要混淆这里的条件对比筛选与Figure1／NMF的主响应窗规则。
- 204个已有逐电极条件检验在原生epoch及statistics文件哈希一致后复用；6个PN和10个LD新位点按原方法补算。参数完全保留：独立试次、双侧、2000次置换、.05/.05、seed42；PN使用原生保留试次，LD直接条件检验使用原精确CORRECT规则，波形仍为全部保留原生试次均值。不重新拟合NMF、不调参、不追查raw数据或启动D0107预处理。
- PN时间画像现在为24个位点／16人的两条件完整均值(Sound+Image)/2；LD为26个位点／15人的Decision−Repeat，保留原Decision baseline显示mask。上方均为对应同一名单的电极均值±SEM。沿用原三阶段窗口、按时间长度的物理宽度、1-mm电极行高、0–.5色阶及负值保存／白色显示；按原首次baseline显著时间规则在当前功能类别中重建行序，同值时保留既有行序。无旧总体显著条挪用。
- 权重比例来自当前224位点模型的三列W除以各行之和，重新计算原有描述性密度，不沿用旧电极的权重／颜色／密度缓存。普通与配对图保留σ=.09及同一参考峰值；叠加图沿用当前已确认的σ=.06、两组共同峰值25%显示门槛、小点及原透明度。它们是描述性构成显示，不是新embedding或统计推断。
- 数据、逐受试者来源与复用／补算记录在 `results/figure3_manual14_2026-11-07/`；旧正文图、notebook和结果先保存到该目录的 `before/`。重画入口为 `scripts/slurm/fig3_manual14_render.sbatch`，统一入口 `src.figure3` 显式路由此版本。先前SI保持自己的旧数据／模型版本，未冒称随正文一起更新；Figure1／2／4–6及所有NMF模型保持不变。本节覆盖下方旧Figure3计数及冻结状态。
- 已完成：Slurm61370677执行正文全部18个代码单元，17张SVG全部生成并通过核对。当前分组、功能标签、W比例、时间画像同一电极名单及带符号数值／显示mask均已验证，743个其他图／模型／原来源／旧notebook文件未变。PN画像三类数量3／12／9，LD为3／13／10。验收文件 `results/fig3/execution_validation.json`；视觉预览通过，原17种面板和原绘图样式保留。旧SI采样入口显式固定原解剖名单，避免未来Run All把新采样与旧标签混用；旧11面板验证脚本仍是历史验收，不用于新版。

## Figure1全图更新为67人采样版本（2026-11-07，作者确认）

- 作者将最初“全脑图改为65人”的要求改为采用补全后的67人版本，并更新Figure1全部面板。全脑采样图、区域采样饼图、各阶段及跨阶段响应饼图、区域响应比例统一限定为这67位有已接受岛叶采样的受试者。
- Figure1独立清单为 `docs/evidence/figure1_cohort67_2026-11-07/`，通过 `src.main_analysis.FIG1_COHORT_DIR` 读取；名单与决定为 `config/figure1_cohort67_2026-11-07.json`。岛叶解剖采样378个位点／67人；原生统计确认声音响应239个位点／61人，Stimulus116、Delay133、Response185。重算受试者×任务清单，不沿用旧版89人全脑背景或65人计数。
- 保持Figure1原有“解剖采样→原生声音响应”规则。Figure2的14个位点分析排除不撤销解剖采样和原生阳性统计，因此Figure1的239与Figure2的225个符合分析资格响应／224个非零NMF输入分别标明。D0107的7个已接受位点保留在采样中，预处理仍暂缓；原生任务QC、缺统计、未见阳性响应分别记录。
- 从已有预处理结果及已核对的任务均值重建239个位点的显示波形；采用各位点内部可用任务等权平均，再画电极均值±电极间SEM。沿用Repeat＋sound、原开区间响应窗、显示窗和全部原绘图样式；不新做time-permutation或NMF拟合。
- 全脑图显示有限坐标采样背景；区域饼图继续使用原有区域QC。这两个电极数量分母可不同，但受试者均限于同一67人；各自导出完整位点表以供核对。保留作者确认的D0064示意与D0028原始TFR示例，TFR的Sustained名称标为历史示例标签。
- 更新入口为 `scripts/slurm/fig1_cohort67.sbatch`；旧Figure1、输出与清单保存在 `results/figure1_cohort67_2026-11-07/before/`。Figure2–6与全部历史／现行NMF模型保持不变。本节覆盖下方旧Figure1冻结与计数记录。
- 已完成：Slurm61369856执行全部21个代码单元，导出14张SVG并通过核对。全脑有限坐标采样12088个位点（左7293／右4795），其中岛叶378（左222／右156）；区域QC后的采样4366、跨阶段确认响应2409。区域表与各阶段饼图的数量／百分比已按67人重算，239个位点的HGA在各阶段均完整；原拟合窗95361个波形样本与补全缓存一致。394个Figure2–6／模型／来源文件未变，验收记录为 `results/fig1/execution_validation.json`。SI仍保持其各自旧版，Figure1表格已明确标注版本。

## 作者采用的14个位点排除版本与Figure2更新（2026-11-07）

- 作者明确采用旧手动排除名单与当前NMF输入交集的14个位点排除版本，并要求重生成 `vizpub/fig2.ipynb`。精确名单与决定记录在 `config/insula_manual14_2026-11-07.json`；本节覆盖下方此前“Figure2保持旧版本”的状态。
- 排除名单：D0028_LAI3-4；D0032_LPI2-3、LPI3-4；D0055_RAI7-8；D0061_LAI2-3、LPI1-2；D0063_RAI1-2；D0067_LPII1-2；D0081_LTAS2-3；D0090_RIP3-4；D0095_LIA3-4；D0096_LIA3-4；D0101_RIA3-4、RIA5-6。仅将其作为分析排除；不把已保存的阳性响应改为阴性，也不撤销其解剖采样记录。
- 完整解剖覆盖仍为378个位点／67人，原生统计确认响应仍为239个位点；排除14个后，符合本版分析纳入的响应位点225个，最终非零且完整的NMF输入224个位点／60人×399特征。那个截0后全零的响应位点仍未进入NMF。
- 新采样与分析资格记录在 `docs/evidence/insula_manual14_2026-11-07/`。模型、输入、分组和选秩指标在 `results/insula_manual14_2026-11-07/nmf/`；Figure2通过 `src.main_analysis.FIG2_NMF_DIR` 与 `FIG2_COHORT_DIR` 显式读取，不覆盖供既有下游使用的历史 `results/nmf/`。
- Figure2沿用三成分展示；三成分命名沿用原算法：对归一化H计算各阶段平均份额，以Hungarian分配对应Stimulus→Sensory、Delay→Sustained、Response→Motor，不转移旧电极标签。已有20次初始化中的最低重构误差模型与200次80%子采样、四种PC聚类各500次的完整评估经输入一致性核对后复用。
- **模型选择和指标峰值分别报告**：本版保留k=3用于Figure2；最大cophenetic仍在k=2。seed42下cophenetic的k2／3／4为0.999032／0.998391／0.970170，平均ARI为0.982942／0.976276／0.868011；另一随机种子也复现2／3平台及4下降。此版本在536种探索性删减方案比较后由作者选定，随机种子重复不构成独立数据验证，不能把曲线改写为“所有指标都在3最高”。
- 保留Repeat＋sound、原生任务坏道QC、保存cluster mask与原开区间响应窗、三个拟合窗，以及跨任务等权平均、负值截0、整条拼接L2归一化规则。Figure2组曲线为电极均值及电极间SEM；PCA仍按单个PC解释方差>5%纳入。显示Response保留至1.5秒，拟合仍只至1秒；额外显示数据来自相同任务的现成预处理结果，不补造信号或显著性。
- Figure2热图继续将保存mask=False处留白，并在各阶段按首次显著时间独立排序。旧rank图实际绘制平均ARI，其错误纵轴“Cosine sil.”更正为“Mean ARI”。左右半球比例图沿用解剖覆盖作为分母（378，含手动分析排除和响应证据未定的采样记录），分子使用本版224个模型位点；原Fisher／chi-square检验随输入重算，Sustained星号由新p值决定。
- Response显示的1.5秒边界不补造采样点：现成数据最后一个样本为1.4921875秒，保留原半开右边界及1.5秒坐标轴。旧下游分析的冻结输入校验记录继续保存，不把旧解码结果冒充为新224位点模型的结果。
- 旧Figure2、图和入口参数保存到 `results/insula_manual14_2026-11-07/before/`。Figure1及Figure3–6现有图、统计和历史模型保持此前版本，尚未按新224个位点重做；后续使用必须明确区分版本。
- 已完成：`vizpub/fig2.ipynb` 的20个代码单元全部执行，16张SVG全部重生成；Sensory37、Sustained57、Motor130。执行核对记录为 `results/fig2/execution_validation.json`。269个受保护的历史输入／其他图文件未变。左右采样分母为L222／R156；重算Sustained Fisher p=0.0132022，保留其星号；总体chi-square p=0.0324165。原有Figure2执行／重画入口已连接本版数据。

## 电极补全与NMF成分数复评（2026-11-07，作者批准）

- 当前采样／响应清单入口为 `docs/evidence/insula_completion_2026-11-07/`，由 `src.main_analysis.COHORT_DIR` 指向；当前版本注册在 `config/current_insula_analysis.json`。执行和复核入口为 `vizpub/insula_completion_2026-11-07.ipynb`。
- 将作者明确确认的30个位点补入解剖采样池，采样从348个位点／65人更新为378个位点／67人。沿用 Repeat＋sound、各任务原有QC及主分析原始规则：在 Stimulus (0,.6)、Delay (0,1)、Response (-.5,.5) 开区间内读取已保存的 time-permutation cluster mask，不额外套用Figure3的点水平p值门槛。优先使用已有原生zscore＋statistics H5；原生结果不存在时，保留可核实的原有sound打包波形／mask，并明确记录来源和缺口。没有新做time-permutation检验。
- 确认响应239个位点／61人；按原始负值截0、跨任务等权平均、拼接后L2归一化构建238个位点×399特征的NMF输入。1个响应位点的拼接波形截0后全为0，未进入NMF。相对旧231个输入位点，保留199、新增39、32个未继续进入：16个所有相关任务均被原有native QC排除，15个当前窗口内无阳性mask，1个仍有统计证据缺口。逐位点／逐任务／逐阶段证据均在同目录的CSV及汇总Excel中；缺证据不解释为无响应。
- D0107按作者最新要求暂缓：已检查 `/work/ns458/` 两个对应同名数据集目录，未找到原生Repeat结果。7个已确认解剖位点仍计入采样覆盖，响应状态标记为待补预处理，本轮不进入NMF；未启动该人的预处理，也不追查raw BIDS。
- 新NMF输入、k=2–6各200次80%电极子采样共识复评、每个rank的20次初始化模型，以及原PCA各PC解释方差>5%规则下的四方法聚类指标，保存在 `results/insula_completion_2026-11-07/nmf/`（`NMF_REASSESSMENT_DIR`）。PC聚类k=2–10，随机方法各500次。最大cophenetic规则选择k=2；其0.02近似范围包含2／3／4／5。NMF的ARI和cosine silhouette峰值也在2；GMM的silhouette和Calinski–Harabasz峰值在3，其余三种PC聚类方法两个指标均在2。
- 更新前清单与NMF已保存于 `results/insula_completion_2026-11-07/before/`。原 `results/nmf/` 继续作为既有Figure1–6及下游统计的冻结模型来源（`FROZEN_FIGURE_NMF_DIR`）；本轮没有重画旧图或将旧功能标签自动转移到新rank。新模型使用数字成分标签，全部SVG另存于 `img/insula_completion_2026-11-07/`。下方旧版本计数与冻结规则保留作历史记录。

## Figure 3 正文与 supplementary 统一入口（2026-11-01 用户批准）

- 正文 weight 图整合（2026-11-03 用户确认）：将原 PN21／LD27 双面板与单独 Sound > Image18 图的可执行绘图代码及输出加入 `vizpub/fig3.ipynb`，保留原25个单元及输出。统一调用 `src.figure3.render_component_weights`，SVG导出到 `img/fig3/`，数值及溯源文件保存到 `results/fig3/component_weights/`，并登记主图manifest。同步更新正文Notebook生成脚本，保留现有权重、配色、密度尺度、中心连顶点辅助线及无100%文字的画法。后来的灰蓝叠加图继续留在诊断Notebook；原图、检验与NMF不变。

- 非空间 weight 构成草图（2026-11-02 用户要求 manifold 风格美化）：仍使用 `vizpub/diagnostics/fig3_weight_distributions.ipynb` 的两张三角坐标图，固定 PN21／14人、LD27／16人的三个冻结 loading 比例。保持原始点位；按用户后续要求去掉组平均菱形及对应图例，数值均值仍保留在导出表中，弱化框架，增加低饱和度密度云和等高线。密度采用共同σ=0.09（三角边长1）的等权高斯显示核，各组除以电极数，再共享峰值色阶；只显示有效三元权重范围，等高线为共同峰值的20／40／60／80%。此为描述性密度美化，不代表新embedding、置信区间或显著性；不改变坐标、权重、NMF或正文／SI原图。原始三元坐标和密度网格均保存用于溯源。 用户随后要求加入component颜色：各电极按三个权重比例连续混合 Sensory蓝#2369BD／Sustained金#C4A35A／Motor红#A9373B，保留纯成分颜色图例；密度背景改为中性灰。重着色前后已逐数组核对密度网格，并核对权重、点位和均值表完全一致。 用户随后要求同样展示显著Sound > Image：在同一诊断notebook追加正文响应池的单向18位点／12人独立SVG，双向显著1位点仍单列；沿用原权重、颜色、σ和前两图共同密度峰值，正文／SI不改。用户最新要求同步更新三张诊断图的辅助线：删除竖直虚线及50%倒三角连线，改为等边三角形中心（各成分1/3）分别连接三个顶点的三条实线；保留边缘50%刻度。这三个几何扇区不等同于最大weight功能分类；点位、权重、颜色与密度数组保持不变。 用户批准新增叠加图：同一三元坐标中以灰色表示未检出差异21位点／14人，蓝色表示单向Sound > Image18位点／12人；复用原点位与密度数组，共用σ、峰值及等高线阈值。仅此叠加图中，点色表示条件分组，点大小14、透明度0.60；两组密度云及对应色条透明度均0.40。保留三色顶点名称和中心连顶点的辅助线，不显示均值菱形或顶点100%。同一notebook追加独立SVG，原有图和结果保持不变。

- 正文分类图外观修正（用户要求）：恢复两个圆自然相交的常规 Venn 外观，取消区域外移、人工间隙及各区域单独描边，恢复原有两个整圆的轮廓。作者自行在 Adobe Illustrator 中分开区域；保留不改变外观的填色／数字／标签分组。形状、面积比例、数量与统计不变。正文 notebook 的两张输出同步更新；曲线、脑图、SI 及 NMF 保持原样。更新入口仍为 `scripts/refresh_figure3_venn.py`。

- 修改前已确认旧正文 notebook 与 Git 内容完全一致，并建立检查点 `5924930849762b209838fadf4f991833597086ad`，保存旧正文及当时的 Figure3 代码和 SI。
- 正文唯一入口：`vizpub/fig3.ipynb`。保留条件差异分类示意、功能着色空间图、PN21 位点两条件波形及平均 HGA heatmap、LD27 位点两条件波形及差值 heatmap。仍为独立子图，保持已批准的颜色、窗口、行高、mask 和统计结果。输出统一到 `img/fig3/`、`results/fig3/`，共享入口为 `src/figure3.py`。
- SI 统一入口（用户要求合并）：`vizpub/supplementary/fig3_supplementary.ipynb`。按 Picture Naming、Lexical Delay、Repeat–Passive 组织，前两组各自的采样／响应信息和完整 HGA 放在一起。原采样与完整 HGA 两个 notebook 已合并移除。采样与 baseline 响应部分：先列完整解剖采样 PN162／28 人、LD214／38 人；再列 native QC 通过且配对统计完整128／27、187／35；功能分类后81／24、135／35；目标阶段响应池44／21、71／27。完整采样是覆盖元数据，功能统计排除 Unassigned。Venn 中央表示两条件均相对 baseline 响应：PN Sound35、Image21、交集12；LD Decision61、Repeat58、交集48。与正文中央“未检出条件差异”的分类图严格区分。
- 同一 SI notebook 的完整 HGA 部分：保留 Stimulus／Delay／Response × Sensory／Sustained／Motor 的两条件波形与原始 HGA heatmap，沿用原版函数体、baseline mask 空白规则、P95 色阶及条件配色。Sound–Image81 位点、Decision–Repeat135 位点；Repeat–Passive 保留原缓存与当前功能样本的64 位点。Passive Response 沿用 nominal 对齐，不视为实际发声。
- 所有四任务采样与任务交叠图继续保留在独立 `vizpub/supplementary/task_sampling.ipynb`。不混入正文 Figure3。
- 已淘汰的 weight 混色／高斯展示及重复入口从当前工作流移除。必要的原始统计、波形、冻结 NMF 和排除记录保留；旧代码可从 Git 检查点恢复。正文 Figure1／2／4／5／6 与 NMF 不改。

## 当前统计口径：排除功能 Unassigned（2026-11-01 用户确认，覆盖下方旧规则）

- 所有功能统计只纳入冻结 NMF 标签为 Sensory、Sustained、Motor 的位点。Unassigned 不进入分母、组数、参与者数、曲线／SEM、热图、脑图或成分构成统计；不重新拟合或补分配 NMF。
- 解剖采样总数单独保留为覆盖信息，不称为当前功能统计分母；解剖端点 Unknown 的既有纳入规则不变。全脑 fixed-H projection 有自己的投影成分标签，不能把所有非岛叶位点误当成 Unassigned 删除。
- PN：原解剖配对清单128位点／27人；排除47个 Unassigned 后功能统计81／24，其中 Stimulus 0–1 s 响应池44／21（原49／22）。差异互斥分组18个 Sound>Image、21个未检出差异、4个 Image>Sound，双向显著1个单列。含双向时方向总数为19／5。
- LD：原解剖配对清单187／35；排除52个 Unassigned 后功能统计135／35，其中 Delay 0–1 s 响应池71／27（原78／28）。差异分组27个 Decision>Repeat、43个未检出差异、1个 Repeat>Decision。
- 当前时间画像相应变为 PN21位点／14人、LD27位点／16人；按原方法重算选定位点的 electrode mean±SEM。条件差异热图的原数值、mask、时间窗、行高和色阶沿用，仅删除未分配位点；总体显著条按新样本重算，沿用原5000次 subject-block time_perm_cluster、.05/.05与原阶段校正范围。
- RT 的已完成单电极拟合、置换和原跨通道／阶段校正结果保留；统计报告仅纳入有冻结功能分类的电极。其原校正族仍保留原范围，不借此次删点改变阈值。已按三个成分计算的 HGA–RT encoding 无需重做。原文件、排除名单和更新前图表均保留溯源。
- 统一筛选入口：`src/main_analysis.py::assigned_functional_sites`。新结果存放于 `results/supplementary/fig3_assigned_only_2026-11-01/`；当前 Figure3 分类／时间画像 notebook 为 `vizpub/supplementary/fig3_directional_difference_venn_draft.ipynb`。下方包含 Unassigned 的数字是历史版本，不可继续用于当前统计。

## Figure 3 分组的时间画像（2026-11-01 用户确认）

- 保留三个阶段。PN 25 个 Stimulus 未检出条件差异位点的热图为各电极 (Sound+Image)/2，保留完整均值，不套差异 mask、不逐行归一化；上方同时画这25个位点的 Sound／Image electrode 平均±SEM。
- LD 28 个 Delay Decision>Repeat 位点热图为 Decision−Repeat；沿用已有 Decision baseline 显示 mask，上方同时画这28个位点两条件的 electrode 平均±SEM。色值表示差异大小，不冒充新增逐时间点条件差异显著 mask。
- 两图使用现有显示窗：PN Stimulus[-.5,1]、LD Stimulus[-.5,.6]，两图 Delay[0,1]、Response[-.5,1]；物理宽度正比于时长，原1-mm行高、旧行序子集、功能成分颜色和0–.5色阶不变。负值数值保留，显示为白；不新增平滑或归一化。
- 两图上方曲线使用各自25／28选定样本，不搬用旧49／187样本的总体显著条、不重做统计或NMF。两张独立SVG插入当前分类notebook相应脑图旁，旧统计及全部原图保留不变。

## Figure 3 条件差异三类示意图（2026-10-28 用户确认）

- 交叠圆的中间区域表示“未检出显著条件差异”，不是严格响应集合交集，不表示统计等效，也不是两条件分别相对 baseline 均显著。用户进一步纠正圆大小后，改用各区域面积正比于电极数量，两张图采用相同的每电极面积。PN 的圆总数为 44／29（包含中心25，另列双向1）；LD 圆总数为 77／50（包含中心49）。不再使用等大小的固定圆。极窄的1位点区域用引线标数，其余分组和脑图不变。
- PN Stimulus [0,1] 响应池保持 49 位点／22 人：Sound > Image 单向 19、未检出差异 25、Image > Sound 单向 4；D0080 LPI2–3 在不同时间段双向显著，单列 1 个并单独画脑图，不再重复纳入两个单向集合。
- LD Delay [0,1] 响应池保持 78 位点／28 人：Decision > Repeat 28、未检出差异 49、Repeat > Decision 1。完整采样分母仍为 187／35；本次只更新分类示意及其配套脑图，不变动既有 187／31 位点 HGA 图。
- 每组脑图保留原冻结功能成分配色（Sensory 蓝／Sustained 金／Motor 红／Unassigned 灰），使用相同 Figure 1 局部岛叶、相机、表面投影、4-mm 显示抬升、24-pixel 点和 Figure 2 密度 contour；每成分每半球不足 3 个点时只画点，不降低 contour 门槛。
- 复用原逐电极双侧 time_perm_cluster、2000 次置换、.05/.05 参数及已存方向 mask，不重检、不改试次规则、不重拟合 NMF、不改分析坐标。当前 notebook 为 `vizpub/supplementary/fig3_directional_difference_venn_draft.ipynb`；旧解释归档，新分类及脑图分别导出至 `img/supplementary/fig3_condition_difference_groups/`。

## PN 显示窗扩展：Stimulus −0.5–1 秒（2026-10-28 用户要求）

- 四张当前比较图的 Stimulus 横轴扩展至 [−.5,1] s；Delay [0,1]、Response [−.5,1] 不变，宽度按 1.5∶1∶1.5，保持原 cm/秒及后两列实际宽度。
- 直接显示已完成的 [0,1] 逐电极簇校正 mask，补全 [.6,1]；上方 mean±SEM 仅从已审核的原始位点均值缓存扩展。49 位点池子、20／5／25 分组、原 y 轴／色阶／行序／1-mm 行高和其他画法不变。
- 不重跑或扩展统计：原总体 Stimulus 显著条及其 [−.5,.6] 校正范围不变，新增 [.6,1] 不附未经计算的总体显著条；图注注明这一点。逐电极 mask 本来覆盖完整 [0,1]，不受此上方总体检验范围限制。
- 所有原始统计、mask、p 值、区间和此前数值文件保持字节不变。新显示矩阵／汇总另存 `responsive_pool_difference_masks_2026-10-28/stimulus_display_to1_2026-10-28/`，当前 notebook 和四张 SVG 更新，旧图归档。

## Picture Naming 比较池子：先限定 Stimulus 响应位点（2026-10-28 用户进一步纠正）

- **当前比较池子为 49 个位点／22 人，不是全部 128 个采样位点。** 从已通过解剖和数据完整性审核的 128 个位点中，先保留 Stimulus [0,1] s 对 Sound 或 Image 至少一种有原生 baseline 响应的位点：Sound-only 25、Both 13、Image-only 11。79 个两条件均未检出响应的位点不进入本比较的上方曲线或下方分组。
- 在这 49 个响应位点中，沿用已完成的逐电极条件差异检验：Sound > Image 20 个／13 人，Image > Sound 5 个／4 人，未检出任一方向差异 25 个／15 人。1 个位点在不同时间段双向显著；互斥分解为 19／4／1／25，共 49 个。25 个未检出差异位点由 Sound-only 8、Both 8、Image-only 9 构成，不是 Common 的同义词。
- 三阶段条件差异 mask、参数、校正范围和原始试次处理全部复用；不重新做逐电极检验，不重拟合 NMF。baseline 响应仅用于确定池子；热图仍只显示本电极、本阶段、本方向的簇校正条件差异 mask。
- 上方总体 HGA 同样统一为该 49 位点池子，采用 electrode 平均±SEM。用原 5000 次、subject-block 配对交换、mean_diff、单尾 .05/.05、seed42 的方法，重新计算这 49 个位点对应的显著条；不混用原 128 位点曲线或 p 值。y 轴仍调用原版共享范围函数，仅输入变为 49 个位点。
- 新输出位于 `results/supplementary/sound_image_test/responsive_pool_difference_masks_2026-10-28/`；原 128 位点数据、384 项逐电极检验及上一版 notebook／四张 SVG 均保留归档。当前 notebook 继续使用 `sound_image_cluster_difference.py` 的统一入口，Figure 1/2、NMF、PN Venn/brain 图与 Lexical 结果不变。本节覆盖下方历史规则中的 128 位点显示分母。

## Picture Naming 条件差异热图：逐电极、逐阶段差异 mask（2026-10-28 用户纠正并确认）

- 分母为完整 128 个有效、配对统计完整的岛叶位点／27 位参与者，不先限制到 NMF。逐电极提取原生 Repeat Sound 与 Image 单试次 HGA，保留现有原生 QC 和已批准的 Stimulus 事件来源修复，不额外套用 Lexical 的 CORRECT 筛选。
- 沿用原 `ieeg.calc.stats.time_perm_cluster`：独立试次、默认 t 统计量、双侧、2000 次置换、`p_thresh=.05`、`p_cluster=.05`、seed42。沿用仅两处排序关键词的进程内 NumPy 兼容修复，不修改安装库。
- 用 Stimulus [0,1] s 的检验筛出 Sound > Image 24 位点／13 人、Image > Sound 12 位点／7 人。原始试次均值、试次数、来源哈希及既有 mask 已逐一核对后复用；一个位点在不同时间段有反方向差异。93 个位点在这个 Stimulus 窗口中未检出任一方向，不代表统计等效，也不据此判断其他阶段。
- 用户明确保留三阶段：另对所有 128 位点在 Delay [0,1] s、Response [−0.5,1] s 分别补做相同检验。每个位点、每个完整阶段独立作原时间簇校正，不新增跨位点校正、不缩窄既有 Stimulus 检验范围。
- **热图必须使用本电极、本阶段、本方向的条件差异 mask，不能使用条件相对 baseline 的 mask。** 方向 mask 为原函数返回的双侧簇校正 mask、有效观测、原逐点 p<.05 与对应均值差方向的交集。颜色表示 Sound−Image 或 Image−Sound 的实际 HGA 差值，仅在该差异 mask 为 True 时显示。未经检验的时间点留白，不跨阶段搬用 mask。
- 上方完整 128 位点的均值±SEM、y 轴及已有总体 Sound > Image 显著条不变；其总体检验与下方逐电极检验分别注明。保留原显示窗 Stimulus [−.5,.6]／Delay [0,1]／Response [−.5,1]、时间比例宽度、1-mm 行高、行序、component 配色与 0–.5 色阶。完整 Stimulus [.6,1] 检验结果另存，原显示窗外不外推；负时间未检验，热图留白。
- 当前统一入口为 `vizpub/supplementary/sound_image_test.ipynb`，使用 `src/supplementary/sound_image_cluster_difference.py`。旧 baseline-mask 图和 notebook 整体归档；Run All 不得重新生成旧显示。新统计、mask、完整位点表、显著区间及矩阵另存 `results/supplementary/sound_image_test/condition_difference_masks_2026-10-28/`。Figure 1/2、NMF、PN Venn/brain 图及 Lexical 结果不变。

## 当前 Figure 4 confusion matrix：采用新 0.5 秒 Delay 窗口（2026-10-22 用户已选择）

- 用户最新确认的标签修正：三张矩阵横轴均显示 dorsal／high／labial／low 类别名称，纵轴类别只保留最左矩阵；所有对角线数字固定为黑色。其余数据、配色、色阶和轴位置保持不变。
- 固定取显著簇中心 0.50 s 周围、总长 0.50 s 的 [0.25,0.75) s，128 Hz 下为 64 点。仅重拟合这一项 observed 分类，保留 13 位点／7 人／203 试次、30×5 CV、参考通道填补后排除、原 PCA/SVM 及第一轮五折 held-out 矩阵定义。
- 用户已选择该 0.5 秒结果用于当前 Figure 4，并要求重新导出整组三张矩阵；Sensory–Stimulus 和 Motor–Response 保持已确认时间窗。图注明确这些描述性窗口模型与原阶段准确率柱图模型不同，不把原柱图 p 值附到新矩阵上。
- 新模型及与原约 150-ms 窗口的同色阶对比另存 `results/fig4/delay_halfsecond_confusion_2026-10-22/`，Figure 4 notebook 中间矩阵读取新模型。保留旧矩阵、其他两个 component 的矩阵、原色阶／布局、所有时间曲线和校正结果。本次窗口在观察结果后选定，标为探索性，不附加新的置换显著性，不延伸搜索其他窗口。

## Confusion matrix 色彩对比（2026-10-22 用户要求）

- 增强三张矩阵的颜色对比：保留蓝／金／红基础色与 0.25–0.60 数值范围，统一采用 PowerNorm gamma=0.5，让中间数值颜色更深；色条同步使用相同映射。图注明确颜色采用该幂映射，单元格数字仍为原始行归一化分类比例。只改显示；矩阵、预测、拟合、窗口、类别顺序、布局及其他图不变。

## 按显著簇时间窗重建 confusion matrix（2026-10-22 用户要求）

- 当前 .05 成簇／.10 簇校正结果中，Sensory–Stimulus 取起点跨零显著簇（中心约 −.082–.605 s），Sustained–Delay 取中间显著簇（.426–.574 s），Motor–Response 取跨零整簇（−.324–.152 s，用户已明确确认）。按首末中心区间作为连续特征窗，沿用原半开整数采样裁剪，实际采样边界单独导出。
- 保留原任务／位点／试次、参考通道填补后排除顺序、原 scaler/PCA/SVM 和 30×5 CV／求解器上限。只重拟合 observed 数据，矩阵保留原第一轮五折 held-out 预测及按真实类别行归一化；配色、类别顺序、色标、布局不变。新增结果另存，旧矩阵归档，不附新置换显著性；同数据选择时间窗的描述性矩阵标为探索性。时间曲线、mask、阶段汇总统计、其他图和 NMF 保持现状。

## 当前 articulator 定稿显示：只显示簇校正条（2026-10-22 用户确认）

- 采用已确认的 `p_thresh=0.05`、`p_cluster=0.10` 全九条曲线版本。删除全部未通过簇校正的候选细线，仅显示既有校正后显著条；图注明确写出两个参数。此次仅改显示，所有保存的 mask、p 值、曲线、SE、横轴和其他分析不变。上一显示版本归档，探索性阈值选择的记录保留。

## 最新 articulator 阈值对照：p_thresh=.05、p_cluster=.10（2026-10-22 用户明确要求）

- 九条 articulator 曲线统一保持逐窗成簇 `p_thresh=.05`，最终簇水平阈值改为 `p_cluster=.10`。复用同一簇长度引擎／背景处理、500 条标签置换、全部 57 中心、样本与拟合；不缩短校正范围、不改变任何曲线或 SE。细线为未校正 p<.05 候选区间，粗线本版明确表示校正后 p<.10；说明放图注，原画法及横轴比例保留。
- 新推断独立保存至 `results/fig4/time_cluster_p05_c10_2026-10-22/` 并更新 Figure 4 notebook 的当前引用；归档上一版 .10/.05 图，旧配置／推断／诊断不覆盖。本项是探索性阈值对照，覆盖前一版的当前 articulator 阈值。Lexicality、阶段汇总检验、Figure 1/2 与 NMF 保留现状。

## 全部 articulator 曲线采用同一成簇门槛（2026-10-22 用户明确扩展）

- 将刚审阅的 .10 成簇／.05 最终簇校正方法统一用于三个 component × 三个阶段共九条 articulator 时间曲线，并重画完整原版三阶段图。每条完整 57 中心曲线分别校正，真实与 500 条相同 shuffle 同步按 .10 形成候选簇；复用同一已审阅簇长度函数及背景 0 排除，另存安装版 mask／簇 p 值。原拟合、样本、窗口、准确率、SE、平滑、颜色与横轴比例不变。
- 细色线表示候选区间，原粗色条表示通过最终 .05 校正的区间，统计含义写进 notebook 图注，不在轴内加字。新推断另存于 `results/fig4/time_cluster_010_2026-10-22/`，Figure 4 notebook 统一读取该版本；原 .05 缓存、上一版图和所有独立诊断保留。此项扩展覆盖此前仅限 Sustained–Delay 的范围限制及特殊 .05 未校正叠加；阈值的事后选择仍标为探索性。其他 Figure 4 分析、Figure 1/2 及 NMF 保留现状。

## Sustained–Delay 成簇门槛比较（2026-10-22 用户明确要求）

- 将形成候选簇的逐窗门槛由 p<.05 改为 p<.10，最终簇水平校正标准保持 .05；复用原 balanced accuracy、相同 500 条整音节 shuffle 曲线、13 位点／7 人／203 行及完整 57 中心。真实和 shuffle 同步重新成簇，重算每次 shuffle 的最大正簇长度。仅此独立诊断分开原入口共用的两个阈值，保留已批准的背景 0 修复并另存安装版结果。
- 不重新拟合、不改变指标／模型／窗口／样本，不缩短校正域，不加入 mass 或额外门槛搜索。由于是在观察旧结果后新增的门槛比较，记录为探索性分析，另存结果、图及独立 notebook，当前主 Figure 4、原正式统计和全部旧诊断保持原样。

## Figure 4：显示校正前候选簇（2026-10-22 用户明确要求）

- 随后明确的显示修改：删除图内 “uncorrected p < 0.05”，仅在 notebook 图注说明。每个既有校正前区间改为连续金色线段，三个独立区间之间保留空隙；边界、p 值、mask、曲线及其他面板不变。这覆盖下方先前的虚线及图内标字样式要求。图注须明确 Sustained–Delay 金色段为未校正结果，其他原有条仍为时间簇校正结果。

- 保留当前 articulator time-resolved 图，在 Sustained–Delay 中增加原 time-perm 保存的逐窗 p<.05 连续区间。使用金色虚线，并直接标为 “uncorrected p < 0.05”；原时间簇校正通过区间继续使用实线。保留所有原始曲线、SE、p 值和校正后 mask，不平滑筛选标记、不连接不连续区间、不缩窄校正范围或重新检验。
- 仅更新 Figure 4 对应 notebook 绘图及说明并导出，归档原图。其他 component／阶段、样本、模型、正式统计结果和独立 macro-AUC 等诊断保持原样。这里展示的是校正前候选簇，不代表通过完整时间范围校正。

## Sustained–Delay 独立方法比较（2026-10-22 用户已批准）

- 最新用户明确恢复完整 macro-AUC：续算全部原版 57 个中心到相同 500 次置换，再按原完整校正域做 time perm。保留 300 ms／30 ms、13 位点／7 位参与者／203 行、30×5 CV、原模型／AUC 评分／置换规则和阈值；独立诊断继续采用此前批准的背景 0 修复，另存安装版原 mask。不增加 mass、FDR 或调参。此项授权覆盖此前“只看 6 点”的执行限制；已完成六点局部对照及所有旧诊断保持原样，完整结果追加到独立 notebook，正式结果不动。

- 随后用户明确收窄 macro-AUC 范围：停止完整 57 窗口及最终汇总，只看原 accuracy 中间 bump 的 6 个中心（约 .426–.574 s），保留全部已有缓存，仅为这 6 窗口补齐原 500 次匹配置换。名单来自冻结旧 accuracy，不能看新 AUC 后再选择。局部对照另存 `central_bump_only/`，明确为探索性、未完成原全范围 time-perm 校正；不在缩窄后的数据选择区间重新做 time perm，不把逐点 p 当作校正后显著性。正式结果保持不变，此项收窄覆盖下条完整执行范围。

- 最新明确批准 macro-AUC 单项测试：Sustained–Delay 原版 300 ms／30 ms、57 中心，冻结 13 位点／7 位参与者／203 行、500 次相同整音节置换和 30×5 CV，原填补、训练内 scaler/PCA 及 SVM 不变。每个 held-out fold 内，分别用四类 SVM 连续决策分数计算 one-versus-rest ROC AUC，四类等权，再按原版对折／轮等权平均；不跨折拼接校准分数，不加 softmax。原完整时间校正域及 time-perm 簇长度检验不变；独立诊断沿用已批准的背景 0 修复，同时另存安装版原结果。不做 mass、FDR 或调参。逐个真实／置换拟合核对 accuracy、标签、CV、填补与原缓存一致。另存 `sustain_delay_macro_auc_2026-10-22/` 并将真实／shuffle 与原 accuracy 并排呈现；正式结果及此前诊断不改。

- 最新明确批准 350 ms／30 ms + cluster mass：直接复用已完成的 56 中心／500 次置换曲线及此前相同的 mass 实现。13 位点／7 位参与者／203 行、逐点阈值及原完整校正域保持不变，不重新拟合、不额外调参。另存于 `sustain_delay_350ms_2026-10-22/cluster_mass/` 并追加到现有独立 notebook；先前结果和正式结果均不替换。此项批准覆盖此前未授权 350-ms mass 的限制，仅限本项诊断。

- 随后明确批准 350 ms／30 ms 单项测试：同样的 13 位点／7 位参与者／203 行、500 次整音节置换和 30×5 CV，PCA .8、C=1、原填补和求解器不变；沿用簇长度检验及原完整校正域。原整数规则下是 44 点（343.75 ms）。独立输出到 `sustain_delay_350ms_2026-10-22/` 并追加到现有诊断 notebook，不推广到其他窗口、群体、阶段或正式结果。

- 随后明确批准组合测试：对已完成的 200 ms／20 ms、84 中心／500 次置换曲线，使用此前相同的 cluster mass 检验。样本、原始准确率、逐点选择、置换计划及原完整校正域保持不变，无需重新拟合。保留原三项结果，另存 `combined_200ms_mass/`，在同一独立 notebook 追加四项比较；正式结果不自动更新。

- 仅在独立诊断中比较三项：原参数 300 ms／30 ms 的簇长度检验、相同已存曲线的 cluster mass、200 ms／20 ms 重跑后的簇长度检验。13 位点／7 位参与者／203 条原始试次行保持不变；PCA .8、C=1、30×5 CV、原始填补和收敛规则不变，不做额外调参。
- 三项使用同一确定性计划的前 500 次整音节标签置换；原 300 ms 曲线已有 5000 次结果，另作簇长度／cluster mass 稳定性核对。200 ms 的真实和置换模型均重新拟合完整流程，并核验标签、试次和 CV 划分与参考相同。
- 校正域固定为原来完整 57 窗口中心的首尾范围，不能只取显示的 0–1 s。新网格沿用原整数采样窗口规则，保留该固定域内的中心，记录实际边界；128 Hz 下 200 ms 为 25 个采样点。
- Cluster mass 是正簇内标准化准确率之和乘以步长；逐时点标准化使用 observed 与所有 null 曲线的对称合并均值／SD，保留原逐点阈值及置换排名，再与每次置换在整个校正域内的最大 mass 比较。CV 重复不作为独立参与者。保留原尾比例和 plus-one p 值，已有背景 0 排除修复仅用于独立诊断。
- 输出独立保存在 `results/diagnostics/sustain_delay_mass_window_2026-10-22/`、对应 `img/diagnostics/` 和 `vizpub/diagnostics/sustain_delay_mass_window_comparison.ipynb`。这是下文禁止更换正式检验规则的明确诊断例外；正式结果、其他群体／阶段和 NMF 均不自动更新。

## Figure 4 显示更新及独立支线（2026-10-21 用户已批准）

- **随后明确的 lexicality 例外：Delay 显示 0–1.5 s。** Lexicality 的三个阶段范围为 Stimulus −0.5–0.6 s、Delay 0–1.5 s、Response −0.5–1.5 s，面板宽度按 1.1∶1.5∶2；现有解剖对照和二维矩阵的训练时间横轴同步。Articulator 的 Delay 仍为 0–1 s，宽度比仍为 1.1∶1∶2。仅更新显示，原始拟合／检验数据和 TGM 测试时间纵轴不变。Figure 3 当前只讨论，不修改或运行。

- 全部一维 time-resolved 曲线（含当前解剖对照曲线）显示范围为 Stimulus −0.5–0.6 s、Delay 0–1 s、Response −0.5–1.5 s。用户随后明确要求面板宽度按时间跨度 1.1∶1∶2 设置。二维跨时间矩阵的训练时间横轴随同阶段，测试时间纵轴沿用原范围。只裁剪显示，拟合及检验的完整时间曲线不裁剪。
- Articulator 均值曲线复用 lexical 的 `gaussian_filter1d(sigma=2, mode='nearest')`，sigma 的单位为窗口索引。现有轮间 SE 保留；误差带围绕平滑后的显示均值绘制。原始数值、逐窗 p 和 time-perm mask 不平滑，lexical 已有平滑方法保持原样。
- 另开 Sustain–Delay 独立诊断：仅在原 time-cluster 函数的本地副本中排除 label 0 的背景计数，原时间簇长度统计、严格比较、p_thresh=.05、tails=1 保留。置换次数预先固定为 5000；验证并复用当前 13 电极／7 位参与者／203 行、57 窗口的真实拟合和前 500 次置换，续算同一确定性计划。模型、30×5 CV、C/PCA、求解器、样本和窗口不变，不进行模型调参，不自动替换正式结果。其余群体／阶段和 NMF 保持原样。

## 时间解码检验：2026-10-21 用户纠正，优先于下方旧记录

- **时间分辨 decoding 必须沿用旧版 time perm 调用链。** 原入口是 `src/decoding/run_decoding_resolved.py:cluster_correction`，内部调用 `ieeg.calc.stats.proportion`、`tail_compare` 和 `time_cluster`。使用原单尾 `tails=1`、`p_thresh=0.05`，每条完整时间曲线作原时间簇校正。
- 不得替换为逐窗口 BH-FDR、Bonferroni、TFCE 或另写 cluster-mass 检验，也不额外增加跨曲线校正。此前把全部513个时间窗统一BH-FDR的实现是偏离原版的实现，现已废弃，不能再作为正式结果或现行规则。
- 复用已完成的真实与匹配置换曲线，保留当前已确认的30轮五折总均值、500次置换、样本、电极、时间窗、SE与模型参数。将匹配的真实总均值与每次置换总均值传给原函数；不把CV重复当作独立参与者。保存原函数输出的时间校正 `mask` 与原始逐窗 `p_values`，图直接使用该mask。原函数本身不改写。
- 这次纠正针对时间曲线；九项阶段汇总柱状图原已确认的BH-FDR规则保留。原逐窗FDR图、表和曲线文件归档为已废弃版本，原始拟合缓存与冻结基础方案保留，另记本次推断修正来源。

## Figure 5 解剖保留决定（2026-10-20 用户已批准）

- 用户选择“确认可以保留：名单不变，继续 Figure 5”，保留本轮 MRI 核对的全部 29 个位点、50 条 task×位点记录。冻结采样名单、Figure 1/2 和 NMF 保持原样；原坏道 QC 继续执行。
- 保留决定绑定到 `config/figure5_anatomy_retention_2026-10-20.json` 的具体位点、任务、原队列哈希及本次分区证据。新旧标签与组织冲突记录继续保存，不将作者保留决定改写为已独立确认的组织分类。新的位点或变化后的证据仍须审阅。
- Figure 5 沿用原 Ridge、10×5 嵌套 CV、1000 次置换、种子、窗口及校正族；HGA 为 electrode mean±SEM，Go 保留。只解除已批准冲突对运行的阻挡，定量模型源文件及原生坏道规则不变。记录目录为 `results/audit/figure5_resume_retained_2026-10-20/`。
- D0067 已单独按原 PIAL 坐标修复隔离端点表；53 对双极中点核对通过，Hammers CSV 已重新生成，唯一采样位点 LPII1–2 保持岛叶标签，无新增冲突。全队列 65 位参与者的 118 份 task×subject CSV 已补齐。

## Figure 4 正式统一方案（2026-10-20 用户已批准）

- **随后批准的单项收敛调整：仅阶段汇总 Motor–Response 将 max_iter 从10000提高至30000。** 原10000上限在第9轮、第1折触顶；其余阶段模型和全部时间分辨窗口（包括 Motor–Response 时序）仍用10000。停止容差、C、PCA、样本、试次、CV、置换、校正和画法均不变。例外写入中央配置的 `solver_overrides`，并在冻结基础方案旁保存带哈希的审批记录；不改写其他模型的结果或身份。

- 阶段汇总窗口：Stimulus [0,0.6)、Delay [0,1)、Response [0,1) 秒；保持各模型的电极、原始试次行及标签。Sustain–Delay 主结果为13电极、203行；完整14电极只作事后选择的敏感性分析，不更改 NMF。
- 30轮分层五折，种子42–71。真实与500次整体标签置换均完整运行同一算法；每次用该次标签重新分层，训练折内重新拟合 StandardScaler 和 PCA(.80)。C=1，tol=1e-4，先以 max_iter=10000 检查收敛。保留原 sample_fold 数值填补规则，先在全参考通道上划分与填补，再剔除 D0084；13/14对照的划分与填补必须相同。
- 置换单位仍为相同保留重复数层内的整音节块；同一全局标签向量用于训练、测试、30轮和整条时序。p=(1+置换总均值≥真实总均值的次数)/501，不丢弃失败置换，不在150个fold之间另作t检验。
- 九项阶段检验共同 BH-FDR（q=0.05）。语音类别的九条时间分辨曲线使用[-0.5,1.5]秒内300 ms完整窗、30 ms步长。**时间曲线的检验以文首2026-10-21 time perm纠正规则为准；此前全部窗口共同BH-FDR的规则已废弃。**
- 时序以真实采样窗口中心为横轴。128 Hz下300 ms沿用整数38采样点，开始位置逐窗映射到最近采样点，记录实际边界，避免累积步长取整误差。各时间窗固定通道/试次名单。窗口宽度与既有HGA预处理共同限制时间定位精度。
- bar/时序均报告30轮均值±轮间SE，SE仅反映固定数据上的划分波动。14电极敏感性及13电极事后排除的局限如实报告；BH-FDR不修复排除方案的事后选择。
- 所有真实和置换拟合均需 n_iter 小于各自已批准的上限、且系数有限；默认上限10000，只有阶段汇总 Motor–Response 为30000。先用真实拟合和前20个计划内置换检查收敛，已完成的置换继续复用。任何拟合失败则保留诊断记录、暂停该单元，不能未经审阅提高上限或删掉失败draw后计算p。新旧结果分目录保存，完整检验族通过校验后更新主图。
- 新标准来源为 `config/figure4_phonetic_decoding.json`、共享引擎 `src/decoding/formal_cv.py` 和入口 `src/decoding/figure4_formal.py`。旧冻结CV诊断保留为历史记录，不作为新版p值输入。Figure 1、2和NMF及词汇性/跨任务分析保持冻结。

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

### 历史版本：Figure 4 九项统一（2026-10-19；已被文首2026-10-20方案替代）

- 三个功能群 × 三阶段共九项，全部使用 **30轮五折 balanced accuracy 的平均值**。先计算每轮五折均值，再对30轮平均；误差线为这30个轮次均值的样本标准差除以√30。**SE只反映固定数据上的CV划分波动，不代表参与者／总体不确定性。**
- 每项使用200次完整音节块置换，检验与柱高相同的30轮总均值。只在保留重复次数相同的音节间交换完整标签块；每次全局标签向量同时用于所有固定轮次／fold的训练和测试。p=(1+置换总均值≥真实总均值的次数)/201，单侧、未校正；不能沿用旧首轮p，也不能把150个fold当作独立样本做组间检验。
- 仅Sustain–Delay排除 `D0084_RAI4-5`，保留13个双极位点、7位参与者及固定203条伪总体试次。在原14电极参考输入上冻结试次筛选、CV索引及数值填充之后，拟合前删除该电极列；其他八项电极名单、试次不变。每项真实30×5准确率必须精确复现归档参考结果。
- StandardScaler与PCA仅在各训练折拟合；PCA保留80%解释方差，LinearSVC C=1、max_iter=5000，随机种子42起。所有原时间窗和缺测填充规则保持不变。标签无关的训练折变换经核验后可缓存以加速同一检验。
- 从标准解码函数、统一配置 `config/figure4_phonetic_decoding.json` 和H5结果格式落实规则。Figure 4 notebook直接读取九个正式模型结果；删除单项绘图替换逻辑。各结果保存真实模型电极、原参考输入电极、排除名单、试次、全部CV索引、填充值摘要及200×30×5置换得分。原解剖对照、词汇解码和跨任务解码保持原方法。
- 画法、配色、布局和原首轮OOF混淆矩阵约定不变。Sustain–Delay的混淆矩阵来自同一13电极模型的首轮预测。对应图注须写明轮间SE的含义。
- D0084的排除是在查看逐一剔除结果后选择的，保留探索性说明；该p未校正位点选择，不是删除前后差值检验，也不能声称未见音节／新参与者泛化。全球NMF、Figure 1/2保持冻结，不把该位点全局改标Motor。
- 本次实施及验证：`results/audit/figure4_uniform_repeated_cv_2026-10-19/`。旧代码、单项替换notebook和正式模型均归档于其 `before/`；此前 `results/fig4/approved_sustain_delay_2026-10-19/` 是已被本统一方案取代的历史记录，不再作为绘图输入。
- 完成后的九项数值：`results/fig4/phonetic_category_summary.csv`；模型实际名单和逐轮分数在同目录。Sustain的Stimulus／Delay／Response未校正p依次为0.014925／0.019900／0.039801，因此本版不支持“仅Delay显著”。Motor–Delay为p=0.099502，其余八项均达到未校正p<0.05；不据此添加新的调参或位点筛选。
- 原求解器限制保留：Motor–Response的148/150个真实拟合和18428/30000个置换拟合触及5000次迭代上限；其他八项没有触及上限。所有九项的真实150个得分均精确复现各自参考（Sustain–Delay参考为13电极诊断）。运行完成与得分复现不等于求解器已充分收敛；该限制保存在本轮`validation.json`中，未经用户审阅不调整求解器参数。

### Figure 3–6 更新：2026-10-17 用户已批准

- 冻结当前 Figure 1/2、348 个采样位点、231 个 sound-responsive 位点和三阶段 k=3 模型。Figure 3/4 优先，Figure 5/6 同时更新。
- Figure 3 保留 PN sound–image、LD Decision–Repeat、PN Repeat–Passive，沿用原窗口、逐电极统计和原 FDR family；新名单只在原规则对应的位置相交。核对 Passive 的虚拟 Response 对齐及旧绘图排除，保存缺失和配对名单。
- Figure 4 重新准备并拟合功能群解码。原分类器、解码 PCA、CV、置换次数及时间/二维簇校正保持原样；Figure 2 的单-PC >5% 规则不替代解码 PCA。未改变输入和方法的解剖对照可校验后复用。
- Figure 5：RT 模型按任务使用当前解剖合格且有有效 Repeat+sound 试次的位点，不要求预先 HGA 显著。功能群汇总只取其中具有当前 NMF 标签的位点，其他位点单列。快慢反应 HGA 改为 electrode 等权均值±SEM；RT 相关统计继续原被试级汇总/检验。保留 Delay、Go、Response，Delay/Go 联合校正与 Response 单独校正沿用原实现。
- Figure 6：冻结当前三阶段 H，重新 NNLS 投影、计算空间缓存及区域计数，保留原空间参数和 top 50% 显示规则；不复用旧四阶段投影或旧 no-Go 测试模板。
- 本次路径迁移只隔离输入/输出，不改变方法。所有逐图参与者、位点、试次和缺失清单随结果保存。额外方法/画法变化必须先审阅。

此前 323/227 名单及 Figure 1、两版三阶段 NMF 使用了跨模态显著并集或混合模态波形，不能作为符合本口径的正式主分析结果。旧归档保留作为审计依据。
当前唯一参数入口为 `src/main_analysis.py`；正式 sound-only 名单位于 `docs/evidence/sound_only_cohort_2026-10-17/`。修正时同步更新 supplementary 清单与 Figure 1，再从同一名单重建 NMF 和 Figure 2。
