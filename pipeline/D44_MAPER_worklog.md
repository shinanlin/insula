# D44 MAPER 岛叶电极定位 — 完整历程 Work Log

**位置**：与 MAPER 流程同目录（`pipeline/D44_MAPER_worklog.md`）。日常操作见
`pipeline/README.md`；项目级 parcellation 共识见 `docs/PARCELLATION.md`。

**范围**：为被试 D0044（D44）的 sEEG 双极通道，在 Faillenot 6 区岛叶图谱
（ASG/MSG/PSG/pole(AIC)/ALG/PLG；标签奇偶约定：偶数=左，奇数=右）上完成解剖定位，
并比较两条独立方法：

- **轻量法（lightweight）**：把群体 Faillenot 概率图谱（基于 Hammersmith n30r95 数据库）
  单次非线性配准反变换回 D44 native 空间。
- **MAPER 法**：用 Hammersmith n30r95 全部 30 套个体图谱，通过 MAPER
  （multi-atlas propagation with enhanced registration，MIRTK + NiftySeg）
  直接向 D44 native 空间做多图谱传播 + 标签融合。

最终目的：验证两条路线是否解剖自洽、彼此一致，并把跑得通的流程固化为可在其他被试上
复用的 pipeline。

**状态**：流程与两次分析均已完成并修复过两个关键 bug（见下）。修正版已通过 D44
pilot 的几何和逐电极 QC，可作为后续验证的候选流程；它尚未经过 cohort-level 验证，
正式采用为手稿主分析前必须在 3–5 名其他被试上重复验证（见"局限性"）。

---

## 1. 背景与动机

### 1.1 为什么不能只用 FreeSurfer/Destrieux

FreeSurfer 的 `aparc.a2009s+aseg`（Destrieux）分区对岛叶只有粗粒度标签
（如 `S_circular_insula_inf/sup`、`G_Ins_lg_and_S_cent_ins`），没有手稿需要的
前岛/后岛六亚区划分。因此需要引入专门的岛叶亚区图谱。

### 1.2 为什么选择 Faillenot 六区图谱

Faillenot et al. (2017, *NeuroImage* 150:88-98) 在 30 名健康被试的 3D T1 上手工
勾画了六个岛叶亚区（双侧）：三个短回（ASG/MSG/PSG）、前下皮层（AIC，官方语义是
"anterior inferior cortex"，不等同于整体"anterior insular cortex"）、两个长回
（ALG/PLG）。这批标注被整合进了 Hammersmith n30r95 全脑 95 区图谱体系（区域号
84–95 附近对应岛叶亚区），而不是一个独立分发的图谱包——下载到的是同一套 30 人
T1 + 95 区人工标签，其中包含 Faillenot 的六区。

论文本身报告了两条验证过的自动分割路径：

1. 群体概率图谱 + maximum probability map（MPM），在 MNI152 空间发布，可直接下载，
   反变换回 native space 使用——这是"轻量法"的依据。
2. **MAPER**（多图谱传播 + 标签融合）在 leave-one-out 实验中的表现：
   自动分割与人工标注的空间重叠度平均 Dice ≈0.79（Jaccard ≈0.65），平均体积误差 2.6%。

### 1.3 为什么最终选择先做 MAPER，而不是止步于轻量法

讨论过程中出现过三轮反复，最终共识：

1. **轻量法的结构性天花板**：把 30 人平均后的群体概率图谱非线性配准回单个患者，
   本质上是把一张"已经被平均糊掉"的地图套回一个高变异的小结构（岛叶只有几毫米宽，
   沟回变异很大）。配准再准，边界仍然发虚。
2. **一次"配准 bug 还是定义差异"的排查**证明：D44 上 Destrieux vs Faillenot 质心
   相差 7–9mm，其中一部分（去掉 Destrieux 的环岛沟外圈后，质心显著朝 Faillenot
   靠拢：左 7.7→7.2mm，右 9.1→7.8mm）是两套图谱"画法不同"造成的错觉——Destrieux
   只画岛叶表面灰质带，Faillenot 把岛叶实心填充，天然更偏内侧几毫米。这排除了
   "配准跑歪了"的担心，但也说明**轻量法与 Destrieux 的比较不能直接当精度证据**。
3. **患者本身是非典型脑**：sEEG 患者常有癫痫相关结构异常/既往切除/电极伪影。
   Faillenot 论文报告的 Dice≈0.79 是健康被试间 leave-one-out 得到的，配准到不典型
   大脑时误差可能更大。多图谱融合相比单次配准，理论上对个体配准误差有更好的平均/
   鲁棒性——这是选择更繁琐的 MAPER 路径的核心理由。
4. **MAPER 并非过度设计**：它是原论文自己验证、推荐的方法，开源维护
   （`soundray/maper`），专为集群并行设计（30 atlas × N 目标天然可并行），且其
   registration 依赖组织概率图（灰质/白质/CSF）驱动的初始粗配准，不依赖目标 atlas
   自身的标签，避免了"用目标图谱驱动配准再拿它打分"的方法学循环问题。

结论：**先用 MAPER 在 D44 上做先导验证，通过 QC 后再决定是否推广到全部被试**——
这正是本 work log 记录的主线。

---

## 2. 轻量法的实现（对照基线）

脚本：`src/faillenot_pilot.py`（530 行）+ `src/faillenot_qc.py`（346 行），已在 DCC
git 仓库中，非本次固化范围但作为方法对照保留。

流程：
1. ANTs `antsRegistrationSyNQuick.sh` 做 D44 native T1 → SPM canonical `avg152T1`
   的 rigid+affine+SyN 配准（**必须锁定 Faillenot 图谱本身使用的模板**，而非通用
   FSL MNI152NLin6Asym——两者是不同的 MNI 变体，直接混用会在岛叶这种毫米级结构上
   引入系统偏差）。
2. 用配准得到的逆变换，把 12 张概率图（6 区 × 全体积/GM-masked 两个版本，数值
   0–30 转换为 0–1 概率）从 MNI 空间反变换回 D44 native 网格。
3. 对每个双极通道的 contact1/midpoint/contact2 三点分别提取六区概率，
   `P_anterior = ASG+MSG+PSG+AIC`，`P_posterior = ALG+PLG`，双极共识按空间序
   （Contact1→midpoint→Contact2）判定 Anterior / Posterior / Anterior–Posterior(mix)。
4. 输出：`sub-D0044_desc-faillenot_insula.csv`，位于
   `/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/faillenot/sub-D0044/`。

该方法的已知短板（详见第 1.3 节）：群体概率图边界发虚；对 AMT（前内侧颞叶电极，
作为岛叶阴性对照）误判率较高（见第 6 节）。

---

## 3. MAPER 工具链搭建

### 3.1 容器化（Apptainer/Singularity）

- 定义文件：`scripts/maper_container.def`；构建脚本：`scripts/build_maper_container.sbatch`。
- 容器内编译安装 MIRTK（配准）+ NiftySeg（标签融合）+ MAPER（`soundray/maper`：
  `maper` 主程序、`launchlist-gen` 生成任务清单、`hammers_mith-ancillaries.sh` 辅助脚本）。
- 产物：`/hpc/group/coganlab/nanlinshi/maper_tool/maper.sif`。

### 3.2 一个搭建期 bug：Ubuntu 20.04 → 22.04（glibc 版本）

**现象**：容器构建阶段 apptainer 的 fakeroot 支持反复失败。

**根因**：apptainer 注入的 fakeroot shim（`faked`）编译时依赖较新的 glibc
（需要 2.33/2.34），而 `ubuntu:20.04` 基础镜像只带 glibc 2.31，运行不了。

**修复**：把容器基础镜像换成 `ubuntu:22.04`（glibc 2.35），满足 fakeroot 依赖，
MIRTK/NiftySeg 在 22.04 上也能正常编译。改 base 后重新构建通过。

（此为一次性工具链搭建成本，非科学结论问题，记录于此供未来在新集群/新容器环境
复现时参考。）

---

## 4. Atlas 数据准备与 **Bug 1：图谱几何仿射错位**

### 4.1 数据来源

Hammersmith n30r95 图谱数据库（30 套健康被试 T1 + 95 区人工标签，含 Faillenot
岛叶六区）下载并存放在 `/cwork/ns458/atlases/Hammersmith_n30r95/`（`raw/` 原始
压缩包只读，`derivatives/individual_native_pairs/` 下整理出的 T1-标签配对）。

MAPER 运行所需的三类 ancillary 文件（`onepad/`：T1，`posnorm/`：预配准变换，
`seg/seg95/`：分割标签）整理到 `/cwork/ns458/maper_run/ancillaries/`。

### 4.2 Bug 1 的发现

第一次跑通 MAPER 后，QC 检查发现下载的 30 套 Hammersmith 标签体积
（`ancillaries/seg/seg95/aN.nii.gz`）的 NIfTI 头仿射矩阵与其配对的 T1
（`onepad/aN.nii.gz`）存在系统性偏差：**16–30mm 的纯平移偏移**，且两者
`sform_code` 不一致（标签是 2，T1 是 1）——即标签体积虽然体素阵列（shape）与
T1 一致，但头信息里记录的物理空间位置不对，MAPER 配准/传播标签时会把标签"错位"
地贴到 T1 上。

### 4.3 修复

写 `src/prepare_hammers_native_pairs.py`：对全部 30 套图谱，保留标签体积的体素
数组不变，把头仿射（affine/sform）**强制改写为与配对 T1 严格一致**，输出到
`geometry_corrected_labels/`，并生成 `geometry_correction_manifest.json` 记录
每套图谱校正前后的仿射差异（用于审计）。

**验证**：校正后 30 套图谱标签与对应 T1 的仿射匹配误差全部 <0.05mm（`atol=1e-2`
量级），确认修复成功。用修正后的标签替换 `ancillaries/seg/seg95/` 中的原文件
（原始未校正版本已在本轮清理中删除，校正记录保留在 manifest 中，可随时从
`raw/` 重新生成）后重跑 MAPER。

---

## 5. MAPER 运行与 **Bug 2：电极坐标系判读错误**

### 5.1 Slurm 提交

- `launchlist-gen`（MAPER 自带工具）生成 30 条任务命令
  （`launchlist_D0044.sh`），每条对应一套图谱 → D44 目标的独立配准+标签传播。
- `scripts/run_maper_D0044.sbatch`：Slurm array `--array=1-30%15`，
  每 task 4 CPU / 8G / 3h，在容器内执行对应任务行。
- 单次 pairing 实测 wall time ~3:48，峰值 RSS ~1.6GB；全部 30 task 完成后
  MAPER 自动触发融合（fusion），整体 wall time ~10 分钟。
- 融合产物：`/cwork/ns458/maper_run/output/f30-seg95-D0044.nii.gz`
  （硬标签，95 区，与 30 套图谱共享的融合体积），另有
  `-tc3crisp`/`-tcsep-at96` 两个衍生版本。

### 5.2 第一版提取结果的严重误导（已完全撤回）

第一版提取脚本对 175 个双极通道采样融合标签后，结果显示 MAPER 判定与轻量法在
前部电极（RI1-2/RI2-3）上**严重矛盾**：MAPER 判为 pole，轻量法判为 PLG，
两者质心相距约 20mm。这在当时被误判为方法学层面的严重不一致。

**这个结论完全错误**，根源是两个独立叠加的 bug（Bug 1 图谱仿射错位 + 下面的
Bug 2 坐标系错误），已被完全撤回，不构成任何最终结论的依据。

### 5.3 Bug 2 的发现

排查前部电极异常时，检查 BIDS 电极坐标文件
`sub-D0044_task-LexicalDecRepDelay_coordsystem.json` 与
`sub-D0044_space-ACPC_electrodes.tsv`，发现：

- 第一版提取脚本用 `inv(scanner-affine)`（即 NIfTI 头本身携带的仿射矩阵的逆）
  把电极的 x/y/z 坐标转换为体素坐标——这个假设是**错的**。
- **验证依据**：D44 的 `orig.mgz` 的 `vox2ras_tkr`（FreeSurfer tkRAS 变换）
  平移分量为 `[128, -128, 128]`，正是 256³ 体积的**体积中心惯例**——这是
  FreeSurfer tkRAS 坐标系的标志性特征。
- **交叉验证**：用 `inv(vox2ras_tkr)` 对 RI1-2/RI2-3 采样独立的 Destrieux
  分割（`aparc.a2009s+aseg.mgz`），得到 `S_circular_insula_inf`（下环岛沟）——
  与临床植入记录预期的解剖位置一致，证实 BIDS 电极坐标确实是 **FreeSurfer
  tkRAS**，不是 scanner-affine 空间。

### 5.4 修复

写 `extract_maper_insula_D0044_v2.py`：
- 体素坐标改用 `inv(vox2ras_tkr)`（取自 `orig.mgz` 头信息）而不是融合后 NIfTI
  自身携带的仿射——因为融合体积与 `orig.mgz`/`brainmask.mgz` 共享同一体素网格，
  这样转换不依赖分割文件自己的（可能不可靠的）仿射标注。
- 对每个电极先取精确体素（exact-voxel）标签；若落在 MAPER 分割的白质/灰质
  边界（非岛叶标签），退化为 2mm 球内岛叶体素的多数投票，处理双极中点落在
  灰白质交界的情况。
- 6 区分组：`INSULA_IDS=[20,21,86,87,88,89,90,91,92,93,94,95]`（偶数=左，
  奇数=右），LUT 取自 Hammersmith 官方
  `Hammers_mith_atlases_n30r95_label_indices_SPM12_20160111.txt`。

---

## 6. 修复后的验证结果

### 6.1 空间自洽性（六区质心）

修复后，MAPER 六区在 native tkRAS 空间的质心呈现解剖学上完全合理的前后排列
（右侧 y 坐标）：

```
ASG 22.5 > pole 16.2 > MSG 13.5 > PSG 5.8 > PLG −4.3 ≈ ALG −4.0  (mm)
```

三个短回（ASG/MSG/PSG）与 pole 聚在前部，两个长回（ALG/PLG）聚在最后部，
左右两侧镶嵌对称——图谱内部拓扑完好。

RI1-2 电极 (29, −3.5, 7.5) 到修复后 PLG-R 质心 (27.4, −4.3, 8.5) 仅 ~2mm，
与轻量法判定的 PLG 一致——此前 bug 版本"前部电极落在 pole 质心附近"的结论
被彻底推翻。

### 6.2 175 通道全量对比

| 指标 | 结果 |
|---|---|
| Exact-point 两侧方法均可比较 | 11；一致 9/11 (81.8%) |
| 使用 2mm fallback 后两侧方法均可比较 | 12；一致 10/12 (83.3%) |
| 唯二不一致 | LI7-8、RI5-6：均为 PSG（轻量法）vs ALG（MAPER），PSG/ALG 边界附近的 ±1 区分歧，非前后颠倒 |
| 轻量法标为"岛叶"、但 MAPER 未确认的 AMT 电极 | 6/22 (27%) |
| MAPER 标为"岛叶"的 AMT 电极 | 0/22 (0%) |

`artifact `maper_vs_lightweight_D0044_QC_v2_corrected.png` (Claude Science artifact_id=2bd12d84-1604-4245-9581-d45e6a4f52cc; 本地路径为本机会话缓存，请在 Claude Science 工作区内按 artifact ID 检索)` — 三联 QC 图：六区质心空间排列、
逐电极标签对比、AMT 方法间分歧统计。

### 6.3 判定

1. 两种方法在岛叶六区粗粒度标签上高度一致（83%），残余 2 处不一致均为
   PSG/ALG 边界的邻近区分歧，在电极间距（~3.5–5mm）与多图谱融合平滑尺度下
   属预期的正常边界效应，不构成方法学矛盾。
2. **MAPER 在该 pilot 中更保守**：轻量法（群体概率图反变换）把 27% 的 AMT
   电极标为岛叶，而 MAPER 为 0%。在没有人工逐电极金标准的情况下，这 6 个应称为
   "MAPER 未确认"或"疑似轻量法假阳性"，不能仅凭 MAPER 宣布为确定假阳性。
3. 本次判定基于修复后的正确流程重做，此前"MAPER 与轻量法在前部电极上严重
   矛盾"的结论已作废——那个结论建立在错误的 atlas 仿射与错误坐标系转换之上。

---

## 7. 最终结论与局限性

**当前决定**：MAPER 作为个体被试主方法的候选方案继续验证。D44 中 0% vs 27%
的 AMT 差异支持其更保守，约 82–83% 的区域级一致率说明两种方法总体互相印证；
但单被试结果不足以直接确定全 cohort 的优劣，轻量法继续作为快速交叉检查。

**局限性**：
- 本次验证仅针对 D44 单一被试；论文报告的 MAPER Dice≈0.79 是在健康被试间
  leave-one-out 得到的，癫痫患者的结构异常/既往切除/电极伪影可能使个体配准
  误差比论文报告值更大。
- 建议在正式采用 MAPER native 标签作为手稿主分析之前，对 3–5 名其他被试
  重复本流程（两个 bug 已修复，复跑成本低，单被试 MAPER 融合 wall time
  ~10 分钟），确认约 82–83% 一致率与 AMT 方法间分歧在样本间是否稳定。
- 建议将 PSG/ALG 边界电极（如 LI7-8/RI5-6 这类）在论文中标注为"过渡区/
  边界电极"，而非强制二选一。

---

## 8. 可复用产物索引

- 固化后的流程：本目录 `README.md`（`pipeline/`）。
- 环境搭建记录：`insula/docs/environment_setup.md`。
- 最终判定报告（中文，含图）：artifact `D44_MAPER_verdict_report.md`
  (`c285c4c2-73e6-4f6a-8bba-df00c33bb042`)。
- 最终 QC 图：`artifact `maper_vs_lightweight_D0044_QC_v2_corrected.png` (Claude Science artifact_id=2bd12d84-1604-4245-9581-d45e6a4f52cc; 本地路径为本机会话缓存，请在 Claude Science 工作区内按 artifact ID 检索)`。
- 175 通道全量对比表：`sub-D0044_MAPER_vs_lightweight_comparison_FULL175.csv`（artifact）。
- Atlas 几何校正 manifest：
  `/cwork/ns458/atlases/Hammersmith_n30r95/derivatives/individual_native_pairs/geometry_correction_manifest.json`。
