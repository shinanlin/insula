# Insula 零标签端点与原生解剖文件核对

日期：2026-10-17。范围是旧 Figure 1 主表中，中心有岛叶标签、至少一个端点的 Hammers 标签为 0 的 **132 对双极位点**。本次没有重新拟合 NMF、运行 MAPER、修改纳入名单或覆盖既有结果。

**同日后续更新：** [原生影像恢复与复核记录](recon_restore_2026-10-17.md)。下文保留恢复前的检查状态；后续已从 Windows/Box 补回全部 54 位所需文件，并核对全部 131 个保存坐标。恢复后发现 D0057 的 5 个标签差异、D0086 的 6 个触点坐标版本差异；D0125 的 opercular 标签在恢复的原生分区中得到复现。最新数量和复核程度以该后续记录为准。

## 当前结论

**不能把这批 0 全部解释为白质，也不能保证它们均不涉及邻区灰质。** 原生 MRI 与 MAPER 产物缺失限制了最终判定，但仍存的、与原始端点坐标一致的 FreeSurfer 标签已经提供了一个明确的 opercular 灰质疑点。

- 132 对涉及 54 位参与者，包含 144 次零标签端点记录；去重后为 **131 个触点、131 个原生坐标位置**。
- 144 次端点记录全部找到原生坐标一致的历史 FreeSurfer/Destrieux 标签，匹配误差不超过 **0.001 mm**。这些标签保留在 `xtractaf.csv` 中；这里只读取其继承的原生 parcellation 标签，没有用 AF 概率推断组织类型。
- 找到 11 位参与者的共享重建备份，可对其中 **20 个独立零端点**重新取样原生 FreeSurfer 体素，并核对重建触点坐标。20 个均与历史标签相符。
- 本次涉及的 54 位参与者，**cwork 中均缺少原生 MRI 和 MAPER seg95／tc3crisp**；因此尚未重新验证原 MAPER 的组织判定。

## 1 131 个独立零端点的现有标签

| FreeSurfer/Destrieux 标签类别 | 独立触点数 | 其中已重新核对原生体素及触点坐标 |
|---|---:|---:|
| 明确的脑白质标签 | **72** | 7 |
| 岛叶灰质或岛叶周围沟标签 | **27** | 6 |
| Unknown，组织类型未确定 | **30** | 6 |
| inferior frontal opercular 灰质标签 | **1** | 0 |
| 丘脑标签 | **1** | 1 |
| 合计 | **131** | **20** |

这些数量描述现有 FreeSurfer 标签及其复核程度。27 个岛叶标签提示 MAPER 与 FreeSurfer 的组织边界或分区判定存在差异，不能把它们继续计作已确认白质；30 个 Unknown 也不能直接判为白质或脑脊液。未获得 MRI 复核的标签须保留为历史证据。

按双极对计数，132 对中有 **72 对的所有零端点都具有白质标签证据**；这不等于 72 对全部已经完成 MAPER 组织复核。两个端点都为 0 的配对以及相邻配对共享触点，使独立触点数与配对数不同。

## 2 需要优先复核的两个位点

### D0125_LIA5-6 的 LIA6

- 旧 Hammers 端点标签：**0**。
- 坐标一致的历史 FreeSurfer 标签：**`ctx_lh_G_front_inf-Opercular`**，即左侧额下回盖部灰质。
- 原生 tkRAS 坐标：**x = −31.7141、y = 26.0286、z = 7.9286 mm**。
- 来源为 PictureNaming 的逐端点 parcellation 记录，与原 HGA 坐标误差约 `1e-13 mm`。
- 该双极对端点间距约 **3.23 mm**，并且存在于旧 255 个 NMF 位点中。
- D125 的原生 MRI 和 MAPER 组织分割当前缺失。**现阶段是有明确 opercular 标签证据的疑点，还不能称为经过个体 MRI 最终确认的 operculum 触点。**

这一个位点已足以说明“端点为 0 可以自动视为安全白质”的规则不成立。恢复 D125 后，应优先查看原始 MRI、CT 定位和组织分割，核对该端点与岛叶—盖部边界的位置。

### D0086_RTPS4-9 的 RTPS4

- 旧 Hammers 标签：**0**。
- 历史 FreeSurfer 标签和本次共享原生体素复核均为 **`Left-Thalamus`**。
- 原生 tkRAS 坐标：**−20.2857、−19.8571、−2.9286 mm**；与共享重建中的触点坐标一致。
- 该双极对端点间距约 **19.77 mm**，名称为非连续的 4–9 配对，应同步核对配对构造。
- D0086 还存在已知的跨任务解剖和左右侧记录冲突。因此这里能够确认的是“所保存坐标对应的 FreeSurfer 体素标为丘脑”；触点名、左右侧与实际电极定位仍需统一核查。

本次没有据此删除任何位点或更改配对。

## 3 Cwork 中哪些解剖文件仍在

### 本次涉及的 54 位参与者

对这些参与者对应的原生重建目录逐一检查：

| 文件 | cwork 可用参与者数 |
|---|---:|
| `mri/orig.mgz` | **0 / 54** |
| `mri/brainmask.mgz` | **0 / 54** |
| `mri/aparc.a2009s+aseg.mgz` | **0 / 54** |
| `aseg.mgz`、`wm.mgz`、`ribbon.mgz` | **均为 0 / 54** |
| 重建的原始触点 RAS 坐标文件 | **0 / 54** |
| MAPER `f30-seg95-*.nii.gz` | **0 / 54** |
| MAPER `*-tc3crisp.nii.gz` | **0 / 54** |
| 所检查四个任务 BIDS 的个体 T1w | **均为 0 / 54** |

目录名称仍在，并不表示影像文件仍在。历史 parcellation 标签表与打包 HGA 数据仍有保留。

### 仍保留的较新个体重建

`/cwork/ns458/ECoG_Recon/` 下，以下 **10 份个体重建**仍有 orig、brainmask、Destrieux、aseg、wm、ribbon：

`D147`、`D148`、`D149`、`D150`、`D152A`、`D152B`、`D153`、`D154`、`D157`、`D158`。

标准模板 `cvs_avg35_inMNI152` 也仍在，但不能代替个体影像。

MAPER 的以下 **8 份** seg95 与组织分割产物仍在：

`D0148`、`D0149`、`D0150`、`D0152A`、`D0153`、`D0154`、`D0157`、`D0158`。

这些较新重建不属于当前 132 对涉及的 54 位参与者，不能用于替代本次缺失的原生图像。

## 4 可以复用的 11 位共享备份

在 `/hpc/group/coganlab/ECoG_Recon/` 找到下列参与者的 orig、brainmask、Destrieux 和触点 RAS 文件：

`D0075`、`D0079`、`D0080`、`D0081`、`D0085`、`D0086`、`D0088`、`D0129`、`D0131`、`D0134`、`D0137`。

这 11 位具有现有 MAPER 流程所需的原生 orig/brainmask 输入，可以在核实重建版本后用于重算，无需重新上传相同备份。本次仅做了标签和坐标复核，尚未提交 MAPER。

## 5 需要补回的 43 位与文件

其余 **43 位**没有在本次检查范围内找到匹配的个体重建备份。完整清单带有重建文件夹名和预期恢复路径：

[43 位恢复清单](evidence/zero_endpoints_2026-10-17/restore_43_subjects.csv)

优先恢复 **D125**，以便解决已发现的 opercular 灰质疑点。恢复时保留同一版 FreeSurfer 重建中的：

- `mri/orig.mgz`：原生坐标与体素转换依据。
- `mri/brainmask.mgz`：现有 MAPER 流程的输入。
- `mri/aparc.a2009s+aseg.mgz`：独立的分区／组织核对。
- `elec_recon/` 内触点名称及 RAS 坐标文件：核查坐标来源与电极定位。
- 如有 `aseg.mgz`、`wm.mgz`、`ribbon.mgz`，一并保留以支持组织边界复核。

本次目标中的 D0107 对应 **D107B**，D0139 对应 **D139A**；不能以其他 A/B 重建替代。恢复清单按当前 132 对的任务与重建版本填写，不是整个论文队列的完整恢复清单。

恢复后先完成零端点组织核对，再确定纳入名单；当前数据不足以对全体 54 位直接重算 MAPER。后续关键个体影像、最终分割与标签表宜保留持久副本，避免再次仅依赖 cwork。

## 6 逐触点证据与可重复核对

以下小型表格随论文代码保留；不只依赖临时结果目录：

- [中文逐触点核对表](evidence/zero_endpoints_2026-10-17/零端点逐触点核对.csv)
- [131 个独立端点的完整证据](evidence/zero_endpoints_2026-10-17/unique_zero_contacts.csv)
- [144 次端点记录及逐条坐标匹配](evidence/zero_endpoints_2026-10-17/zero_endpoint_occurrences.csv)
- [132 对双极位点汇总](evidence/zero_endpoints_2026-10-17/pair_summary.csv)
- [54 位参与者的文件可用性](evidence/zero_endpoints_2026-10-17/anatomy_availability.csv)
- [共享备份的 11 位](evidence/zero_endpoints_2026-10-17/shared_backup_11_subjects.csv)
- [仍存的较新 cwork 重建](evidence/zero_endpoints_2026-10-17/surviving_cwork_recons.csv)
- [仍存的较新 MAPER 产物](evidence/zero_endpoints_2026-10-17/surviving_cwork_maper_products.csv)
- [数值汇总](evidence/zero_endpoints_2026-10-17/summary.json)
- [原始目标 132 对及其 HGA 坐标](evidence/zero_endpoints_2026-10-17/target_132_pairs.csv)

端点核对脚本：[audit_zero_endpoints.py](../scripts/audit_zero_endpoints.py)。原始坐标按 native FreeSurfer tkRAS mm 使用，通过 `orig.mgz` 的 `vox2ras_tkr` 逆变换取最近体素；共享重建的触点坐标亦逐项核对。体素结果与历史标签未发现不一致。

文件搜索覆盖项目 cwork 原生重建、MAPER、相关四个 BIDS 数据集，以及共享重建和可访问的项目／团队存储；标准模板、测试用合成影像和其他参与者的图像未用作本次个体证据。输入文件路径、哈希与搜索记录保留在 `results/audit/zero_endpoints_2026-10-17/`。

| 作业 | Slurm 编号 | 用时 | 日志 |
|---|---|---|---|
| 解剖文件盘点 | 58865645 | 1 分 40 秒 | `logs/slurm/anatomy_inventory_58865645.out` |
| 零端点标签与原生坐标核对 | 58865696 | 30 秒 | `logs/slurm/zero_endpoint_audit_58865696.out` |
