# Insula 原生影像恢复与零端点复核

日期：2026-10-17。本记录接续 [此前的零端点审计](zero_endpoint_audit_2026-10-17.md)。恢复范围为 132 对双极位点涉及的 54 位参与者，不能视作整个论文队列的完整影像备份。

**同日后续：** 已按用户指定完成 [D0125 的 MAPER 与 LIA6 核对](maper_D0125_LIA6_2026-10-17.md)。LIA6 的 MAPER 中心分类为 CSF，FreeSurfer 为盖部灰质；不支持把它视作已确认白质端点。本文下方“尚未提交个体 MAPER”的状态保留为恢复完成时的记录。其他参与者与 NMF 尚未重跑。

## 1 恢复已完成

- Windows Globus 源目录：`/C/Users/ns458/Box/CoganLab/ECoG_Recon_Full`。
- DCC 接收目录：`/cwork/ns458/ECoG_Recon`。
- **54 位参与者、863 个文件、521,190,299 字节（约 521 MB）**，Globus 状态为 **SUCCEEDED**。
- Globus 任务：`998f6fea-ca7d-11f1-9382-0effdcd59acd`。启用传输校验和校验；文件存在性、大小及接收端 SHA256 另有记录。
- 恢复 `orig.mgz`、`brainmask.mgz`、`aparc.a2009s+aseg.mgz`，以及可用的 aseg、wm、ribbon、aparc+aseg 和触点名称／RAS 文件。此次恢复的是已有 FreeSurfer 重建文件，未重新运行 FreeSurfer。
- **D0107 使用 D107B，D0139 使用 D139A**，没有互换植入版本。

所有目标文件在恢复前均缺失；此次没有请求源端修改或目标端删除。54 位的 orig、brainmask、Destrieux 分区图像形状与物理仿射一致。

与仍存共享备份交叉比较的 **11 位、33 个 MRI 文件 SHA256 全部一致**。131 个保存坐标均能在恢复源或共享源中找到一致的触点记录，其中恢复的 Windows 源直接匹配 125 个，其余 6 个属于下述 D0086 版本差异。

## 2 131 个独立零端点的恢复后分区

按历史 HGA 保存的原生 tkRAS 坐标，通过各自 orig 的 `vox2ras_tkr` 逆变换重新取最近体素。以下统计描述这些**保存坐标**在恢复的 FreeSurfer 分区中的标签，尚不能替代重算后的 MAPER 组织分类：

| 恢复后的 FreeSurfer 标签类别 | 独立触点数 |
|---|---:|
| Cerebral-White-Matter | 67 |
| 岛叶或岛叶周围沟 | 27 |
| Unknown | 30 |
| 额下回盖部 | 1 |
| 其他皮层灰质 | 2 |
| 其他标签：WM-hypointensities 2、尾状核 1、丘脑 1 | 4 |
| 合计 | **131** |

旧的 72 个明确白质标签来自历史缓存。恢复后有 5 个标签不一致，全部属于 D0057，因此不能继续把旧的 72 当作本次复核结果。

### D0125：opercular 标签在恢复影像中得到复现

- 双极位点 **D0125_LIA5-6** 的端点 **LIA6**。
- 保存坐标与恢复的触点 RAS 文件一致；原生坐标为 **−31.7141、26.0286、7.9286 mm**。
- 恢复后的原生 Destrieux 分区体素为 **`ctx_lh_G_front_inf-Opercular`**，与历史缓存相同。
- 该双极位点存在于旧 255 个 NMF 输入中。

因此，“Hammers 端点标签 0 都是白质、均不涉及邻区灰质”的解释不能成立。这里确认的是个体 FreeSurfer 分区标签及坐标一致性；与岛叶—盖部组织边界的最终关系仍应结合个体影像和重建后的 MAPER 组织分类判断。

### D0057：5 个标签差异

6 个待核对触点的坐标均与恢复的 RAS 文件一致，其中 5 个原生分区标签不同：

| 触点 | 历史标签 | 恢复后的分区标签 |
|---|---|---|
| RAI2 | Right-Cerebral-White-Matter | Right-Caudate |
| RPI1、RPI3 | Right-Cerebral-White-Matter | WM-hypointensities |
| RPI5、RPI6 | Right-Cerebral-White-Matter | ctx_rh_S_cingul-Marginalis |

这些差异提示历史标签与当前影像来源尚未完全统一。不能仅凭当前标签把这些触点重新认定为上述脑区，也没有据此自动删除位点。需要核对重建版本，并结合 MRI 和新的组织分割复核。

### D0086：6 个触点的坐标版本差异

恢复的 Windows RAS 文件中，LTMM6、LTPI3、RTMM6、RTMM7、RTPS4、RTPS9 与历史 HGA 坐标不一致；最小差异约 **35–87 mm**。Windows 触点名称文件日期为 2024-03-25，共享备份为 2023-12-01，两套坐标不能直接互换。

历史保存坐标已在先前的共享备份中找到一致记录。新文件未被用于改写 HGA 坐标或通道名称。此前所说 RTPS4 的丘脑标签，只能解释为保存坐标的体素标签，不能由此确定当前 Windows 同名触点的实际位置。逐触点表分别保留恢复源和共享源的匹配情况；逐文件哈希比较另存。

补充比较确认 D86 的 **orig、brainmask、aparc.a2009s+aseg 三个 MRI 文件与共享备份逐字节一致**。目前发现的来源差异涉及触点文件；MRI 文件相同并不能解决旧通道名称、左右侧和实际触点之间的对应关系。

## 3 MAPER 图谱输入已恢复

从 [Hammers n30r95 作者发布页](https://soundray.org/hammers-n30r95/) 恢复了 30 套原始 MRI 和 95 区标签。原始压缩包及其 SHA256 保留在持久存储：

`/hpc/group/coganlab/nanlinshi/cache/maper/hammers_n30r95/`

预配准变换和 companion MRI 来自项目已有的 `hammers_mith-ancillaries-n30r95.tar`。完整运行输入整理到：

`/cwork/ns458/maper_run/restore_20261017/ancillaries/`

沿用项目现有的原始图谱标签几何校正步骤；随后逐套检查 companion MRI、标签和预变换。9 套 companion MRI 将 0.9375 mm 网格间距舍入为 0.938 mm，已按物理坐标最近邻对齐，并要求全体体素的标签数组保持不变、最大索引位移小于半个体素；本次最大索引位移为 **0.05254 体素**。边界采用最近值延拓，避免恒零填充误删边缘标签。**30 套全部通过**，记录见图谱输入清单。

尚未提交个体 MAPER 重建，也未重新拟合 NMF 或更改纳入名单。下一步应先用 D0125 验证恢复后的 MAPER 流程及组织取样，再批量重建；D0057 的标签差异和 D0086 的坐标来源需要单独保留并处理。

## 4 固定证据与复现入口

- [54 位恢复清单](evidence/globus_recon_restore_2026-10-17/windows_subject_inventory.csv)
- [863 个文件的源路径、目标路径和大小](evidence/globus_recon_restore_2026-10-17/transfer_file_manifest.csv)
- [Globus 完成记录](evidence/globus_recon_restore_2026-10-17/completed_transfer_summary.json)
- [恢复文件 SHA256](evidence/globus_recon_restore_2026-10-17/restored_file_hashes.json)
- [131 个触点的逐体素／坐标复核](evidence/globus_recon_restore_2026-10-17/restored_zero_contact_voxel_checks.csv)
- [恢复后的 MRI 几何检查](evidence/globus_recon_restore_2026-10-17/restored_geometry_checks.csv)
- [共享备份与恢复影像的逐文件比较](evidence/globus_recon_restore_2026-10-17/shared_backup_source_comparison.csv)
- [30 套图谱输入及几何核对记录](evidence/globus_recon_restore_2026-10-17/atlas_input_restore.json)
- [复核汇总](evidence/globus_recon_restore_2026-10-17/restored_verification_summary.json)

恢复清单生成脚本：`scripts/inventory_globus_recon_restore.py`；传输脚本：`scripts/submit_globus_recon_restore.py`；影像核对：`scripts/verify_restored_native_anatomy.py`；图谱恢复：`scripts/restore_maper_atlas_inputs.py`。详细结果仍在 `results/audit/globus_recon_restore_2026-10-17/`；这里固定的小型证据不依赖该临时结果目录。

| 完成的检查 | Slurm 作业 | 日志 |
|---|---|---|
| Windows 文件盘点及失败重试 | 58865938、58865979 | `logs/slurm/globus_recon_inventory_*.out` |
| 全部恢复文件、原生分区与触点坐标 | 58866440 | `logs/slurm/recon_restore_verify_58866440.out` |
| 共享源／恢复源补充比较 | 58866507 | `logs/slurm/recon_restore_verify_58866507.out` |
| 图谱输入恢复与最终几何检查 | 58866492 | `logs/slurm/maper_atlas_restore_58866492.out` |

图谱输入准备的中间作业 58866310、58866443、58866453 因 companion 网格检查而停止；这些中间文件未用于个体分析。最终成功作业 58866492 重新生成并验证了完整的 30 套输入。
