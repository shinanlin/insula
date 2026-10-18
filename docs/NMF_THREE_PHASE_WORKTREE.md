# Insula Paper：sound-only 三阶段 NMF

工作区 `/cwork/ns458/insula-paper`，分支 `codex/paper`。主分析必须遵循 [MAIN_ANALYSIS.md](MAIN_ANALYSIS.md)，参数与颜色统一来自 `src/main_analysis.py`。

## 当前输入

- **采样：323 个位点 / 65 位参与者**。sound 记录与中点、双端点解剖要求均满足。
- **sound 阳性：213 个位点 / 61 位参与者**。仅用 sound Repeat 在 Stimulus 0–0.6 s、Delay 0–1 s、Response −0.5–0.5 s 开区间内的显著性。
- 103 个位点缺完整 sound 证据，7 个在现有记录下完整未显著；缺失证据不算阴性。
- 三阶段 sound 波形均完整覆盖全部 213 个位点；NMF 直接读取 Figure 1 保存的 site means。没有 image/text 波形、额外补零或旧 26 位点名单。
- 拟合窗口：Stimulus **0–0.6 s**、Delay **0–1 s**、Response **−0.5–1 s**。共 **399** 列（77、129、193），其中采样端点按闭区间纳入。
- sound 试次均值先按可用任务在位点内等权平均；整行负值截断后 L2 归一化。

## 重建顺序

1. `scripts/slurm/build_sound_only_cohort.sh`：从核实的解剖和统计证据生成 sound-only 采样与显著名单；输出 `docs/evidence/sound_only_cohort_2026-10-17/`。
2. `scripts/slurm/execute_fig1_notebook.sh`：生成 Figure 1 sound-only 数量、曲线与 SVG；同时保存三阶段 site means。
3. `scripts/slurm/nmf_three_phase_fit.sh`：读取同一 213 位点名单与 Figure 1 sound means；k=3、20 次初始化，选择最小重建误差，归一化 H 并按对应 W 最大载荷赋标签。
4. `scripts/slurm/nmf_three_phase_rank.sh`：同一输入下比较 k=2–6，每个 k 200 次 80% 子抽样。最高 cophenetic 如实报告，不覆盖实际拟合 k。
5. `scripts/slurm/execute_fig2_notebook.sh`：更新 Figure 2，验证 source membership 全部为 sound、名单吻合、W/H 与标签一致，并归档到 `docs/evidence/figure2_sound_only_2026-10-17/`。

## 当前结果

sensory **29**、sustained **61**、motor **123** 位点。20 次初始化均收敛且标签一致。k=3 cophenetic **0.9815**、子抽样 ARI **0.8080**；最高 cophenetic 为 k=2（**0.9871**）。不能写成秩检验选择了 k=3。

原混合模态 227 位点模型及 218 位点诊断对照仅是历史证据，不是正式 sound-only 主分析。先前 218 个有 sound 波形的位点中，只有 **213 个**符合 sound 显著性；“有 sound 波形”不能替代“sound 显著”。详细变化见 [sound-only 修正记录](sound_only_correction_2026-10-17.md)。
