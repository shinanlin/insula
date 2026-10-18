# Insula Paper：sound-only 三阶段 NMF

工作区 `/cwork/ns458/insula-paper`，分支 `codex/paper`。主分析遵循 [MAIN_ANALYSIS.md](MAIN_ANALYSIS.md)。

## 当前输入

- 采样 **348 个位点／65 位参与者**；包括按用户最新决定保留并明确标记的 25 个 Unknown 端点位点。
- sound 阳性 **231 个位点／62 位参与者**；另 110 个缺完整 sound 统计、7 个在完整记录下未显著。缺失不能判阴性。
- 筛选仅使用 sound Repeat 在 Stimulus 0–0.6、Delay 0–1、Response −0.5–0.5 s 开区间的显著 mask。
- Figure 1 保存的三个阶段 sound site means 完整覆盖全部 231 个阳性位点；没有 image/text 输入、缺段补零或旧 26 位点名单。
- 拟合窗口：Stimulus 0–0.6、Delay 0–1、Response −0.5–1 s，按闭区间取样，共399列（77、129、193）。
- 可用任务在位点内等权平均；负值截断后整行 L2 归一化。
- 群体曲线直接对 electrode 波形等权平均，保留电极层面的 SEM；不先平均参与者内位点。
- PCA／k-means 按旧函数选择单个解释方差 >5% 的 PC（保留原函数最少两维兜底），不用累计 90% 规则。当前数据保留 3 个 PC；这项更改不重新拟合 NMF。

## 重建顺序

1. `scripts/slurm/build_sound_only_cohort.sh`：重建 sound 采样／显著清单，明确标记保留的 Unknown 端点。首次本轮恢复使用 `scripts/slurm/restore_unknown_cohort.sh`，先保存旧版快照。
2. `scripts/slurm/execute_fig1_notebook.sh`：执行原 Figure 1 Notebook，更新 SVG 和 site means。
3. `scripts/slurm/nmf_three_phase_fit.sh`：同一 sound 名单，k=3，20 次初始化，归一化 H 后按 W 最大载荷赋标签。
4. `scripts/slurm/nmf_three_phase_rank.sh`：同一输入，k=2–6，每个 k 200 次 80% 子抽样；最高 cophenetic 不覆盖指定 k。
5. `scripts/slurm/execute_fig2_notebook.sh`：更新 Figure 2 并验证名单、模态、因子和标签。

## 当前结果

成分位点数：{"motor": 129, "sustain": 67, "sensory": 35}。20 次初始化中 20 次收敛，平均初始化 ARI 为 1.0000。

k=3 cophenetic 为 **0.9920**、子抽样平均 ARI 为 **0.9330**；最高 cophenetic 对应 k=2。完整秩评估见 `results/nmf/rank_selection_metrics.csv`。本轮恢复、逐端点标签、旧版快照与验证见 [Unknown 恢复记录](unknown_endpoints_restored_2026-10-17.md)。先前 323／213 版本是恢复 Unknown 前的 sound-only 版本；227 位点混合模态及218位点诊断均为历史结果。
