# R01 Grant Figure Checklist

> Organized by Specific Aim / Sub-aim. Each figure entry includes: data source, task(s), what to plot, and revisions needed.

---

## Approach (C3 Section)

### Figure C3-1 — Electrode Coverage

| Item | Detail |
|------|--------|
| **Section** | C3 Approach |
| **What to plot** | Two views: (1) whole-brain electrode distribution across all patients; (2) AIC electrode distribution specifically |
| **Key message** | Demonstrate comprehensive SEEG coverage with dense sampling of the AIC |
| **Outputs** | `grantfig/c3_1_whole_brain_electrode_coverage.svg`, `grantfig/c3_1_aic_electrode_coverage.svg` |
| **Status** | Draft in `grant/methods.ipynb` |
| **TODO** | Currently includes electrodes from all 5 tasks. Final version should use a specific subset of subjects (TBD). Keep as-is until subject selection is finalized. |

### Figure C3-2 — AIC Electrode and Signal Characterization

| Item | Detail |
|------|--------|
| **Section** | C3 Approach |
| **What to plot** | Two panels: |
| | **Panel A:** Schematic illustration showing electrode insertion into the AIC |
| | **Panel B:** Time-Frequency Representation (TFR) of the electrode signal |
| **Key message** | Demonstrate the recording methodology and signal quality from AIC electrodes |
| **Outputs** | `grantfig/c3_2_aic_electrode_strip.svg`, `grantfig/c3_2_example_aic_tfr.svg` |
| **Status** | Draft in `grant/methods.ipynb` |

---

## Aim 1: Domain-General Operations of the AIC During Speech

> **Notebook:** `grant/aim1.ipynb` · **Figure outputs:** `grant/grantfig/aim1_*.svg`
>
> **Status (2026-05):** Aim 1 draft figures are **largely complete**. Remaining work: (1) 1.1B common-subject count investigation; (2) 1.1C2 alternative visualization TBD; (3) 1.2B deferred; (4) finalize captions. Spatial Brain SVGs must be re-rendered **locally** (Jupyter + MNE Brain backend); HPC headless often fails.

### Aim 1 — design conventions (recorded)

| Topic | Decision |
|-------|----------|
| **Insula spatial view** | Top-down insula focus: `show_view(azimuth=180/0, elevation=90, distance=180, focalpoint=lh/rh_insula_center)`; insula labels gray (`alpha=0.6`). Used for 1.1A/B/C spatial and 1.3 spatial. |
| **Overlap spatial (1.1A/B/C1/C2)** | Category-colored points (`Overlap` / `-only`); subtle size range `SPATIAL_SIZE_MIN=15`, `MAX=19`, `gamma=0.35`; no opacity. Helper: `plot_aic_spatial_categories()`. |
| **HGA magnitude spatial (fig2-style)** | Used for 1.2A heatmaps and **1.3 spatial**: opaque points, `Normalize(0, p95)`, size **8–30**, `lighting=False`, no `opacity`. |
| **Effect-size spatial (fig4-style)** | **Not used** for grant Aim 1 after 2026-05 review; 1.3 spatial was switched from fig4 (`TwoSlopeNorm` + alpha) to fig2-style for visual consistency. |
| **SMC definition** | `PrG`, `PoG`, `Subcentral` → `SMC` in notebook. Contrast region for 1.2; **not** PIC. |
| **Common subjects** | Required when comparing across tasks (1.1B, 1.1C2). Helpers: `common_subjects_across_tasks()`, `subset_panel_c2_common()`. |
| **Sternberg load** | Any-load rule: channel significant if **any** of load3/5/7/9 has `mask==True` in window (`get_sig_channels_any_load`). |
| **Obsolete outputs** | Early/wrong drafts still on disk but **not** in current notebook pipeline: `aim1_1b_output_effector_*.svg`, `aim1_2_maintenance_*.svg`, `aim1_compare_action_*.svg`, etc. Ignore unless resurrecting old designs. |

**Key helpers** (all in `aim1.ipynb` helpers cell ~9): `common_subjects_across_tasks`, `subset_panel_c2_common`, `plot_aic_spatial_categories`, `plot_repeat_channel_heatmap`, `load_decision_vs_repeat_contrast`, `get_decision_gt_repeat_channels`, `plot_insula_decision_effect_spatial`, `get_sig_channels_any_load`. Re-run helpers cell after any helper edit before downstream panels.

**Run order:** setup imports → helpers (~9) → panel cells in notebook order. Spatial cells require local Brain backend.

### Aim 1.1 — Domain-Generality of AIC Engagement

Multi-panel composite figure demonstrating AIC engagement is not speech-specific across three dimensions: content, input modality, and output effector.

#### Panel A: Input Modality

| Item | Detail |
|------|--------|
| **Sub-aim** | 1.1 — input modality manipulation (auditory vs. visual) |
| **Task** | Picture Naming (image vs. sound conditions) |
| **Data** | AIC electrode HGA |
| **What to plot** | Contrast between image and sound conditions (NOT PIC contrast). Should plot the spatial distribution and the temporal waveform. The spatial distribution should be the overlap between sound and image. |
| **Time window** | Stimulus window, extended by 0.5–1 second into delay period |
| **Regions** | AIC only; avoid including posterior auditory cortex responses |
| **Outputs** | `aim1_1a_input_modality_AIC.svg`, `aim1_1a_input_modality_spatial_AIC.svg` |
| **Key message** | AIC activates regardless of input modality |
| **Status** | Draft complete in `grant/aim1.ipynb` |

#### Panel B: Lexical Delay vs No-Delay (Decision)

| Item | Detail |
|------|--------|
| **Sub-aim** | 1.1 — delay manipulation under matched cognitive demand (lexical decision) |
| **Tasks** | Lexical No-Delay and Lexical Delay, **Decision** condition only; common subjects across tasks |
| **Data** | AIC electrode HGA (sound stimuli) |
| **What to plot** | Temporal traces (stimulus + response) and spatial overlap (No-delay-only / Delay-only / Overlap) |
| **Time window** | Spatial overlap: response window (−0.5 to 0.5 s); temporal: stimulus + response panels |
| **Key message** | AIC engagement for lexical decision is comparable with and without delay period |
| **Status** | Draft in `grant/aim1.ipynb` — `aim1_1b_delay_nodelay_decision_AIC.svg` and `aim1_1b_delay_nodelay_decision_spatial_AIC.svg` |
| **Known issue** | See [Open issues — Panel B common subjects](#open-issues--panel-b-common-subjects) below. Current plots use only **4** common subjects; expected count is much higher. Treat drafts as provisional until this is resolved. |

#### Panel C: Content / Spatial Overlap

| Item | Detail |
|------|--------|
| **Sub-aim** | 1.1 — content manipulation (environment vs. speech) |
| **Data source** | `../../sternberg/results/EnvironmentalSternberg(bipolar)` (`*_condition.csv`); overlap with insula `HGAs` LexicalDelay Repeat |
| **Load rule** | Aggregate load3/5/7/9: channel significant if **any** load has `mask == True` in the analysis window |
| **C1** | Environment vs **Speech** (word + nonword combined); temporal 4-phase per-channel heatmap (Environment / Word / Non-word) + spatial overlap |
| **C2** | Sternberg **environment** (maintenance, any load) vs LexicalDelay **Repeat** (**delay**); **common subjects only** (`subset_panel_c2_common`); temporal overlay on one axes; phases in `PANEL_C_CONFIG` |
| **Outputs** | `aim1_1c_environment_vs_speech_AIC.svg`, `aim1_1c_environment_vs_speech_spatial_AIC.svg`, `aim1_1c_sternberg_vs_repeat_AIC.svg`, `aim1_1c_environment_vs_repeat_spatial_AIC.svg` |
| **Regions** | AIC only |
| **Key message** | AIC shows overlapping but not identical spatial activation for environment, speech, and production-related activity |
| **Status** | Draft in `grant/aim1.ipynb` — run cells locally for Brain spatial SVGs |
| **Known issue** | See [Open issues — Panel C2 overlap / visualization](#open-issues--panel-c2-overlap--visualization). Cross-task spatial overlap yields **very few** overlapping electrodes, so current plots look weak; may switch visualization approach later. **Keep current drafts for now.** |

---

### Aim 1.2 — Dissociating Maintenance from Sensory/Execution

| Item | Detail |
|------|--------|
| **Sub-aim** | 1.2 — dissociate maintenance from sensory and execution patterns |
| **Panel A** | **AIC vs SMC** regional comparison (fig2-style): Lexical **Repeat**, sound, four phases (`stimulus`, `delay`, `go`, `response`); hue = ROI (not AIC vs PIC) |
| **Panel A outputs** | `aim1_2a_aic_vs_smc_repeat.svg` (lineplot); `aim1_2a_heatmap_AIC_L.svg`, `aim1_2a_heatmap_SMC_L.svg` (per-channel heatmaps, fig2-style) |
| **Panel B** | **Two planned comparisons** (to implement later; **skip for now**): **(1) Passive vs Delayed Repeat** — dissociate maintenance from passive sensory; **(2) Delayed Repeat vs No-delay Repeat** — maintenance vs execution timing (stimulus + response windows). Likely separate figures per ROI (AIC, SMC) as needed. |
| **Panel B (placeholder)** | Current notebook cells plot all **three** conditions on one axes (`Passive` / `No-delay Repeat` / `Delayed Repeat`) — **provisional only**, not the final Panel B design. |
| **Panel B outputs (planned)** | `grantfig/aim1_2b_passive_vs_repeat_AIC_SMC.svg` (B1: Passive vs Repeat, AIC+SMC × stimulus+response) |
| **Panel B1 note** | SMC Passive trace in **Response** column is display-shifted **−0.7 s** post-hoc; all panels use **xlim −0.5 to 1.5 s** |
| **Data** | AIC and SMC HGA (PrG/PoG/Subcentral → SMC); sound stimuli |
| **What to plot (B1)** | **Passive vs Delayed Repeat**: stimulus-aligned — passive flat, delayed repeat sustained during delay |
| **What to plot (B2)** | **Delayed Repeat vs No-delay Repeat**: stimulus window (no-delay brief bump vs sustained delay) + response window (similar pre-response ramps) |
| **Stimuli** | Common lexical stimuli across no-delay and delay tasks |
| **Contrast region** | SMC (not posterior cortex) |
| **Key message** | AIC is silent during passive listening but shows sustained delay activity when maintenance required; SMC provides motor/execution contrast in Panel A |
| **Status** | Panel A draft complete (`aim1_2a_*`). Panel B1 draft: `aim1_2b_passive_vs_repeat_AIC_SMC.svg` (Passive vs Repeat; SMC Passive response −0.7 s shift). B2 deferred. |

---

### Aim 1.3 — Cognitive Demand Modulation

| Item | Detail |
|------|--------|
| **Sub-aim** | 1.3 — AIC scales with cognitive demand |
| **Tasks** | Lexical Delay: Repetition and decision. |
| **Data** | Lexical Delay |
| **What to plot** | AIC activity comparison: higher engagement for more demanding task (lexical decision > repetition). Include temporal waveform and AIC-only insula spatial points. |
| **Outputs** | `aim1_3_cognitive_demand_AIC.svg` (temporal); `aim1_3_cognitive_demand_spatial_AIC.svg` (AIC-only spatial points) |
| **Spatial rule** | Load `DecisionVsRepeat` univariate contrast for `LexicalDelay`; show channels with `significant == True`, `mask == True`, `direction == Decision`, `phase == delay`, and `0 < time < 0.8`; aggregate `mean_diff` by channel. Render fig2-style (opaque points, `Normalize(0, p95)`, size 8–30). |
| **Key message** | AIC activity scales with cognitive demand, consistent with domain-general control |
| **Status** | Draft complete — temporal + spatial SVGs in `grantfig/` (spatial: fig2-style opaque points; re-run locally if Brain backend changes) |

---

### (Optional) Reaction Time Figure

| Item | Detail |
|------|--------|
| **Placement** | After Aim 1.3 in `grant/aim1.ipynb` |
| **Data** | Existing RT prediction h5: `results/LexicalDelay(bipolar)/`, `results/LexicalNoDelay(bipolar)/` |
| **LexicalDelay** | Decision; **Delay phase**; any time point with `mask==True` (cluster-corrected RT prediction) |
| **LexicalNoDelay** | Decision; union of **Stimulus** and **Response**; any significant cluster in either phase (no time window) |
| **What to plot** | **Temporal lineplot:** AIC mean R² vs time on shared **Stimulus / Response** panels; LexicalDelay (speech) vs LexicalNoDelay (button) Decision overlaid; cluster-corrected sig bar (≥10% channels per task). **Temporal heatmap:** 2×2 per-channel R² (rows = Delay / No-delay; cols = Stimulus / Response); channel pools match spatial (Delay phase sig vs Stimulus∪Response sig); non-sig → NaN; sorted by first sig time per phase. **Spatial:** Delay-task-only / No-delay-task-only / Overlap (Pearson r as point size) |
| **Outputs** | `grantfig/aim1_rt_delay_vs_nodelay_decision_temporal_AIC.svg` (lineplot); `grantfig/aim1_rt_delay_vs_nodelay_decision_heatmap_AIC.svg` (heatmap); `grantfig/aim1_rt_delay_vs_nodelay_decision_spatial_AIC.svg` (spatial) |
| **Status** | Temporal lineplot + heatmap + spatial code in `grant/aim1.ipynb`; re-run in `ieeg` to regenerate SVGs |

---

## Aim 2: Speech-Specific Representations in the AIC

### Aim 2.1 — Encoding/Decoding of Linguistic and Articulatory Content

| Item | Detail |
|------|--------|
| **Sub-aim** | 2.1 — encoding and decoding models |
| **Task** | PhonemeSequence (articulatory + phoneme); Lexical Delay (lexicality repeat + decision) |
| **Data** | AIC electrode activity (Left hemisphere; no PIC) |
| **What to plot** | 2.1A: articulatory + phoneme window decoding (Stimulus vs Response scatter; phoneme uses recording-1). 2.1C/D: time-resolved lexicality (Repeat vs Decision). |
| **Key revision** | REMOVE PIC from this figure |
| **Key message** | AIC carries both higher-level linguistic and articulatory motor content |
| **Draft outputs** | `grantfig/aim2_1a_articulator_window.svg`; `grantfig/aim2_1a_phoneme_window.svg`; `aim2_1c_lexicality_repeat_temporal.svg`; `aim2_1d_lexicality_decision_temporal.svg` |
| **Notebook** | `grant/aim2.ipynb` (helpers + panel cells) |
| **Status** | **Draft complete** — review panel-by-panel. |

---

### Aim 2.2 — Cross-Task Generalization of Representations

| Item | Detail |
|------|--------|
| **Sub-aim** | 2.2 — cross-task decoding |
| **Task** | Lexical Delay (decision condition) + cross-decoding between conditions |
| **Data** | AIC electrode activity |
| **What to plot** | Cross-task decoding: train Repeat → test Decision (AIC, Delay phase, square aspect) |
| **Key message** | AIC representations generalize abstractly across output goals |
| **Draft outputs** | `grantfig/aim2_2b_cross_repeat2decision_delay.svg` |
| **Status** | **Draft complete** — 2.2B cross-decode overlay (Decision within-task shown in 2.1D). Review panel-by-panel. |

---

### Aim 2.3 — Single-Neuron Evidence (Microwire)

| Item | Detail |
|------|--------|
| **Sub-aim** | 2.3 — microwire recordings |
| **Data** | SEEG electrodes with novel microwires in AIC |
| **What to plot** | Single-neuron examples showing co-existence of higher-order linguistic and articulatory information |
| **Key message** | Higher-order and articulatory information coexist within single AIC neurons |
| **Status** | **Out of scope** for current `aim2.ipynb` pipeline (microwire / single-neuron) |

---

## Aim 3: AIC as Cognitive-Motor Interface in the Speech Network

### Aim 3.1 — Effective Connectivity (CCEPs)

| Item | Detail |
|------|--------|
| **Sub-aim** | 3.1 — cortico-cortical evoked potentials |
| **Data** | CCEP data: AIC stimulation → speech network regions (and vice versa) |
| **What to plot** | Connectivity map between AIC and canonical speech regions |
| **Key message** | AIC has direct effective connectivity with both cognitive and motor speech regions |
| **Status** | WAITING FOR CCEP FIGURE — data/analysis not yet ready |

---

### Aim 3.2 — Directed Functional Connectivity

| Item | Detail |
|------|--------|
| **Sub-aim** | 3.2 — directed functional connectivity during task |
| **Data** | Pulling from all the repetition tasks. High gamma cross-correlation between AIC and speech regions |
| **What to plot** | Temporal flow: cognitive regions → AIC → motor regions. Two time windows: (1) early window: cognitive-to-AIC flow; (2) later window: AIC-to-motor flow |
| **Key revision** | Current figure shows AIC leads SMC in stimulus window — need to also show cognitive-to-AIC directionality. REMOVE PIC. |
| **Key message** | Information flows cognitive → AIC → motor; AIC is an intermediary hub |
| **Status** | NEEDS MAJOR REVISION: reorganize for two temporal windows; remove PIC; show full flow |

---

### Aim 3.3 — Cross-Regional Multivariate Decoding

| Item | Detail |
|------|--------|
| **Sub-aim** | 3.3 — cross-regional decoding (windowed + CCA) |
| **LexicalDelay** | `lexicality`; train `AICl` → test `IFGl`, `STGl`, `MFGl`, `SMCl`; Repeat; 4 phases |
| **PhonemeSequence** | `articulator` (recording-1); same partner ROIs/phases |
| **Pipeline** | `src/run_cross_roi_window.py` + `scripts/cross_roi_window.sh` (32 SLURM jobs) |
| **Results** | `results/{Task}(cross_roi)(bipolar)/sub-AICl2{partner}/(cross)(window){datatype}/` |
| **What to plot** | Two panels: lexicality vs articulator cross-decode accuracy by partner (IFG/STG/MFG/SMC) × phase |
| **Outputs** | `grantfig/aim3_3_lexicality_cross_roi.svg`, `grantfig/aim3_3_articulator_cross_roi.svg` |
| **Key message** | AIC shares linguistic (lexicality) and articulatory (PhonemeSequence) content with frontal/temporal/motor partners |
| **Status** | **IN PROGRESS** — script/notebook pending Agent mode write; do not use old `cross_roi` resolved results |

---

## Supplementary / Grant-Level Items

### Task Summary Table

| Item | Detail |
|------|--------|
| **Purpose** | Clarify for reviewers which tasks map to which aims (multiple aims share tasks) |
| **Content** | Table listing each task, key properties (stimuli, modality, delay, response type), and which aims/sub-aims use it |
| **Key message** | Reduce reviewer confusion about overlapping task structure |
| **Status** | NEEDS CREATION from scratch |

### Figure Captions

| Item | Detail |
|------|--------|
| **Purpose** | Each figure needs a complete caption for the grant |
| **Status** | Pending — write after figures are finalized |

---

## Open issues (follow-up)

Items to revisit after remaining grant figures are in place. Do not block other panels unless noted.

### Open issues — Panel B common subjects

| Item | Detail |
|------|--------|
| **Figure** | Aim 1.1 Panel B — Lexical Delay vs No-Delay (Decision) |
| **Notebook** | `grant/aim1.ipynb` — `common_subjects_across_tasks()` (helpers cell ~9), Panel B temporal and spatial cells under Aim 1.1 |
| **Symptom** | After filtering to subjects present in **both** `LexicalNoDelay` and `LexicalDelay` with `description == Decision`, `roi == AIC`, and `modality == sound`, only **4** common subjects remain. This is likely too few for the intended comparison. |
| **Expectation** | Common-subject count should be **substantially higher** (on the order of many more patients with both tasks; rough memory: ~20+ at task level before strict ROI filters). |
| **Current impact** | Waveform and spatial overlap figures (`aim1_1b_delay_nodelay_decision_AIC.svg`, `aim1_1b_delay_nodelay_decision_spatial_AIC.svg`) are **provisional** — hue/overlap logic is correct (task-level contrast, not Repeat vs Decision), but N may be wrong. |
| **Checks when revisiting** | (1) Re-count common subjects **without** AIC filter vs **with** AIC filter. (2) Confirm `subject` IDs are consistent across `LexicalNoDelay` and `LexicalDelay` HGA paths. (3) Verify Decision rows exist for both tasks after insula AIC/PIC reclassification and `exlude_insula.csv` exclusions. (4) Compare to Aim 1.2 subject set (passive / no-delay repeat / delayed repeat) for consistency. (5) Print per-task subject lists in notebook before intersection to see where drop-off occurs. |
| **Status** | **TODO** — investigate after other grant figure work is done |

### Open issues — Panel C2 overlap / visualization

| Item | Detail |
|------|--------|
| **Figure** | Aim 1.1 Panel C2 — Environmental Sternberg (environment, maintenance) vs LexicalDelay Repeat (delay) |
| **Notebook** | `grant/aim1.ipynb` — `subset_panel_c2_common()` (helpers cell ~9), C2 temporal (~25), C2 spatial (~27) |
| **Symptom** | After restricting to **common subjects** across Environmental Sternberg and LexicalDelay Repeat, the number of **spatially overlapping significant electrodes** (cross-task overlap categories) is **very small**. Temporal and spatial SVGs therefore **do not read well** for grant purposes. |
| **Decision (2026-05)** | **Leave current implementation and outputs as provisional drafts** (`aim1_1c_sternberg_vs_repeat_AIC.svg`, `aim1_1c_environment_vs_repeat_spatial_AIC.svg`). We may adopt **alternative visualization** later (e.g., different overlap definition, summary metric, or non–brain-surface display). No change required until that direction is chosen. |
| **Status** | **Documented** — acceptable placeholder; revisit visualization strategy when Panel C is finalized |

---

## Summary Status

| Figure | Aim | Status |
|--------|-----|--------|
| C3-1 — Electrode Coverage | Approach | Draft in `methods.ipynb` |
| C3-2 — AIC Electrode & TFR | Approach | Draft in `methods.ipynb` |
| 1.1A — Input Modality | 1.1 | Draft complete |
| 1.1B — Delay vs No-Delay (Decision) | 1.1 | Draft complete (**provisional** — only 4 common subjects; see Open issues) |
| 1.1C — Content / Spatial Overlap | 1.1 | Draft complete (C2 overlap sparse — see Open issues) |
| 1.2A — AIC vs SMC (Repeat) | 1.2 | Draft complete (lineplot + AIC/SMC heatmaps) |
| 1.2B — Maintenance contrasts | 1.2 | **Deferred** — (1) Passive vs Delayed Repeat; (2) Delay vs No-delay Repeat |
| 1.3 — Cognitive Demand | 1.3 | Draft complete (temporal + fig2-style spatial) |
| 2.1 — Encoding/Decoding | 2.1 | Draft complete (`aim2_1a/c/d_*.svg`) |
| 2.2 — Cross-Task Decoding | 2.2 | Draft complete (`aim2_2b_*.svg`) |
| 2.3 — Microwire Single-Neuron | 2.3 | Out of scope (current task) |
| 3.1 — CCEPs | 3.1 | Waiting for data |
| 3.2 — Directed Connectivity | 3.2 | Needs major revision |
| 3.3 — Cross-Regional Decoding | 3.3 | Check/revise |
| Task Summary Table | All | Needs creation |
| Figure Captions | All | Pending |
