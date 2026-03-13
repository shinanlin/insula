# Paper Narrative: "Initiate and Maintain Goal Command"

## The Story in One Sentence
**The anterior insula (AIC) is not a speech motor area — it is a supramodal action controller that initiates and maintains goal-directed commands across effectors and cognitive demands.**

---

## Core Datasets & Tasks
*All claims in the narrative are anchored by specific conditions across these 5 BIDS datasets.*

1. **`BIDS-1.0_LexicalDecRepDelay` (Core Task)**
   * **Design**: Stimulus → Delay (~1s) → Go Cue → Response. Separates perception, rule maintenance, and motor execution.
   * **Conditions**: `Repeat` (say what you heard) vs `Decision` (Lexical Decision: Yes/No, is it a word?).
2. **`BIDS-1.0_LexicalDecRepNoDelay` (Motor Control Task)**
   * **Design**: No delay. Immediate response after stimulus.
   * **Conditions**: `Repeat` (vocal execution) vs `:=:` (**Passive**, listen only, no motor output).
3. **`BIDS-1.3_PictureNaming` (Input Modality Task)**
   * **Design**: Respond to varied inputs (images, text, sound).
   * **Purpose**: Proves AIC activation is not dependent on auditory input (supramodal input).
4. **`BIDS-1.4_Phoneme_sequencing` (Audio-Motor Task)**
   * **Design**: Listen and repeat non-words/syllables (e.g., vug, gab).
   * **Purpose**: Removes high-level semantics to test pure phonetic-motor representations.
5. **`BIDS-1.4_SentenceRep` (Complexity Task)**
   * **Design**: Listen and repeat full sentences or words.
   * **Purpose**: Tests higher-order sequence/syntactic loading.

---

---

## Section 1: Insula is Underestimated
**Core claim**: Insula is densely sampled in sEEG and gets activated across many speech tasks, yet its role is poorly understood.

| Analysis | Status | Figure | Notes |
|----------|--------|--------|-------|
| Electrode coverage map (N electrodes per ROI) | ✅ Done | Fig 1 | Show dense sampling in insula |
| HGA activation rates (% sig electrodes) | ✅ Done | Fig 1/2 | Show high activation rates across tasks |
| Multi-task HGA traces (LexDelay, LexNoDelay, PicNaming, SentRep) | ✅ Done | Fig 2 | AIC reliably activates across 4 tasks |

> [!TIP]
> **Nothing new needed here.** This section uses existing fig1/fig2 data.

---

## Section 2: AIC vs PIC Dissociation
**Core claim**: AIC and PIC are functionally distinct — AIC shows sustained activity tied to action, PIC shows transient sensory responses.

| Analysis | Status | Figure | Notes |
|----------|--------|--------|-------|
| Sustained "square block" HGA in AIC (Decision Go phase) | ✅ Done | Fig 2 | Clear sustained vs transient dissociation |
| Temporal progression: STG→PIC→SMC→IFG→AIC | ✅ Done | Fig 2d | Onset latency mixed-effects model |
| Lateralization (amplitude-based) | ✅ Done | Fig 2 | PIC left-lateralized (p=0.029); AIC bilateral |
| Lateralization (incidence-based) | ⚠️ Partial | Supp? | Contradicts amplitude; needs debugging or just drop |

> [!IMPORTANT]
> **Decision needed**: Is the incidence-based lateralization worth debugging, or do we just report amplitude-based and move on?

---

## Section 3: Motor/Action Aspect — "Initiate"
**Core claim**: AIC is only active when action is required, can predict RT, and generalizes across input/output modalities.

| Analysis | Status | Figure | Notes |
|----------|--------|--------|-------|
| **Repeat vs Passive**: AIC active only when action required | ✅ Done | Fig 3 | Passive = no action = no AIC |
| **RT prediction**: AIC Delay HGA predicts RT (best predictor) | ✅ Done | Fig 3 | AIC > PIC > STG for RT prediction |
| **Cross-task**: PicNaming (image→button) also activates AIC | ✅ Done | Fig 2 | Shows supramodal, not speech-specific |
| **Effector specificity** (P1): Repeat=mouth vs Decision=hand | 🔴 TODO | Fig 3? | LexNoDelay comparison. **PI top priority** |

> [!WARNING]
> **P1 (Effector Specificity) is still undone.** This is PI's #1 priority. Needed to prove AIC is not just "speech motor" but generalizes across effectors.

### What's needed for P1:
- Compare HGA in LexNoDelay: Repeat (vocal) vs Decision (button press)
- If AIC activates similarly for both → supramodal action controller
- If AIC only activates for vocal → speech-specific (weaker story)

---

## Section 4: Higher-Order Cognition — "Maintain Goal Command"
**Core claim**: AIC activity scales with cognitive demand and maintains task-relevant information throughout planning.

| Analysis | Status | Figure | Notes |
|----------|--------|--------|-------|
| **Univariate: Decision > Repeat** | 🔄 Running | Fig 4 | Batch 44094932. Nightingale rose chart done. AIC = #3 ROI (20 elecs) |
| **Nightingale rose chart** (Top ROIs) | ✅ Done | Fig 4 | OFC(33), SFG(24), AIC(20), MFG(18)... |
| **Brain surface delta plot** | ✅ Done | Fig 4 | Mean diff projected onto pial surface |
| **Cross-condition decoding** (Word/Nonword) | ✅ Done | Fig 4 | Decision = sustained block, Repeat = transient |
| **Temporal generalization 2D** | ✅ Done | Fig 4 | Stimulus shared → Go condition-specific |
| **Phoneme decoding** (articulation planning) | 🔄 Running | Fig 3/4 | Batch 44111650. Resubmitted with all trials |
| **Lexical info maintenance** (Word vs Nonword in AIC) | ✅ Done | Fig 4 | AIC can decode lexicality during Delay |
| **dACC cannot decode** → AIC-dACC dissociation | ✅ Done | Fig 4 | dACC = salience/conflict only, not content |

> [!NOTE]
> PI mentioned wanting "latent space" / fancier analysis. This likely means **temporal generalization matrices** (which we already have!) or **representational similarity analysis (RSA)**. The 2D cross-decoding plots ARE latent space analyses — we should frame them that way in the manuscript.

---

## 🚦 Priority Action Items (Ordered)

### 🔴 Immediate (this week)
1. **P1: Effector Specificity** — Run HGA comparison on LexNoDelay Repeat(mouth) vs Decision(hand). This is the #1 blocker.
2. **Check batch 44094932** (univariate contrasts) — should be finishing soon. Load results, verify all phases.
3. **Check batch 44111650** (phoneme decoding) — after completion, check if Decision STG/SMC now decode.

### 🟡 Soon (next few days)
4. **Univariate for other phases** — Once delay is validated, generate rose charts for stimulus/go/response to show timing specificity.
5. **Connectivity (xcorr)** — batch 44058208 should be done. Aggregate and visualize.
6. **Finalize figures** — Compile Fig 1-4 with captions.

### 🟢 Later
7. **Narrative draft** — Write the 4-section story with figure references.
8. **RSA / "fancy" analysis** — If PI insists, RSA on AIC representations across conditions. But temporal generalization already covers this.
9. **Sentence Repetition** — Low priority supplementary.

---

## Preprocessing Pipeline & Data Derivatives
*Technical reference for data flow from EDF to Machine Learning datasets.*

### 1. Epoching & Feature Extraction ([extract_ieeg_epochs.py](file:///cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/code/extract_ieeg_epochs.py))
- Reads continuous re-referenced EDFs (`/derivatives/bipolar`).
- Epochs by event phases (Stimulus, Delay, Go, Response) + Baseline (~-0.5s to 0s pre-cue).
- Extracts frequency bands (e.g., 70-200Hz High-Gamma via continuous wavelet), resamples to 128Hz.
- **Outputs (`/derivatives/epoch(bipolar)`)**:
  - `epoch(raw)`: Raw bipolar time-domain segments.
  - `epoch(band)(raw)`: Bandpass filtered only.
  - `epoch(band)(power)`: **Absolute energy (power)** sequences for both Task and Baseline.

### 2. Statistical Masking ([time_perm_bands.py](file:///cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/code/time_perm_bands.py))
- Runs time-permutation cluster testing: Task Power vs Baseline Power.
- Computes FDR-corrected p-values and binary significance masks for specific time windows.
- Converts absolute Power to Z-score using Baseline mean/std.
- **Outputs**:
  - `/derivatives/statistics`: Pure H5 statistical results (`mask`, `pvals`, `sig_ch_names`).
  - `/derivatives/epoch(bipolar)/epoch(band)(zscore)`: Normalized continuous amplitude traces (plotted as HGA).
  - `/derivatives/epoch(bipolar)/epoch(band)(sig)`: **Z-scored data ONLY for electrodes showing significant activation** in that window. 

### 3. Parcellation ([parcellation.py](file:///cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/code/parcellation.py))
- Maps electrode XYZ coordinates (from `electrodes.tsv`) to FreeSurfer atlas (`aparc.a2009s+aseg.mgz`).
- Uses majority voting inside a sphere (default 3mm/voxels).
- Handles bad labels (e.g., White-Matter) to ensure valid cortical assignment.
- **Output (`/derivatives/parcellation`)**: `*aparc2009s.csv` mapping electrodes to ROIs (e.g., AIC, STG).

### 4. Machine-Learning Preparation
**Step A: Clean Significant Datasets ([save_effective_sig_electrodes.py](file:///cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/code/save_effective_sig_electrodes.py))**
- Intersects significant channels (`epoch(band)(sig)`) with valid GM parcellation labels (drops White-Matter/Unknown).
- Embeds a JSON `roi_map` dictionary into the H5 file metadata.
- **Output**: `epoch(band)(sig)(effective)` — The purest, artifact-free significant electrode data ready for downstream ML.

**Step B: Final Assembly ([prepare_decoding_dataset.py](file:///cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/code/prepare_decoding_dataset.py))**
- Loads the [(effective)](file:///hpc/home/ns458/coganlab/nanlinshi/insula/src/direct_cross_decoder.py#275-299) datasets for specific, requested ROIs.
- Applies trial-level filtering (e.g., keeps only 'CORRECT' trials for lexicality; keeps top-4 phonemes for articulation).
- Flattens into [(Trials, Channels, Time)](file:///hpc/home/ns458/coganlab/nanlinshi/insula/src/direct_cross_decoder.py#275-299) matrix `X` and target array `y`.
- **Output (`/derivatives/decoding(bipolar)`)**: Final H5 datasets directly ingestible by standard ML classifiers.

---

## Figure Layout (Draft)

| Figure | Content | Status |
|--------|---------|--------|
| **Fig 1** | Electrode coverage + activation rates | ✅ |
| **Fig 2** | HGA traces (4 tasks) + onset latency + lateralization | ✅ |
| **Fig 3** | Action specificity: Repeat vs Passive + RT prediction + (Effector?) | ⚠️ Need P1 |
| **Fig 4** | Cognitive control: Decision>Repeat univariate + cross-decoding + lexical maintenance | 🔄 Partial |
