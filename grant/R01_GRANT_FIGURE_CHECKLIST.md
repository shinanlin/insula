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
| **Status** | Needs creation |

### Figure C3-2 — AIC Electrode and Signal Characterization

| Item | Detail |
|------|--------|
| **Section** | C3 Approach |
| **What to plot** | Two panels: |
| | **Panel A:** Schematic illustration showing electrode insertion into the AIC |
| | **Panel B:** Time-Frequency Representation (TFR) of the electrode signal |
| **Key message** | Demonstrate the recording methodology and signal quality from AIC electrodes |
| **Status** | Needs creation |

---

## Aim 1: Domain-General Operations of the AIC During Speech

### Aim 1.1 — Domain-Generality of AIC Engagement

Multi-panel composite figure demonstrating AIC engagement is not speech-specific across three dimensions: content, input modality, and output effector.

#### Panel B: Input Modality

| Item | Detail |
|------|--------|
| **Sub-aim** | 1.1 — input modality manipulation (auditory vs. visual) |
| **Task** | Picture Naming (image vs. sound conditions) |
| **Data** | AIC electrode HGA |
| **What to plot** | Contrast between image and sound conditions (NOT PIC contrast). Should plot the spatial distribution and the temporal waveform. The spatial distribution should be the overlap between sound and image. |
| **Time window** | Stimulus window, extended by 0.5–1 second into delay period |
| **Regions** | AIC only; avoid including posterior auditory cortex responses |
| **Key message** | AIC activates regardless of input modality |
| **Status** | NEEDS REVISION: switch from PIC to image-vs-sound contrast; adjust time window |

#### Panel C: Output Effector

| Item | Detail |
|------|--------|
| **Sub-aim** | 1.1 — output effector manipulation (spoken vs. manual) |
| **Task** | Lexical No-Delay and no delay — both conditions are decisions (spoken vs. button-press lexical decision) |
| **Data** | AIC electrode HGA |
| **What to plot** | Spoken vs. manual response traces, response window only. Also, there should be a special plot about the distribution of significant channels and the time-per-trial waveform. |
| **Time window** | Response window only (avoid confounding with WM effects from delay) |
| **Design note** | Same task with two response types (not two different tasks) keeps comparison clean |
| **Key message** | AIC engages for both spoken and manual output |
| **Status** | NEEDS REVISION: restrict to response window; use lexical no-delay with both decision types |

#### Panel C: Spatial Overlap (Content Manipulation)

| Item | Detail |
|------|--------|
| **Sub-aim** | 1.1 — content manipulation (shapes vs. speech) |
| **Task** | Environment Sternberg and Lexical Delay |
| **Data** | AIC electrode HGA; environment vs. non-word conditions + overlap with production |
| **What to plot** | Two panels: (1) spatial overlap between environment and non-word within environment task; (2) overlap with production-related activity |
| **Traces** | Expand from words-only to words, non-words, and environment |
| **Regions** | AIC only; may include both hemispheres to increase channel count |
| **Key message** | AIC shows overlapping but not identical spatial activation for speech and non-speech — some variability expected |
| **Status** | NEEDS REVISION: add non-word and environment traces; add production overlap panel |

---

### Aim 1.2 — Dissociating Maintenance from Sensory/Execution

| Item | Detail |
|------|--------|
| **Sub-aim** | 1.2 — dissociate maintenance from sensory and execution patterns |
| **Tasks** | Three conditions from Lexical No-Delay and Lexical Delay: (1) passive listening, (2) repetition (no delay), (3) delayed repetition |
| **Data** | AIC electrode HGA; SMC as contrast region (NOT posterior cortex) |
| **What to plot** | Three-trace plot with two alignment panels. The comparison should be twofold: 1. The first layer of comparison should be the passive listening, the delayed repetition, and the no-delay repetition, in the stimulus time window. 2. The second fold should be the delayed repetition and the no-delay repetition in the response time window. |
| **Input-aligned** | Stimulus-locked: passive listening goes flat; delayed repetition shows sustained activity. No-delay repetition: cut off line after initial bump. |
| **Output-aligned** | Response-locked: passive listening flat; repetition and delayed repetition show similar pre-response ramps |
| **Stimuli** | Common lexical stimuli across no-delay and delay tasks |
| **Contrast region** | SMC (not posterior cortex) — avoids questions about what posterior regions are doing |
| **Key message** | AIC is silent during passive listening but shows sustained delay activity when maintenance required |
| **Status** | NEEDS REVISION: replace posterior with SMC; create three-condition traces; handle line cutoff for no-delay |

---

### Aim 1.3 — Cognitive Demand Modulation

| Item | Detail |
|------|--------|
| **Sub-aim** | 1.3 — AIC scales with cognitive demand |
| **Tasks** | Lexical Delay: Repetition and decision. |
| **Data** | Lexical Delay |
| **What to plot** | AIC activity comparison: higher engagement for more demanding task (lexical decision > repetition), showing only the insula electrodes |
| **Key message** | AIC activity scales with cognitive demand, consistent with domain-general control |
| **Status** | Show only insula data |

---

### (Optional) Reaction Time Figure

| Item | Detail |
|------|--------|
| **Placement** | Flexible — possibly Aim 1.1 (predictor of motor activity) or Aim 1.2 |
| **Data** | Lexical delay |
| **What to plot** | RT as predictor of AIC activity |
| **Status** | Data to be included; final placement TBD |

---

## Aim 2: Speech-Specific Representations in the AIC

### Aim 2.1 — Encoding/Decoding of Linguistic and Articulatory Content

| Item | Detail |
|------|--------|
| **Sub-aim** | 2.1 — encoding and decoding models |
| **Task** | Lexical Delay (repeat condition) |
| **Data** | AIC electrode activity |
| **What to plot** | Window decoding for articulatory features during perception and production. Time-resolved decoding for lexical status: lexical status (word vs. non-word), both repeat and decision. |
| **Lexical decoding** | Performance primarily in delay period — display separately |
| **Key revision** | REMOVE PIC from this figure |
| **Key message** | AIC carries both higher-level linguistic and articulatory motor content |
| **Status** | NEEDS REVISION: remove PIC; separate lexical decoding display for delay period |

---

### Aim 2.2 — Cross-Task Generalization of Representations

| Item | Detail |
|------|--------|
| **Sub-aim** | 2.2 — cross-task decoding |
| **Task** | Lexical Delay (decision condition) + cross-decoding between conditions |
| **Data** | AIC electrode activity |
| **What to plot** | (1) Within-task decoding for decision condition; (2) Cross-task decoding (train on repeat, test on decision, or vice versa) |
| **Key message** | AIC representations generalize abstractly across output goals |
| **Status** | Extend from Figure 2.1; add decision condition and cross-decoding panels |

---

### Aim 2.3 — Single-Neuron Evidence (Microwire)

| Item | Detail |
|------|--------|
| **Sub-aim** | 2.3 — microwire recordings |
| **Data** | SEEG electrodes with novel microwires in AIC |
| **What to plot** | Single-neuron examples showing co-existence of higher-order linguistic and articulatory information |
| **Key message** | Higher-order and articulatory information coexist within single AIC neurons |
| **Status** | AI-generated version added to directory; check .AI file |

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
| **Sub-aim** | 3.3 — cross-regional decoding |
| **Data** | AIC + cognitive regions + motor regions (Analysis TBD) |
| **What to plot** | Cross-regional decoding: AIC shares linguistic content with cognitive regions and motor-readiness signals with motor regions |
| **Key message** | AIC shares representational content with both upstream cognitive and downstream motor regions |
| **Status** | Check if current cross-correlation figure needs changes |

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

## Summary Status

| Figure | Aim | Status |
|--------|-----|--------|
| C3-1 — Electrode Coverage | Approach | Needs creation |
| C3-2 — AIC Electrode & TFR | Approach | Needs creation |
| 1.1B — Input Modality | 1.1 | Needs revision |
| 1.1C — Output Effector | 1.1 | Needs revision |
| 1.1C — Spatial Overlap | 1.1 | Needs revision |
| 1.2 — Maintenance (Space & Time) | 1.2 | Needs revision |
| 1.3 — Cognitive Demand | 1.3 | Minor revision |
| 2.1 — Encoding/Decoding | 2.1 | Needs revision (remove PIC) |
| 2.2 — Cross-Task Decoding | 2.2 | Needs creation/extension |
| 2.3 — Microwire Single-Neuron | 2.3 | Check .AI file |
| 3.1 — CCEPs | 3.1 | Waiting for data |
| 3.2 — Directed Connectivity | 3.2 | Needs major revision |
| 3.3 — Cross-Regional Decoding | 3.3 | Check/revise |
| Task Summary Table | All | Needs creation |
| Figure Captions | All | Pending |
