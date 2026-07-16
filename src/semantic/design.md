# Design: Lexical-semantic RSA / encoding in AIC

## Scientific framing

Project hypothesis: AIC performs domain-general operations over
speech-specific representations (lexical status, phonetic/articulatory
content). Semantic content is adjacent but was not the designed Aim 2
endpoint.

Lexical Delay can still ask a narrower question:

> After hearing a real word, does AIC maintain a representation whose
> geometry tracks meaning (beyond sound and lexical familiarity)?

If yes, AIC is not only a control hub or articulatory buffer; it also
holds lexical-semantic structure during the delay/decision window.

## What this task is *not*

- Not a semantic category experiment (no animal/tool design).
- Word vs Nonword ≠ semantic decoding.
- High/low frequency is mentioned in project docs but is not currently
  a decoding datatype; frequency enters as a **control regressor**.

## Stimulus set (Lexical Delay)

Approximate inventory (subject D0092 events; confirm when building table):

- ~42 Word tokens (e.g. baron, bison, cabin, modem, penal, …)
- ~41 Nonword tokens (matched length ≈ 5 letters)
- Conditions: `Yes_No` (Decision) and `Repeat`
- Phases: Stimulus, Delay, Go, Response (plus Cue)

Condition string grammar in epochs/events:

```text
{Cue|Auditory_stim|Delay|Go|Resp}/{Yes_No|Repeat}/{Word|Nonword}/{token}/{CORRECT|...}
```

Same parsing is already used in univariate contrasts and
`prepare_decoding_dataset.py`.

## Tier A — Lexical status baseline

Reuse existing tools; do not duplicate:

- `derivatives/decoding(bipolar)/sub-AIC{l,r}/lexicality/`
- `src/decoding/run_decoding_resolved.py --datatype lexicality`
- Univariate `WordVsNonword{Decision,Repeat}` in `src/univariate/contrasts.py`

Report for AIC (and STG control):

1. Time-resolved lexicality accuracy by phase.
2. Decision vs Repeat.
3. Whether Delay remains above chance.

Interpretation: lexical status / access, **not** semantic content.

## Tier B — Semantic RSA (main analysis)

### Inclusion

- Lexicality == Word only.
- Prefer CORRECT trials.
- Primary ROI: AIC (Hammers, `mix == False` as in decoding prep).
- Controls: STG (auditory), SMC (motor), PIC (posterior insula).

### Neural patterns

1. Load epoch HGA (or packaged ROI decoding matrices if item labels available).
2. Average trials within item (token) → item × channel × time (or item × feature).
3. Primary window: **Delay** full epoch; secondary: Stimulus late half.
4. Optional: PCA/channel selection to stable AIC contacts.

Neural RDM: pairwise dissimilarity among items
(1 − Pearson/Spearman correlation, or Euclidean on z-scored patterns).

### Model RDMs

| RDM | Source | Role |
|-----|--------|------|
| Semantic | GloVe / fastText / WordNet path | Target |
| Phonological | Phoneme-seq edit distance or articulatory features (g2p) | Control |
| Orthographic | Levenshtein on spellings | Control |
| Frequency | Absolute difference in log SUBTLEX (or similar) | Control |
| Length | Absolute letter/phoneme length difference | Control |

Embeddings may live under group cache paths (avoid `$HOME`), e.g.
`/hpc/group/coganlab/nanlinshi/embedding/` or project `results/semantic/`.

### Statistics

1. Spearman ρ between upper triangles of neural RDM and semantic RDM.
2. Item-label permutation null (or Mantel).
3. Partial / residual RSA: regress phonology (+ frequency, length) out of
   semantic and/or neural RDM vectors, then correlate.
4. Split by description: Decision vs Repeat.
5. Optional: leave-one-item-out consistency.

### Positive result criteria (pre-register in notes)

All preferred:

1. Semantic–neural ρ significant in AIC Delay.
2. Partial ρ survives phonology + frequency.
3. Decision ≥ Repeat (or Decision significant, Repeat not).
4. Pattern distinct from STG early Stimulus phonology-dominated RSA.

Claim language if met:

> AIC Delay geometry tracks lexical-semantic structure among real words
> beyond phonological and frequency similarity.

## Tier C — Continuous semantic encoding

Align with `src/encoding/` style:

- Predictors: semantic PC1..k from word embeddings; optional concreteness.
- Nuisance: phoneme/articulator indicators, log frequency, acoustic envelope if available.
- Target: AIC channel HGA over time; cluster / FDR across channels×time.
- Prefer trial-level models with item-aware CV (group by token).

## Tier D — Generalization (later)

1. Cross-condition: train Decision RSA/encoding → test Repeat (or decoder
   transfer for item neighborhood structure).
2. Cross-task only if shared concepts exist (Lexical Delay ∩ Picture Naming
   is essentially empty; do not force).
3. Environment Sternberg: domain-general maintenance contrast, not verbal
   semantics.

## Module map

| File | Role |
|------|------|
| `features.py` | Stimulus table: token, lexicality, freq, phonemes, embeddings |
| `rsa.py` | RDM construction, Spearman/partial RSA, permutation |
| `run_rsa.py` | CLI entry for Tier B (SLURM target) |
| `encoding_semantic.py` | Tier C runner (add when Tier B works) |
| `README.md` | Package overview |
| `design.md` | This document |

## Outputs

Suggested under results (gitignored):

```text
results/semantic/LexicalDelay/
  stimulus_table.csv
  rdm_semantic.npy
  rdm_phonology.npy
  rsa_AIC_Delay_Decision.csv
  rsa_partial_*.csv
```

## Non-goals (v1)

- Multi-class semantic category decoding.
- Calling lexicality plots “semantic”.
- Heavy new preprocessing; consume existing bipolar HGA / decoding derivatives.
- Writing large embedding files into git.

## Implementation notes

- Use `rootutils.setup_root(..., indicator=".project-root")`.
- Prefer `from src.semantic...` imports.
- ROI naming: `AICl` / `AICr` as `--subject` style elsewhere.
- Atlas default: `hammers`.
- conda env: `ieeg`.
