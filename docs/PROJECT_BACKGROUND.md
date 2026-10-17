# Project Background: Insula R01

This document captures the current manuscript framing and task battery for the Insula R01 analysis workspace. Scientific framing updated on 2026-10-14.

## Scientific Motivation

Lesion, stimulation, neuroimaging, and intracranial recording studies implicate the insula in speech, but the localization and interpretation of its contributions remain debated. Auditory responses, movement-related activity, and disputed accounts of articulatory planning establish the relevance of the insula while leaving its internal functional organization incompletely understood.

The manuscript centers on **the functional architecture of speech within the human insula**. Its motivating question is how speech-related functions are organized within this region. Sensory, sustained, and motor response profiles are findings to characterize, rather than three categories presupposed by that question.

The introduction should establish the insula's relevance to speech, explain how existing evidence reveals different aspects of its contribution, and identify the unresolved organizational question. High spatial and temporal resolution enables the study; task contrasts, representational analyses, and behavioral prediction give the functional distinctions their interpretation.

## Central Findings and Interpretation

The current results support spatial differentiation of speech-related functions within the insula:

- Sensory responses provide an auditory reference within the organization and are concentrated toward posterior insular territories.
- Sustained responses carry decodable articulator-category information during the interval between listening and speaking, supporting an interpretation involving retention of speech-related content.
- Motor-related responses contribute evidence linking insular activity to speech output, including prediction of trial reaction time.

Sustained representation and motor-related behavioral prediction are complementary parts of the paper; neither alone defines its scope. Functional groups may overlap spatially and respond in more than one task phase. Individual anatomy and functional profiles should be examined without equating a broad anterior/posterior partition with functional boundaries.

Use "spatial differentiation" or "functional organization" for the central framing. A continuous functional gradient requires quantitative evidence beyond a smoothed spatial map. Delay-period decoding supports the presence of category-related information without by itself establishing a pure motor code or distinguishing retention from rehearsal and preparation. Reaction-time prediction establishes behavioral relevance without demonstrating causal control of speech initiation. See the decoding and RT methods for the exact inference and validation scope.

## Core Methods

The project uses stereoelectroencephalography (sEEG) from epilepsy patients with insular coverage. This provides millimeter-scale spatial resolution and millisecond-scale temporal resolution.

Core manuscript analyses include:

- High-gamma activity (HGA).
- Multivariate encoding and decoding.
- Time-resolved reaction-time prediction.
- Spatial characterization and comparison of functional response profiles.

The broader workspace also supports:

- Micro-wire single-unit recordings.
- Cortico-cortical evoked potentials (CCEP).
- Directed functional connectivity.

## Manuscript Questions

### Functional Organization

Characterize insular response profiles and their spatial distribution, then test how input modality and behavioral demands modulate them.

- Separate responses associated with input, the delay interval, and output.
- Compare passive and active conditions, repetition and lexical decision, and auditory and visual inputs where the task design permits.
- Relate functional profiles to anatomy while accounting for individual variation and electrode coverage.

### Representational Content

Characterize what speech-related information the functional groups carry and when it is available.

- Decode articulator categories and lexical status across task phases.
- Test cross-condition generalization where matching data are available.
- Interpret sustained delay-period decoding alongside task modulation, without treating sustained activity alone as evidence for working memory.

### Behavioral Relevance and Network Context

Assess how activity relates to speech behavior and place the insular functional profiles in their broader cortical context.

- Predict trial reaction time from insular activity, including motor-group sites.
- Distinguish held-out prediction from post-selection descriptions of HGA amplitude and reaction time.
- Compare insular response profiles with responses elsewhere in the brain. Waveform similarity is not connectivity; CCEP and other network analyses have separate evidential roles when included.

## Task Battery

### Lexical Delay

Purpose: Tests lexical processing and speech production with an explicit delay between stimulus and response. The delay separates auditory input from vocal output, allowing retention- and preparation-related activity to be examined.

Trial flow:

```text
Cue (~2 s) -> Audio stimulus -> Delay (~1 s) -> Go "Speak" (~0.5 s) -> Vocal response (~1.5 s) -> ISI (~0.75 s)
```

Conditions:

- `Decision`: cue is "Yes/No"; subjects say "Yes" for words and "No" for non-words.
- `Repeat`: cue is "Repeat"; subjects repeat the stimulus aloud.

Typical parameters: 4 blocks x 84 trials, for 336 total trials. Stimuli include high-frequency words, low-frequency words, and non-words.

Supports task-context comparisons, delay-period representation, cross-condition decoding, and reaction-time analyses.

### Lexical No Delay

Purpose: Tests lexical processing without an explicit delay period. The decision condition uses button press rather than vocal response, enabling cleaner reaction-time measurement.

Trial flow:

```text
Cue (~2 s) -> Audio stimulus -> Immediate response -> ISI (~0.25 s)
```

Conditions:

- `Decision`: cue is "Yes/No"; subjects press left for yes and right for no.
- `Repeat`: cue is "Repeat"; subjects repeat the stimulus aloud.
- `Passive`: cue is ":=:"; no response.

Typical parameters: 4 blocks x 126 trials, for 504 total trials.

Supports comparisons of passive listening, repetition, and manual lexical decision. Inclusion in a manuscript panel depends on the analysis actually used there.

### Picture Naming

Purpose: Tests naming of common objects across modalities, enabling comparison of visual and auditory pathways converging on a common speech output. The text condition is excluded because of insufficient responses.

Trial flow:

```text
Color cue (1 s) -> Wait (0.5 s) -> Stimulus (1 s) -> Delay (1 s) -> Go "Speak" (0.5 s) -> Response (1.75 s) -> ITI (1.25 s)
```

Conditions:

- `Repeat`: green cue, vocal naming/repetition.
- `Passive`: red cue, no response.

Modalities:

- Picture.
- Sound.

Typical parameters: 5 blocks x 72 trials, for 360 total trials. Stimuli are apple, duck, star, and umbrella.

Supports input-modality and active/passive comparisons, as well as reaction-time analyses.

### Phoneme Sequencing

Purpose: Tests speech perception and production at the phoneme level using non-words, isolating phonological processing from lexical and semantic processing.

Trial flow:

```text
"Listen" + Audio -> Delay (~1 s) -> Go "Speak" (~0.5 s) -> Vocal repeat (~1.5 s) -> ISI (~0.75 s)
```

Conditions:

- `Repeat` only.

Typical parameters: 4 blocks x 52 trials, for 208 total trials. Stimuli include 26 CVC and 26 VCV non-words.

Supports articulator-category decoding across task phases and reaction-time analyses.

### Environment Sternberg

Purpose: Provides non-speech working-memory comparisons within the broader task battery; its inclusion in the current manuscript depends on the analyses presented.

Design notes:

- Stimuli include environmental sounds and shapes.
- Responses can be vocal or manual.
- The delay period is matched to Lexical Delay.
- Comparisons with speech tasks can examine the influence of content and response requirements.

## Analysis Mapping

- Functional organization: characterize response profiles and their anatomical distributions.
- Task context: compare input modality, passive/active conditions, and repetition/decision demands using the relevant within-task contrasts.
- Representation: decode articulator categories and lexicality, including delay-period content and cross-condition generalization.
- Behavior: predict reaction time and describe HGA–RT associations within their validation limits.
- Network context: compare extra-insular response profiles; use CCEP or other connectivity analyses only for the claims those methods support.
