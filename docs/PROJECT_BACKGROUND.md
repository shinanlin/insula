# Project Background: Insula R01

This document captures the scientific framing and task battery for the Insula R01 analysis workspace.

## Scientific Motivation

The anterior insular cortex (AIC) has been linked to speech production for more than a century, but its role remains unresolved. Lesion and stroke studies associate AIC damage with selective speech deficits, including apraxia of speech, transient mutism, and speech production impairment in primary progressive aphasia. Neuroimaging studies also consistently report AIC activation during speech tasks.

These findings have motivated speech-specific accounts in which the AIC acts as an articulatory planner or verbal working memory hub. At the same time, the AIC is also active across many non-speech tasks as part of cognitive control and salience networks. This has motivated domain-general accounts in which AIC activity reflects general control demands rather than speech-specific content.

The project is built around the tension between these views. Speech-specific accounts do not explain why AIC engages broadly across non-linguistic cognitive demands. Domain-general accounts do not explain why focal AIC damage can cause selective speech-related deficits.

## Overarching Hypothesis

The AIC performs domain-general cognitive operations over speech-specific neural representations.

In this view, AIC is a cognitive-motor interface:

- Domain-general operations include maintenance and goal/action-directed control.
- Speech-specific representations include lexical status, phonetic content, and articulator identity.
- AIC engagement reflects domain-general computations acting on speech-relevant content, rather than a purely speech-specific or purely domain-general function.

## Core Methods

The project uses stereoelectroencephalography (sEEG) from epilepsy patients with insular coverage. This provides millimeter-scale spatial resolution and millisecond-scale temporal resolution.

Primary analysis modalities include:

- High-gamma activity (HGA).
- Multivariate encoding and decoding.
- Micro-wire single-unit recordings.
- Cortico-cortical evoked potentials (CCEP).
- Directed functional connectivity.

## Specific Aims

### Aim 1: Domain-General Operations

Characterize when and why AIC engages during speech.

- Aim 1.1 tests domain-generality by manipulating content, input modality, and output effector.
- Aim 1.2 dissociates maintenance from sensory and execution-related activity in space and time.
- Aim 1.3 tests cognitive demand scaling by comparing delayed repetition with delayed spoken lexical decision.

### Aim 2: Speech-Specific Representations

Characterize what speech content AIC encodes.

- Aim 2.1 tests encoding and decoding of lexical status and phonetic/articulatory content.
- Aim 2.2 tests cross-task decoding to determine whether representations generalize abstractly across output goals.
- Aim 2.3 tests whether higher-order and articulatory information coexist within single neurons using micro-wire recordings.

### Aim 3: Cognitive-Motor Interface

Establish where AIC sits in the speech network.

- Aim 3.1 maps effective connectivity using CCEP.
- Aim 3.2 tests directed functional connectivity from cognitive regions through AIC to motor regions.
- Aim 3.3 uses cross-regional multivariate decoding to characterize information flow.

## Task Battery

### Lexical Delay

Purpose: Tests lexical processing and speech production with an explicit delay between stimulus and response. The delay period engages working memory and separates perceptual processing from motor planning.

Trial flow:

```text
Cue (~2 s) -> Audio stimulus -> Delay (~1 s) -> Go "Speak" (~0.5 s) -> Vocal response (~1.5 s) -> ISI (~0.75 s)
```

Conditions:

- `Decision`: cue is "Yes/No"; subjects say "Yes" for words and "No" for non-words.
- `Repeat`: cue is "Repeat"; subjects repeat the stimulus aloud.

Typical parameters: 4 blocks x 84 trials, for 336 total trials. Stimuli include high-frequency words, low-frequency words, and non-words.

Used in Aim 1.2, Aim 1.3, Aim 2.1, and Aim 2.2.

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

Used in Aim 1.1 and Aim 1.2.

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

Used in Aim 1.1 for modality manipulation.

### Phoneme Sequencing

Purpose: Tests speech perception and production at the phoneme level using non-words, isolating phonological processing from lexical and semantic processing.

Trial flow:

```text
"Listen" + Audio -> Delay (~1 s) -> Go "Speak" (~0.5 s) -> Vocal repeat (~1.5 s) -> ISI (~0.75 s)
```

Conditions:

- `Repeat` only.

Typical parameters: 4 blocks x 52 trials, for 208 total trials. Stimuli include 26 CVC and 26 VCV non-words.

Used in Aim 2.1 for articulatory decoding.

### Environment Sternberg

Purpose: Tests domain-generality of AIC maintenance using non-speech content in a Sternberg working memory structure matched to the speech tasks.

Design notes:

- Stimuli include environmental sounds and shapes.
- Responses can be vocal or manual.
- The delay period is matched to Lexical Delay.
- This task provides the cleanest contrast with speech content for Aim 1.1.

## Analysis Mapping

- Domain-generality analyses compare speech and non-speech content, auditory and visual inputs, and vocal and manual output demands.
- Maintenance analyses compare passive listening, repetition, and delayed repetition.
- Cognitive-demand analyses compare delayed repetition with delayed spoken lexical decision.
- Speech-representation analyses decode lexicality, phonetic content, token identity, and articulator identity.
- Network analyses combine CCEP, directed connectivity, cross-correlation, and cross-regional decoding.
