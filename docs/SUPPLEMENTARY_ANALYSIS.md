# Picture Naming source and participant scope clarification (2026-10-23)

Box has 44 subject directories, 39 subjects with nonempty raw EEG and 38 with clean EEG in that specific backup. The current Figure 1 whole-brain Repeat+sound coordinate records and matching SI participant Venn include **37** PN subjects; the two raw-data subjects outside that current package are D0121 (author-approved noise exclusion) and D0149 (not in the historical package). The frozen accepted insula list has 28 participants; the current eligible paired native-statistics sample has 27 after original endpoint QC. These are distinct scopes and must not be described as “44 analyzed, only 27 with insula.”

D0086/D0107 both have five raw and five clean EDFs in Box and historical mean-HGA/coordinate exports in the current paper. Their insula candidates remain pending because of cross-task native-coordinate conflicts; their local canonical BIDS currently lacks the complete Repeat single-trial input set. D0149 already has raw/clean/bipolar/delay EDFs and 24 HGA/24 statistics products in DCC, although its Box clean backup is absent. Repeat sound/image six phase products were checked for matching epoch/statistics dimensions. Its current local parcellation is aparc2009s, and it remains outside the paper's frozen Hammers/tissue/identity eligibility audit. Do not infer missing local preprocessing from the Box backup. Read-only evidence: `results/audit/picture_naming_subject_layers_2026-10-23/`; no inclusion rule, cohort, main figure or NMF has changed.

# Latest Figure 3 denominator correction audit (2026-10-23)

The author requested ALL frozen anatomically accepted task samples with complete native paired statistics, without NMF inclusion or any prior response gate. The completed read-only audit is `results/audit/fig3_all_anatomical_denominators_2026-10-23/`, including an Excel workbook, every site/subject, original QC provenance and a Chinese review note.

- Picture Naming Stimulus [0,1] s: 162 anatomical sites / 28 participants; 34 sites excluded by existing original native endpoint QC; 128 eligible paired sites / 27 participants. Existing NMF-restricted plots cover only 81 / 24. Full-cohort native mask counts are Sound 38, Image 24, common 13. All four approved native Stimulus repairs take precedence.
- Lexical Delay Delay [0,1] s: 214 anatomical sites / 38 participants; 27 sites excluded by original endpoint QC; 187 eligible paired sites / 35 participants. Existing plots cover only 123 / 34. Full-cohort counts are Decision 67, Repeat 59, common 48. The additional 64 sites are 12 NMF sites previously excluded by a response gate and 52 non-NMF sites.
- The four absent restore records were verified against every original clean-run QC TSV and historical eligibility: PN D0139, LD D0054/D0059/D0081 have all sampled sites removed by existing noisy/muscle endpoint QC. All 61 unavailable sites have original QC explanations; no unresolved non-QC statistical gaps remain.
- Existing Decision > Repeat positives (29) cover only the old 123-site tested universe; the additional 64 sites remain untested, not negative. No new test or redraw has been performed. The earlier figures and their documented 81/123 denominators below are preserved historical/subset outputs pending the author's review of this audit.

# Current Figure 3 condition-set maps (author request, 2026-10-23)

The current new condition-colored display is in `vizpub/supplementary/fig3_condition_response_maps.ipynb`. Picture Naming uses Stimulus **0–1 s** in the frozen paired 81-site / 24-participant cohort, reads complete original baseline masks with the approved native Stimulus repair taking precedence, and shows Sound-only, Image-only and Both in blue, orange and purple. It is not an expansion of the main sound selection or NMF window. Prior 0–0.6-s figures below remain historical and are not silently overwritten.

Lexical Delay uses Delay **0–1 s**, the original 123-site / 34-participant cohort that actually received the existing Decision > Repeat test. Its three overlapping Venn sets and corresponding maps are Decision relative to baseline, Repeat relative to baseline, and the existing 29-site Decision > Repeat result. The 12 extra native-good pre-gate sites have no condition-difference result and are not coded as tested negatives. The three maps retain overlapping membership rather than forcing difference sites into an exclusive baseline class. Condition colors replace component colors in these new maps; no NMF classes appear.

SVGs: `img/supplementary/fig3_condition_response_maps/`. Exact memberships, source hashes and condition/participant counts: `results/supplementary/fig3_condition_response_maps_2026-10-23/`. No tests are added, existing difference masks/statistics and main figures are preserved. Display geometry follows Figure 1 insula-only pial snap with its 4-mm visibility lift; contours use the original Figure 2 density helper and are not significance boundaries.

# Figure 3 complete response characterization (2026-10-23)

The audit also identifies the upstream complete Lexical Delay native-good pool: 135 sites / 35 participants. Its 12 sites excluded by the old any-phase sound-response gate are confirmed from their native Delay masks to be Neither-detected. A separate 135-site Venn/brain sheet and workbook front sheet describe this full pool; the original 123-site analyses, tests, HGA and subset panels remain unchanged. Sound–Image remains 81 sites / 24 participants.

`vizpub/supplementary/fig3_response_characterization.ipynb` audits the existing Sound–Image Stimulus [0,.6] and Decision–Repeat Delay [0,1] cohorts, original native baseline-mask classifications, complete per-component/participant counts, NMF-to-comparison denominator flow and assembly version consistency. Its Venns and four-class insula maps are descriptive and use one shared membership table. Participant counts overlap between cells and are not additive. Results and an Excel workbook live under `results/supplementary/fig3_response_characterization_2026-10-23/`; current subset panels are copied into `review_bundle/`. Main figures, NMF, native tests and existing HGA/heatmaps are preserved. The supplied assembly screenshot mixes 18 Picture-responsive heatmap rows with an 11-common-site map; newer editable assemblies require their actual source file to audit.

# Supplementary analyses

## Sound–Image common responders (author approved, 2026-10-21)

The active lower Picture heatmaps and BrainPlot both use the same 11 common
Stimulus responders / 9 participants: 2 Sensory, 6 Sustained, 3 Motor. Each site
has an existing baseline-mask hit in Sound and in Image within Stimulus [0,.6] s;
the significant time bins need not coincide. No new test is run.

Upper means/SEM retain the full 81-site cohort. The original Venn stays at
Sound 33, Picture 18, common 11. Preserve the corrected native input version,
all plotting conventions, current contour helpers and tightened spacing.
Lists and manifests are under `sound_image_test/common_stimulus_response/`;
the active brain SVG is `fig3_sound_image_common_stimulus_brain.svg`.
The prior 18-site Picture-only heatmap/map version is archived under
`sound_image_test/history/picture_only_before_common_2026-10-21/`.
Earlier 18-site descriptions below are historical for this selection.

## Selected-site BrainPlot contours (author requested, 2026-10-21)

The two standalone maps now show their existing colored selected points with
the original Figure 2 KDE outline/fill, omitting all nonsignificant gray points.
Decision/Repeat still maps the 29 Delay Decision > Repeat sites; Sound/Image
still maps all 18 Picture Stimulus baseline responders, including the 11 shared
Sound/Picture responders and seven Picture-only responders.

Reuse the current Figure 1 insula surface/snap/camera and exact Figure 2 contour
helper bodies and constants. Contours describe spatial density and carry no new
significance inference. A component/hemisphere with fewer than three sites has
points only, as in the original helper; small single-site contour islands are
discarded by that same source rule. Display audits preserve this distinction.
Only the brain SVG and its notebook output are refreshed; all HGA, heatmap,
Venn, statistical and cohort files remain protected.

## PN Stimulus event repair (author approved, 2026-10-21)

The author approved the audited single-line change from substring phase matching
to exact matching of the first slash-delimited event token. This excludes
StimulusOffset from Stimulus without changing baseline events, reference, bad-channel
QC, time windows, gamma extraction, normalization, time-permutation parameters or seeds.
Regenerate only Stimulus for D0076, D0079, D0094 and D0108, in both conditions and
all three modalities, under `results/native_stimulus_event_repair_2026-10-21/`.
Validate the retained event samples, channels, task power and baseline against the
corresponding original data; retain all original files.

Refresh the standalone Sound–Image notebook from its own repaired cache, replacing
only Repeat Stimulus inputs of the three affected participants already in its
frozen 81-site cohort. Preserve other phases and participants, shared context cache,
main figures, NMF and earlier condition-difference statistics. The Picture baseline
selection rule and all plotting conventions remain the same. Counts such as the
18 sites below describe the archived pre-repair state; the corrected counts are
read from the new membership manifest. No Sound–Image difference test is added.

Completed repair: all 24 products passed event/channel/power/baseline checks;
retained power and baseline differ by exactly zero from the original matching
data. D0079_LPI8-9 Picture peak increased from .235823 to .354109 but remains
outside the original temporal mask. The selected list stays at 18 sites / 13
participants (3 Sensory, 8 Sustained, 7 Motor), with unchanged Venn membership.
The standalone notebook and all three SVGs have been refreshed; original
statistics and the pre-repair version remain archived. See the repair's
`修复结果说明.txt` and the standalone `event_repair_2026-10-21/` manifest.

## Active Sound–Image question: Picture-responsive sites (author correction, 2026-10-21)

Use the existing Picture baseline mask within Stimulus [0,.6] s in the
current 81-site sound-discovery NMF cohort. There are 18 sites / 13 subjects:
Sensory 3, Sustained 8, Motor 7. Eleven also have a Sound baseline hit in
this window, and seven have only a Picture baseline hit. Do not require
Sound significance in the same phase and do not select Sound > Image.

Upper Sound/Image curves retain all 81 electrodes with electrode mean ± SEM;
remove the newly introduced between-condition bars. Lower three-phase
heatmaps show Picture activity of the selected 18 sites, with their existing
baseline masks and original paired scales. The brain map uses this exact
same list. Venn continues to describe baseline overlap in the full cohort.
No new significance test is run. NMF and main figures stay unchanged.

Current lists and manifests are under `sound_image_test/picture_stimulus_response/`.
The earlier difference-based draft is superseded and archived under
`sound_image_test/history/sound_gt_image_superseded_2026-10-21/`.

## Sound–Image standalone extension (2026-10-21)

`sound_image_test.ipynb` follows the reviewed Decision/Repeat structure for
Picture Naming Repeat trials. The same 81 frozen NMF sites / 24 participants
are available for sound and image in all three phases. Original native PN
trial QC is retained; these ListenSpeak events have no CORRECT accuracy
token, so the Lexical Delay event parser is not imposed.

Select sites using Stimulus [0,.6] s, the original default-t, independent,
two-sided 2000-draw `time_perm_cluster`, .05 forming/final thresholds and
positive mean differences within significant temporal bins. Reuse only the
two previously reviewed process-local NumPy sort-keyword substitutions;
installed iEEG code is unchanged. Temporal correction is within each site.

Upper sound/image HGA curves retain all 81 sites, electrode mean ± SEM,
purple/black contrast colors and the reviewed subject-block paired
5000-draw one-sided temporal test within each displayed phase. Use the
original Sound–Image automatic shared-y helper. Lower Sound-only heatmaps
show selected sites by frozen component across Stimulus [-.5,.6], Delay
[0,1], Response [-.5,1] s. Original baseline blanking, shared order, paired
P95 scales and time-proportional widths are retained, with 1-mm rows.

Venn regions describe saved baseline responses in Stimulus [0,.6] s for
the full cohort. The brain map instead shows the directly tested positive
difference subset using canonical Figure 1 local-insula helpers and
24/10-pixel selected/context points. Display projection is audited separately
from unchanged analytic coordinates. HGA/heatmaps, Venn and brain maps are
separate SVGs. Repeat–Passive is not run in this extension.

## Sound / Image complete-cohort difference display (2026-10-28)

The author explicitly selected the existing 24 Sound > Image sites from the
completed native Stimulus [0,1] s comparison, rather than the 13 common baseline
responders. The current standalone `sound_image_test.ipynb` shows all 128 paired
anatomical sites / 27 participants above and these 24 sites / 13 participants
below (7 Sensory, 7 Sustained, 8 Motor, 2 Unassigned). All 768 native input cells
passed a separate audit; original QC and exact-Stimulus repairs are retained.
Lower values are Sound minus Image, with the Sound native baseline display mask,
original component colors, 1-mm rows and fixed 0–0.5 limits. Negative numeric
differences are retained but display white. The original HGA display windows
remain [-.5,.6], [0,1], [-.5,1]; the selection window remains [0,1] s.
Pooled bars use the same approved 5000-draw subject-block time_perm_cluster
method as Decision/Repeat. Existing per-electrode tests and all other figures
remain unchanged. The old 81-site/common-11 notebook and SVG are archived;
current outputs are `sound_image_test/difference_heatmap_full_cohort_2026-10-28/`.

## Decision / Repeat standalone layout test (2026-10-21)

### Current HGA-difference display (2026-10-27)

**Later author correction:** the active trace–heatmap notebook now uses the
completed full Lexical cohort (187 sites / 35 participants) and its saved 31
Decision > Repeat sites (3 Sensory, 14 Sustained, 12 Motor, 2 Unassigned).
The 123/29 version described below is archived. Upper mean ± SEM and pooled
subject-block time-permutation bars use the full cohort, retaining the approved
5000 permutations and .05/.05 settings; old-cohort bars are not transplanted.
Per-electrode tests and the 31-site list are reused unchanged. All heatmaps now
use fixed 0–0.5 color limits, white for negative differences and original
component hues for positive values; raw signed differences are preserved.
The canonical input is `fig3_lexical_delay_condition_sets/full_cohort_2026-10-27/`;
new outputs are `decision_repeat_test/difference_heatmap_full_cohort_2026-10-27/`.
The standalone notebook's old cells were archived before replacing its active
entry, eliminating the stale 123/29 data path. Current Delay display is [0,1] s.

The author requested lower heatmaps of Decision minus Repeat, retaining each
component's color family. The existing standalone layout keeps its 123-site
upper electrode-mean ± SEM curves and saved significance bars, and its 29
Delay-selected lower rows (3 Sensory / 14 Sustained / 12 Motor). Match each
electrode/time sample and subtract the unmasked native condition means before
retaining the original Decision-baseline display mask. Missing pairs stay blank;
this blanking is not a new test of the difference. Positive differences use the
component's blue/gold/red; negative differences use gray, with zero white and a
common symmetric scale based on P95 absolute displayed differences (floor .15 z).
The original geometry, windows, 1-mm rows and row order are preserved. Original
Decision-only arrays and all statistical/trace files remain unchanged; new
difference matrices and display validation are in
`results/supplementary/decision_repeat_test/difference_heatmap_2026-10-27/`.
This display does not expand the historical trace/heatmap subset to the separate
187-site Lexical characterization or recompute its 31-site difference selection.

Pooled condition-difference bars were separately approved after input review:
123 sites / 34 subjects, paired Decision/Repeat, coherent condition swaps
within subject, installed `time_perm_cluster` with mean-difference statistic,
5000 draws, seed 42, one-sided, p_thresh=p_cluster=.05. Subject contributions
are S/N times their electrode sum, retaining the plotted electrode mean.
Each displayed phase has its own temporal correction family. Purple bars
above the HGA axes use only this saved condition-difference mask. Baseline
heatmap masks and prior per-electrode/P75 contrasts are separate evidence.
The installed iEEG library and its original temporal extent algorithm are
retained. Saved inputs, approval, raw pointwise p values, masks and intervals
are available within the standalone test's results directory.

Later author display correction: Repeat black, Decision purple (#6A5ACD),
pooled HGA ylim 0–0.4. Use the exact canonical blue/gold/red component
palette for the Decision heatmaps, removing condition-based darkening.
Reuse measured panel widths, trace height, heatmap row density, margins
and gaps from the original Figure 3 renderer. The one-curve-row,
Decision-only layout retains those individual panel proportions.

`decision_repeat_test.ipynb` is an isolated Figure 3 draft. Its initial layout
places two pooled electrode-mean ± SEM curves (Decision and Repeat) above
three Decision-only component heatmap rows, across the original three HGA
phase windows. The user explicitly confirmed pooling electrodes for the two top curves;
only the lower heatmaps are split by component. This draft is not a promotion
to main Figure 3.

Reuse the existing Lexical Delay sound comparison cache, task scope and
frozen NMF. Each bipolar site has equal weight in the pooled curves. Keep
the original shared electrode order, paired-condition P95 scales and blank
mask-false bins. The one-condition matrices and scales are checked against
the original Figure 3 renderer. Phase widths follow elapsed time and the
heatmap rows use a constant physical height per site. The original site selection is retained in the pooled curves. The additional
Delay electrode-selection test below is separately approved; main notebooks stay frozen.

### Local-insula marker-size correction (2026-10-21)

The author requested larger electrode markers. This map uses 24-pixel
selected points (previously 12) and 10-pixel context points (previously 7).
Canonical Figure 1 helper bodies, mesh, camera and projected coordinates
are unchanged. No electrode selection or statistical result changes.

### Local-insula renderer correction (2026-10-21)

The author corrected the spatial display to match other local insula panels.
The Delay difference map now reuses exact helper bodies from Figure 1 cell
7731cdbd: hidden full-hemisphere actors, an insula-only pial mesh, the original
camera-facing insula vertex snap and 4-mm display lift, and Figure 1 marker
sizes. The old decoding renderer could leave markers obscured by translucent
surface layers and did not restrict snapping to the drawn insula.
Analytic coordinates and memberships stay unchanged; display coordinates
and source-function hashes are exported under delay_overlap_maps/local_insula_display/.
Execute only --section delay_brain_display; preserve Venn, HGA and statistics.

### Delay Venn and condition-difference brain maps (2026-10-21)

The added section in decision_repeat_test.ipynb uses the same 123 sites and
Delay [0,1] s. Venn sets are Decision and Repeat saved native baseline mask
hits in this window. Export all four exclusive regions, including neither,
and all site memberships. A condition-only baseline hit is not a direct
condition-selectivity test. The spatial panels instead show the existing
29 significant Decision > Repeat sites, colored by their frozen NMF group,
with the other 94 sites gray. Reuse the existing template surface renderer,
projection, camera, point sizes and canonical palette. No inference or
selection is rerun. The --section delay_overlap_maps execution path runs
only the new cells and setup, preserving old cells, HGA SVG and all results.

### Heatmap row-height display update (2026-10-21)

The author requested readable lower heatmaps after the 29-site selection.
Use 0.10 cm (1 mm) per electrode identically in all phases and components,
superseding original physical row density for this standalone draft only.
Keep phase widths, upper curve geometry, palettes, scales, values, masks and
row membership unchanged. Load the saved approved condition-test results;
default RECOMPUTE_POOLED_TEST=False prevents inference from rerunning.
The upper curves and all statistical/selection files are hash-protected.

### Delay-selected Decision heatmaps (approved 2026-10-21)

Test all 123 existing sites separately using current native sound-only
Lexical Delay trials and the exact original contrast script CORRECT-only
event rule. Use Delay [0,1] s, the original iEEG time_perm_cluster default
t statistic, independent trial-label permutations, 2000 draws, two-sided,
p_thresh=p_cluster=.05. Clusters extend only in time within a site; this
is temporal correction per electrode, without an added across-electrode
correction family. Freeze seed 42 for reproducibility.

Keep positive Decision-minus-Repeat bins inside the saved temporal-cluster
mask and the original p<.05 hit rule. Determine direction at the significant
bins rather than from the whole-phase average. The selected list is shared
across all three Decision heatmap phase columns, retaining the reviewed
row order, baseline mask blanking, scales, palettes and physical row density.
The upper pooled curves, 123-site cohort and saved group bars stay unchanged.

The original two-sided code cannot execute its internal argsort(stable=False)
on NumPy 1.26.4. The user separately approved the exact two substitutions
to kind='quicksort' in a process-local copy of _comp_by_sort. No installed
source is edited and the original rank and cluster algorithms stay intact.
Inputs, approval, all tested/selected electrodes, intervals and counts are
saved under decision_repeat_test/delay_electrode_selection/.

## Later anatomy review and linked HGA (2026-10-18)

The user confirmed the original Figure 1 physical-contact SEEG schematic.
All 15 companion anatomy panels now use physical gray shanks and cylindrical
contacts at the unchanged reconstructed centers, smaller MRI markers, and
upright native sagittal/coronal MRI. The static MRI illustration explicitly
labels the original Figure 1 +/-4-mm projection convention. This supersedes
the previous 1-mm oblique-plane display, without changing contact coordinates,
the existing three review flags, control pairs, or signal results.

`viewer/operculum_inspector/` builds an independent native MRI/contact/HGA
inspector. It contains all 134 real contacts on the example shafts across six
participants, and the recorded bipolar pairs on those shafts regardless of
atlas class. All pair endpoints/midpoints are checked against the matching
LEPTO reconstruction. Click a real contact for its native orthogonal slices,
or a bipolar midpoint/menu entry for its two contacts and three-phase HGA.
MRI shows actual layers, with the same half-voxel rounding as the original
anatomical code. Atlas reference is optional and off by default.

HGA is Repeat+sound, single-electrode trial mean over existing native zscore
epochs. Native QC stays in force. The saved unchanged orthogonalization branch
is a documented fallback for an existing analyzed example if needed. No new
significance, pair selection, orthogonalization or main-cohort calculation is
introduced. Task names, trial counts, missing channels and source files are
recorded. Manual review exports are separate from analysis metadata.

The portable offline viewer is under `results/supplementary/operculum/viewer/`;
the same verified data have a web build that loads one participant at a time.

## Paired Delay P75

`vizpub/supplementary/paired_delay_p75.ipynb` contains the original Figure 3
paired-P75 loading, matching, Wilcoxon and visualization code. It generates
`img/supplementary/paired_delay_p75/supp_paired_delay_p75.svg` and complete
electrode/subject/statistics tables. It uses the current frozen NMF labels
and reproduces current Figure 3's 94 PN Repeat–Passive pairs and 137 LD
Decision–Repeat pairs. Saved original electrode permutation/FDR results and
the original Delay 0–0.5-s per-trial percentile summary are retained. The
group test, asymmetric highlight rules, palette and layout are unchanged.

## Separate execution

Each topic has its own notebook in `vizpub/supplementary/`. Figure exports
are SVG files in `img/supplementary/<topic>/`; source lists, results and
execution receipts are in `results/supplementary/<topic>/`. Main Figure 1–6
notebooks and NMF are protected during supplementary execution.

## Figure 3 condition comparisons (updated with author approval 2026-10-21)

`vizpub/supplementary/context_comparisons.ipynb` contains verbatim copies of
the original Figure 3 multiphase plotting functions. It displays separate
Sound–Image, Decision–Repeat and Repeat–Passive figures. On 2026-10-21 the
author requested that each comparison be saved and shown as two SVGs, with
no programmatic assembly: `supp_<key>_waveforms_heatmaps.svg` and
`supp_<key>_brain_maps.svg`. The HGA panels
retain component rows and Stimulus/Delay/Response columns, each with the
original electrode mean ± SEM and paired masked heatmap. Current frozen
three-phase NMF assignments are reused. Original phase display limits,
condition palettes, electrode ordering and P95 heatmap scales are retained.

The saved HGA SVG is 13 cm wide. Electrode rows in the heatmaps are drawn at
0.032 cm, about twice the previous saved row; the earlier 8 cm height cap is
not applied, because it squashed those rows. Phase-width ratios, colors,
scales, masks and row order stay in place. Column spacing uses gridspec
wspace 0.26.
The brain maps are one horizontal row of four panels: left and right insula
for the first condition, then left and right for the second. Each unilateral
insula image is 3.5 cm wide. Each hemisphere
follows the saved Figure 1 panel: insula-only mesh, electrodes at their own
template coordinates, the same crop, and Hammers gyrus boundaries. Marker
color remains the frozen NMF component for a baseline-responsive site. On 2026-10-21 the
author approved replacing the three-phase union with **one mapped phase per
comparison**: Stimulus for Sound–Image, Delay for Decision–Repeat, and Stimulus
for Repeat–Passive. The Repeat–Passive map was moved from Delay to the displayed Stimulus window on 2026-10-21, using the existing baseline masks. Use the exact displayed window of that phase, label its
bounds, and label the corresponding HGA column. The author subsequently
requested removal of the added phase-column borders; preserve the labels
without enclosing rectangles. Colors are current NMF
components. Maps and heatmaps use the same electrode identifiers;
these maps are not tests of condition differences. Only baseline-responsive
sites are drawn; nonsignificant candidates are omitted from the scatter.
No new permutation test or multiple-comparison family is introduced.
The prior all-three-phase-union version is preserved under the topic's
`history/three_phase_union_2026-10-20/` directory.

Native waveforms/masks use the validated restored inputs. Passive Response
reuses its original stored nominal alignment and statistics, restricted to
the current good task electrode set; it has no actual vocal onset. This
legacy branch, source files, missing data, per-phase membership and copied
function hashes are recorded under
`results/supplementary/context_comparisons/`. Outputs are under the matching
`img/supplementary/context_comparisons/` directory. Main notebooks, NMF and
their images are not regenerated by this notebook.

## Task sampling

`task_sampling.ipynb` displays the frozen Figure 1 Repeat + sound cohort:
348 unique sampled sites, 231 responsive sites, 65 sampled participants.
It includes task coverage, task totals and complete subject × task CSVs.
The redundant subject × task matrix figure has been removed at the user's
request; overlap is displayed in the Venn diagrams.
Missing response evidence remains missing. Ongoing native preprocessing
restoration does not silently redefine this cohort.

Immediately below task-insula coverage, four-set Venn diagrams show
participants, all recorded bipolar subject/site identifiers, and accepted
sampled insula sites. All 15 exclusive regions (including zeros) are shown.
Counts and full membership lists are exported; areas are not proportional
to set sizes. The participant and whole-brain sets use Figure 1's complete
Repeat+sound coordinate records after D0121 exclusion, before regional or
response filtering. Whole-brain site identifiers retain the Figure 1 naming
rule; native-coordinate conflicts are separately listed. Subject identity
comes from the explicit subject column, including the two D0091 recorded
names lacking the usual underscore delimiter. This adds descriptive plots
without changing sampling decisions or any main analysis.

Task display names are **Phoneme Sequencing**, **Lexical Delay**,
**Picture Naming**, and **Word Repetition**. Internal task identifiers and
input paths remain the original names.

### Hammers subdivisions and stage-mean HGA

Count all 348 sampled sites once using their frozen Figure 1 Hammers
midpoint labels, separating hemispheres and using one color per insular
subdivision. Export the full membership list and the six-subdivision counts.

Map the same 231 sound-responsive electrode waveforms used for Figure 1.
Mean HGA windows are Stimulus 0–0.6 s, Delay 0–1 s, and Response −0.5–1 s.
Both color and point area represent the mean HGA, as selected by the user.
Reuse the old `insula-aicl-delay/notebooks/local_insula.ipynb` 2D mapping:
area 5–60 pt², color and area clipped from zero to the within-phase P95.
Each phase has its own labeled magnitude scale; colors/areas alone do not
give absolute comparisons across phases. Negative values remain unchanged
in the source table and take the minimum display size/color. Use the frozen
Figure 1 background and coordinate projection. Gray rings show sampled
sites outside the frozen waveform cohort; they do not assert negative tests.

## Operculum control: approved 2026-10-18

The user selected **bandpassed-signal orthogonalization, followed by HGA**,
with nonshared bipolar contacts. This replaces the historical notebook's
trial-wise regression of already extracted HGA envelopes.

### Input selection

- AIC sites come from the accepted task-specific sampled cohort, with its
  Hammers grouping; HGA significance is not used to select the control.
- Frontal operculum (FO) midpoint is verified in native Destrieux
  `G_front_inf-Opercular`, using each participant's MRI. Endpoint coordinates
  must match native contact reconstruction. FO endpoints must be FO gray
  matter or white matter. Unknown tissue and region conflicts remain listed
  in the anatomical audit rather than entering the control.
- Known CSF contacts, including D0125 LIA6, remain excluded. A native FO
  label that conflicts with a Hammers insula midpoint is unresolved.
- Choose the nearest eligible FO site within subject, task and hemisphere,
  without any shared monopolar contacts. Use only good native channels.
- Use the existing Repeat + sound raw epochs and original bad-event/channel
  rejection. Record unavailable phases explicitly. Do not substitute old HGA
  averages for native voltage. This is a supplementary subset, not a change
  to the main sampling rule.

### Signal operation

Use the existing IEEG Gaussian filterbank centers, kernels, floating-point
precision and envelope **sum** over 70–200 Hz subbands. For complex analytic
signals A and B in each subband, calculate A relative to B as
`i * (B/abs(B)) * imag(A * conj(B)/abs(B))`, then take its magnitude.
Calculate the reverse direction independently. An exactly zero reference
has no defined phase and triggers an explicit audit rather than imputation.
Reference: Hipp et al. 2012, https://doi.org/10.1038/nn.3101.

Apply the same processing to baseline and task epochs. Preserve native
padding, crop task epochs to −1..1.5 s and baseline to −0.5..0 s, then
resample to 128 Hz. Normalize each branch against its own correspondingly
processed baseline. Save both physical envelope amplitudes and normalized
responses. A change in z score is not a physical amplitude retention ratio.

The unchanged branch must reproduce both the native HGA extractor and the
saved native power. Check that a purely instantaneous artificial mixture
is removed and a quadrature component is retained, using real filtered data.

### Statistical scope and displays

Use native `time_perm_cluster`: 5000 permutations, one-tailed task > baseline,
cluster-forming p=0.1, final p<0.05, ignore channel adjacency. Tests are
separate per channel/phase, as in the preprocessing pipeline. Use native
z scoring and 15-SD outlier handling. The fixed seed is recorded.

Display each pair's pre/post AIC responses and the reverse-direction FO
responses for Stimulus, Delay and Response. Summaries retain every analyzed
pair, including weak or negative responses. Reused FO sites and repeated AIC
sites across tasks are identified; there is no test assuming all pairs are
independent and no pooled group p value. A group inference beyond this scope
needs a reviewed statistical unit and correction family.

Retained task-related activity supports a component not explained by the
chosen neighbor's instantaneous in-phase signal. Orthogonalization itself
does not establish complete statistical or biological independence, nor
exclude every form of shared activity or leakage.

### Companion MRI and individual 3D localization

Each of the 15 analyzed pair × task waveform figures has a matching anatomy
figure displayed immediately afterward in `operculum.ipynb`. At the user's
latest request, display MRI only, without a CT overlay. Show all four
real contacts and both exact bipolar midpoints. Use two perpendicular
oblique native MRI planes through each contact axis, with a 180 mm field
showing the surrounding hemisphere and labeled insula/operculum boundaries.
Preserve these same planes and orient them by anatomy, rather than by the
electrode shaft: superior up for coronal/sagittal-like planes, anterior up
for axial-like planes (classified by the dominant plane-normal axis).
Show orientation letters and a 20 mm scale. Retain lateral plus oblique individual native pial views. Coordinates are retained
without cortical snapping. Insula and frontal operculum boundaries are
colored separately; tissue/region labels are printed for every point.
An endpoint may legitimately be white matter under the approved pair rule.

The user's full-shank update follows Figure 1: connect all existing native
reconstruction contacts on each selected shaft with a gray trajectory and
show every real contact. Color contacts by their native Destrieux label
(insula purple, frontal operculum blue, other tissue gray); selected pair
endpoints retain colored outlines and the bipolar midpoints remain diamonds.
The 3D close view includes the complete shafts. MRI retains its approved
upright planes and shows only shaft/contact portions intersecting the 1-mm
slab. Save all full-shank coordinates and native labels separately; do not
infer additional contacts or change any sampling or signal-analysis labels.

Restore existing registered CT and native surfaces from the Box Windows
reconstruction backup, recording source files and hashes. D0057's exported
contacts exactly match the 2022 `D57_2.LEPTO` file, not the 2021 `D57.LEPTO`
file; use the matching D57_2 CT, MRI and surfaces for this display. Do not
change the stored analysis coordinates to make them fit the wrong image.
D0032's existing
registered CT uses the historical name `ctINt1.nii.gz`; other participants
use `postInPre.nii.gz`. For the historical audit only, sample registered CT in original MRI
scanner geometry; do not perform a new registration or shift electrodes.
Export every native point and the corresponding image sources for review.

The matching-image review identifies two issues requiring review before
using all existing waveforms as anatomically confirmed AIC–FO controls:

- D0057 RAMT9–10 is `ctx_rh_S_temporal_sup` at its midpoint in the matching
  D57_2 MRI, with an Unknown first endpoint and a white-matter second
  endpoint. The earlier FO label came from the mismatched D57 MRI. Both
  its Phoneme Sequencing and Word Repetition figures remain visible as
  flagged candidate controls, not confirmed FO evidence.
- D0079 LAI1 is about 6 mm from the nearest voxel with registered CT
  intensity ≥1000. The original coordinate matches its LEPTO record, so
  this discrepancy is retained for review rather than resolved by moving
  the point. This intensity-distance check is a localization diagnostic,
  not a tissue classification or an automatic exclusion criterion.

Retain all 15 previously computed waveform records and their statistical
outputs for audit. Flag the three affected pair × task records. Do not
silently replace partners, change the main cohort, or re-fit NMF. The prior
15-record response counts are provisional and do not establish 15
anatomically confirmed AIC–FO controls.
# D0079 user-designated illustrative case (2026-10-18)

The user identified LAI7–8 as AIC and confirmed LFMI2–3 as the operculum
example (the spoken LFOMI name was corrected explicitly). This is distinct
from the original automatic LAI7–8 / LFMI7–8 control. Keep this
response-inspected example separate from the automatic nearest-partner
list and its descriptive summary. The native Destrieux midpoint labels
remain superior circular insular sulcus and cerebral white matter,
respectively; LFMI2 and LFMI3 are superior circular insular sulcus and
white matter. User localization is recorded alongside these automatic
labels, without changing main sampling or the old control results.

Use the existing Lexical Delay Repeat+sound epochs, original bad-event and
channel handling, nonshared contacts, approved 70–200 Hz analytic-subband
orthogonalization in both directions before envelopes, matching baseline
processing and the unchanged 5000-permutation phase/branch tests. The
original signal runner is imported unchanged with an isolated output root.
It verifies native HGA power reproduction and preserves the frozen main
inputs and original automatic-control lists/summaries by hash.

- Definition: `viewer/operculum_inspector/cases.json`.
- Runner: `scripts/run_operculum_manual_case.py`.
- Native inputs, points, atlas discrepancy and signal results:
  `results/supplementary/operculum/manual_cases/D0079_LAI7-8_LFMI2-3/`.
- Independent notebook: `vizpub/supplementary/operculum_D0079_case.ipynb`.
- Figures: `img/supplementary/operculum_D0079_case/`.
- Viewer: one-click case preset, manual-role colors, paired original HGA
  and clickable exact midpoint locations; original atlas labels remain visible.

