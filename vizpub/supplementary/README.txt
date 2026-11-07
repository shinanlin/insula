Independent supplementary notebooks

Each supplementary topic has its own notebook here. Main Figure 1-6
notebooks are not executed or edited to generate these figures.
Outputs: img/supplementary/<topic>/ and results/supplementary/<topic>/.
Publication exports are SVG only. Execute through Slurm using
scripts/slurm/supplementary_notebook.sbatch <topic>.

task_sampling.ipynb
- Reuses the approved Figure 1 sound Repeat cohort and display coordinates.
- Four-task L/R insular coverage and sampled/confirmed-response counts;
  detailed site and subject/task CSVs accompany it. The redundant
  subject-by-task matrix figure is removed; overlap uses Venn diagrams.
- Frozen cohort: 348 unique sampled sites / 65 subjects; 231 confirmed
  responsive sites / 62 subjects. Task-site records are not unique-site totals.
- Incomplete response evidence remains explicit and is not a negative result.
- This displays the frozen Figure 1 cohort, not a newly validated cohort
  formed from the native preprocessing restoration now in progress.
- Directly below task-insula coverage: three four-task Venn diagrams for
  all recorded participants, all recorded bipolar subject/site identifiers,
  and accepted sampled insula sites. No significance filter. Every exclusive
  region and its members are exported; ellipse areas do not encode counts.
- Whole-brain participant/site membership reuses Figure 1 Repeat+sound
  coordinate records after D0121 exclusion. It precedes ROI/insula filtering;
  the participant diagram is not limited to participants with insula sites.
  Recorded-name overlap does not adjudicate native-coordinate conflicts.
- All task names are written in full: Phoneme Sequencing, Lexical Delay,
  Picture Naming, and Word Repetition.
- Hammers insular subdivision counts use all 348 sampled sites, one color
  per subdivision. Stage-mean HGA maps reuse the 231 frozen Figure 1
  waveforms: Stimulus 0-0.6, Delay 0-1, Response -0.5-1 s. Both point area
  and color encode HGA, with the old 2D zero-to-phase-P95 size mapping.

operculum.ipynb
- Focus: task-related insular activity retained after orthogonalization.
- User-approved method (2026-10-18): analytic-signal orthogonalization within
  the existing 70-200 Hz subbands, before HGA envelope extraction, with the
  same operation applied to baseline and the reverse direction separately.
- Historical reference: insula-aicl-delay/notebooks/opercular.ipynb performs
  trial-wise HGA linear projection residuals. The D0106 example shares a
  contact. This is distinct from carrier-level orthogonalization before HGA.
- Broad Hammers IFG labels are not silently equated with frontal operculum.
  Native MRI verifies FO midpoint and endpoint labels; coordinates must match
  native contacts. Pairs do not share contacts and preserve native QC.
- The notebook shows all completed eligible pairs, including weak responses,
  with before/after AIC and reverse FO traces, physical amplitude and z-score
  summaries, per-channel/phase native tests, and processing provenance.
- Every waveform figure is immediately followed by its companion wide-context MRI
  and individual 3D localization: four real contacts and two exact virtual
  bipolar midpoints, original native coordinates without cortical snapping.
  MRI uses a 180 mm anatomical field with insula/operculum labels; CT is not
  displayed, per the latest request. Old CT checks remain audit evidence.
  Preserve each contact plane but orient the display by anatomy: superior up
  for coronal/sagittal-like planes, anterior up for axial-like planes, rather
  than using the electrode shaft as the horizontal direction.
  The 3D views also show every reconstructed contact along each complete
  selected shank. Gray trajectories/other contacts provide context; native
  insula contacts are purple and native FO contacts blue. Selected real
  contacts retain outlined markers and virtual midpoints are diamonds.
  MRI shows the intersecting 1-mm slab; the 3D close view includes both
  complete shanks. All full-shank coordinates are exported separately.
- Matching-image review uses D57_2 for D0057; RAMT9-10 has a superior
  temporal sulcus midpoint there, rather than verified FO. D0079 LAI1 is
  about 6 mm from visible high-intensity CT. Three pair/task records are
  explicitly flagged in anatomy/current_anatomy_review.csv, including both
  D0057 task records. All old waveforms remain visible for review. Existing
  15-record response counts are provisional, not 15 confirmed FO controls.
- Repeated partners are explicit; there is no independent-pair group test.
  Retained responses do not prove independent neural generators.
- Detailed method and limits: docs/SUPPLEMENTARY_ANALYSIS.md.

Execution receipts record that main notebooks and NMF files are unchanged.
