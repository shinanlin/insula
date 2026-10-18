# Insula Paper analysis rules

- Main analyses use **Repeat + sound only**, including responsive-site selection, Figure 1 HGA and significance counts, NMF discovery inputs, and dependent main-analysis figures.
- Never pool image/picture or text into the main selection or discovery waveform. Those conditions are separate modality comparisons; they cannot add a site to the main sound-responsive cohort.
- Use `src/main_analysis.py` for conditions, windows, cohort paths and palettes. Read `docs/MAIN_ANALYSIS.md` before changing them. A task named PictureNaming contains auditory trials too; select rows by modality, not by task name.
- Anatomical sampling and statistical responsiveness are separate steps. Keep unresolved anatomy and missing statistics explicit; do not treat missing evidence as negative.
- Retain zero endpoints whose native aparc, aseg and ribbon are all Unknown/unassigned, as explicitly requested by the user. Flag them as accepted Unknown, never as confirmed white matter. Insular midpoint and identity checks still apply; known CSF/noninsula gray matter and coordinate/version conflicts retain their existing decisions. Hammers-zero/FreeSurfer-insula disagreements remain separately pending.
- Use the historical ROI palette, especially SMC `#F39B35`. Anatomical ROI colors and NMF component colors are separate conventions.
- Run processing, model fitting, plotting and tests via Slurm. Use existing environments, and export figures as SVG only. Keep outputs in this `/cwork` worktree; original insula-functional inputs are read-only.
