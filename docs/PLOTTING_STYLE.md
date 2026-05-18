# Plotting Style Guide

This guide describes conventions for notebooks and figure outputs in the Insula analysis workspace.

## Plotting Stack

Use the existing plotting stack unless there is a specific reason to add another dependency:

- `matplotlib` for core plotting and saving.
- `seaborn` for line plots, bar plots, heatmaps, and style helpers.
- `mne` and `mne.viz.Brain` for cortical surface visualizations.
- `matplotlib_venn` only where set diagrams are needed.

Publication and grant figures should usually be exported as SVG.

## Notebook Locations

- `viz/`: working and historical figure notebooks.
- `vizpub/`: publication-oriented notebooks. This is the actual directory name, even if it is informally called `viz_pub`.
- `grant/`: R01 grant-specific notebooks and figures.
- `notebooks/`: exploratory analyses that are not yet part of a figure pipeline.

Keep figure-production notebooks close to their intended output context. For example, grant-specific plots should stay in `grant/`, while paper figures should be prepared from `viz/` or `vizpub/`.

## Canonical Style Setup

Most figure notebooks use a shared style pattern. New figure notebooks should start with a similar setup:

```python
import matplotlib.pyplot as plt
import seaborn as sns

cm = 1 / 2.54
plt.rcParams["svg.fonttype"] = "none"

fontsize = 7
fontdict = {"fontsize": fontsize}
```

Using `svg.fonttype = "none"` keeps SVG text editable in downstream design tools.

## Observed Notebook Template

The style source for this guide is `vizpub/fig2.ipynb`, which is a better reference for publication-style Figure 2 panels. It uses a compact two-cell setup: one cell for scientific/plotting imports and one cell for visual style constants.

Typical imports:

```python
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from IPython.display import Audio
import IPython.display as ipd
from scipy.io import wavfile
import tempfile
import os
import librosa
import pandas as pd
import seaborn as sns
import h5py
import mne
from scipy.stats import zscore
from mne_bids import BIDSPath, read_raw_bids
from matplotlib_venn import venn2, venn2_circles
from tqdm import tqdm
import xarray as xr
```

Typical style cell:

```python
cm = 1 / 2.54
plt.rcParams["svg.fonttype"] = "none"

fontdict = dict(fontsize=7)
fontsize = 7

red = "#A9373B"
blue = "#2369BD"
orange = "#CC8963"
green = "#009944"

stg_color = "#20B2AA"
smc_color = "#6A5ACD"
insula_color = "#D4AF37"

reds = sns.light_palette(red, as_cmap=True)
blues = sns.light_palette(blue, as_cmap=True)
oranges = sns.light_palette(orange, as_cmap=True)
greens = sns.light_palette(green, as_cmap=True)

recon_dir = "/cwork/ns458/ECoG_Recon/"
mne.viz.set_3d_backend("notebook")  # MNE 3D in-notebook static backend
```

This pattern keeps figure notebooks self-contained. If a future shared style helper is introduced, it should preserve these names so older notebooks can be migrated gradually.

## Canonical Colors

Use the existing color vocabulary where possible:

```python
red = "#A9373B"
blue = "#2369BD"
orange = "#CC8963"
green = "#009944"

stg_color = "#20B2AA"
smc_color = "#6A5ACD"
insula_color = "#D4AF37"
```

Common ROI color roles:

- `insula_color`: insula or AIC/PIC emphasis.
- `stg_color`: superior temporal gyrus.
- `smc_color`: sensorimotor cortex.

When making heatmaps, prefer palette-derived colormaps over ad hoc colors:

```python
cmap = sns.light_palette(insula_color, as_cmap=True)
```

## Figure Size

Specify figure sizes in centimeters using the `cm` conversion factor:

```python
fig, ax = plt.subplots(figsize=(8 * cm, 5 * cm))
```

Use compact figure sizes and font sizes suitable for manuscript panels. The common base font size is 7 pt.

## Axes and Seaborn Style

For time series and grouped summaries:

- Use `sns.lineplot` or `sns.barplot` when the plot is naturally tidy-data based.
- Use direct `matplotlib` calls when plotting arrays, brain projections, or highly customized panels.
- Use `sns.despine(ax=ax, offset=1, trim=True)` for most clean publication axes.
- Keep axis labels short and publication-ready.
- Do not encode critical analysis parameters only in the plot title; include them in filenames and notebook variables as well.

When using a palette with a `hue` variable, make sure the palette has the same number of colors as hue levels.

## Spines, Ticks, and Line Widths

`vizpub/fig2.ipynb` uses thin, publication-style axes. The most common settings are:

```python
ax.tick_params(labelsize=7, width=0.75, length=2, which="both")
plt.setp(ax.spines.values(), linewidth=0.75)
sns.despine(ax=ax, offset=1, trim=True)
```

Equivalent loop form is also used:

```python
for spine in ax.spines.values():
    spine.set_linewidth(0.75)
sns.despine(ax=ax, offset=1, trim=True)
```

Use these defaults for most small manuscript panels:

- Axis spine linewidth: `0.75`.
- Tick width: `0.75`.
- Tick length: `2`.
- Tick label size: `7`.
- Main lineplot width: `lw=1`.
- Zero/reference lines: `linewidth=0.5` for small time-series and heatmap panels.

Typical time-series panel:

```python
sns.lineplot(
    data=subset,
    x="time",
    y="value",
    hue="roi",
    ax=ax,
    palette=colors,
    lw=1,
)

ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5)
ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
ax.tick_params(labelsize=7, width=0.75, length=2, which="both")
plt.setp(ax.spines.values(), linewidth=0.75)
sns.despine(ax=ax, offset=1, trim=True)
```

For multi-panel HGA traces where y-axis labels are intentionally suppressed after the first panel, Figure 2 hides the left spine and removes y ticks:

```python
ax.set_yticklabels([])
ax.set_yticks([])
ax.spines["left"].set_visible(False)
```

For heatmaps, Figure 2 uses `pcolormesh(..., rasterized=True)`, no y ticks, no left spine, and no despine offset:

```python
im = ax.pcolormesh(T, C, da.values, cmap=cmap, vmin=0, vmax=1.5, shading="auto", rasterized=True)
sns.despine(ax=ax, offset=0, trim=False)
ax.spines["left"].set_visible(False)
ax.set_yticks([])
ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5)
```

For larger bar or connectivity summary panels, slightly stronger tick settings are acceptable:

```python
ax.tick_params(labelsize=8, width=0.8, length=3, which="both")
for spine in ax.spines.values():
    spine.set_linewidth(0.75)
sns.despine(ax=ax, offset=2, trim=True)
```

Patch and marker outlines are usually thin but visible:

- Stacked bar patch edge: `edgecolor="white", linewidth=0.9`.
- Scatter outline: `edgecolor="black", linewidth=1`.
- Subject traces or strip lines: `linewidths=1.2`, often with `alpha=0.7`.

## Brain Plots

For cortical surface plots, use the existing MNE workflow and reconstruction directory:

```python
recon_dir = "/cwork/ns458/ECoG_Recon/"
```

In notebooks, the common backend is:

```python
import mne

mne.viz.set_3d_backend("notebook")
```

If a figure must be exported headlessly from SLURM, document the backend and rendering assumptions in the notebook or script.

Figure notebooks commonly build left and right hemisphere `Brain(...)` objects, highlight insula labels with `add_label(..., color=insula_color, alpha=0.5 or 0.6)`, and overlay electrode locations with `add_foci(...)`. For unlabeled background regions, light gray labels such as `(0.9, 0.9, 0.9)` are used.

## Save Locations

Main manuscript figures:

```text
img/fig<figure_number>/
```

Examples:

```text
img/fig2/
img/fig3/
img/fig4/
```

Miscellaneous or exploratory outputs:

```text
img/misc/
```

Grant-specific outputs:

```text
grant/grantfig/
```

3D cross-correlation viewers:

```text
viz/3d_xcorr/
```

Remember that `img/` is gitignored. Grant figures under `grant/grantfig/` may be versioned depending on project needs.

## Save Format

Use SVG for manuscript and grant vector figures:

```python
fig.savefig(
    "../img/fig3/fig3_<description>.svg",
    dpi=300,
    bbox_inches="tight",
)
```

Use `fig.savefig(...)` rather than `plt.savefig(...)` when a notebook has multiple active figures or axes.

Use PNG only when raster output is required, such as screenshots, videos, or image-based intermediate products.

## File Naming

Prefer filenames that encode the figure number and key analysis dimensions:

```text
fig<figure_number>_<task>_<analysis>_<roi>_<phase>.svg
```

Examples:

```text
fig3_LexicalDelay_phoneme.svg
phoneme_window_decoding_Delay.svg
cross_task_signed_heatmap_Decision_direction_first.svg
aim1_SMC_L_HGAs.svg
fig2_AIPI_spatial_Stimulus.svg
heatmap_AIC_L_channels.svg
fig2_delay_AIC_connectivity_brain.svg
```

Use lowercase only when matching an existing family of files. Otherwise, preserve task names such as `LexicalDelay` and phase names such as `Stimulus`.

## Notebook Workflow

Recommended notebook order:

1. Imports and style setup.
2. Project paths and task parameters.
3. Data loading.
4. Data cleaning or aggregation.
5. Statistical summaries.
6. Figure construction.
7. Save/export cell.

During exploration, it is acceptable to comment out `savefig` calls. For a final figure run, make save cells explicit and ensure filenames match the intended figure panel.

Avoid copying large blocks of style code across many notebooks when a shared helper would be clearer. A future refactor could add a small shared plotting helper, but current notebooks still mostly define style inline.

## Titles, Labels, and Text

- Use concise axis labels.
- Prefer task and condition names that match the codebase: `Repeat`, `Decision`, `Passive`, `Stimulus`, `Delay`, `Go`, `Response`.
- Avoid absolute file paths in titles.
- Avoid embedding interpretation-heavy conclusions directly into panel titles.
- Use legends only when they add information not already clear from panel layout.

## Anti-Patterns

Avoid the following in new figure code:

- Saving publication panels as PNG unless required.
- Writing figures to arbitrary temporary paths.
- Using hard-coded one-off colors when a canonical color already exists.
- Relying on a hidden active `plt` figure instead of saving an explicit `fig`.
- Leaving final figure filenames ambiguous, such as `plot.svg` or `test.svg`.
- Mixing grant-specific output into `img/figN/` unless it is also part of the manuscript figure workflow.
