"""Temporary visual review of the refreshed scientific SVGs; run via Slurm."""
from pathlib import Path
import io
import os
import cairosvg
from PIL import Image, ImageDraw

assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
ROOT = Path('/cwork/ns458/insula-paper')
PREVIEW = Path('/hpc/home/ns458/.codex/visualizations/2026/11/07/01a1b77b-8243-7f90-a68e-a0cfb4ba70d4/fig1_cohort67_contact_sheet.png')
panels = [
    ('Whole-brain sampling: 67 participants', 'fig1_all_electrodes_projected.svg'),
    ('Insula: 378 sampled / 239 confirmed responsive', 'fig1_insula_electrodes_gyrus_lines.svg'),
    ('Regional responsive / sampled composition', 'fig1_roi_breakdown_sig_and_sampled.svg'),
    ('Response composition by phase', 'fig1_phase_roi_breakdown.svg'),
    ('Confirmed response yield by region', 'fig1_roi_significant_yield.svg'),
    ('Pooled HGA: 239 electrode means', 'fig1_insula_average_traces.svg'),
]
canvas = Image.new('RGB', (1800, 1800), 'white')
draw = ImageDraw.Draw(canvas)
for i, (title, name) in enumerate(panels):
    rendered = cairosvg.svg2png(url=str(ROOT/'img/fig1'/name), output_width=1400)
    im = Image.open(io.BytesIO(rendered)).convert('RGB')
    im.thumbnail((860, 535))
    x, y = 900 * (i % 2), 600 * (i // 2)
    draw.text((x + 20, y + 15), title, fill='black')
    canvas.paste(im, (x + (900 - im.width)//2, y + 50 + (540 - im.height)//2))
PREVIEW.parent.mkdir(parents=True, exist_ok=True)
canvas.save(PREVIEW)
print(PREVIEW)
