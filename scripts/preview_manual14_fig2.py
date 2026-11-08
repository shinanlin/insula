"""Create a temporary visual-QA contact sheet outside scientific output paths."""
from pathlib import Path
import json
import os
import cairosvg
import pandas as pd
from PIL import Image,ImageDraw

assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
ROOT=Path('/cwork/ns458/insula-paper')
assert json.loads((ROOT/'results/insula_manual14_2026-11-07/completed.json').read_text())['passed']
PREVIEW=Path('/hpc/home/ns458/.codex/visualizations/2026/11/07/01a1b77b-8243-7f90-a68e-a0cfb4ba70d4')
PREVIEW.mkdir(parents=True,exist_ok=True)
names=['fig2_nmf_traces.svg','fig2_hga_traces.svg','fig2_hga_heatmap.svg',
       'fig2_insula_spatial_kdecontour.svg','fig2_nmf_rank_k.svg','fig2_laterality_cluster_rate.svg']
panels=[]
for name in names:
    path=PREVIEW/(name.removesuffix('.svg')+'_manual14_preview.png')
    cairosvg.svg2png(url=str(ROOT/'img/fig2'/name),write_to=str(path),output_width=1000,background_color='white')
    picture=Image.open(path).convert('RGB');picture.thumbnail((900,700))
    panels.append((name,picture.copy()))
canvas=Image.new('RGB',(1840,2250),'white');draw=ImageDraw.Draw(canvas)
for i,(name,picture) in enumerate(panels):
    x=20+(i%2)*920;y=20+(i//2)*740
    draw.text((x,y),name,fill='black')
    canvas.paste(picture,(x,y+25))
destination=PREVIEW/'fig2_manual14_contact_sheet.png';canvas.save(destination)
print('PREVIEW',destination,flush=True)
for name in ['pooled_waveform_summary.csv','cluster_waveform_summary.csv']:
    table=pd.read_csv(ROOT/'results/fig2'/name)
    print(name,'maximum mean+SEM:',float((table['mean']+table['sem']).max()),flush=True)
