"""Visual check of the new scientific Figure3 SVGs, via Slurm only."""
from pathlib import Path
import io
import os
import cairosvg
from PIL import Image, ImageDraw

assert os.environ.get('SLURM_JOB_ID'),'Run via Slurm'
ROOT=Path('/cwork/ns458/insula-paper')
DEST=Path('/hpc/home/ns458/.codex/visualizations/2026/11/07/01a1b77b-8243-7f90-a68e-a0cfb4ba70d4/fig3_manual14_contact_sheet.png')
panels=[
    ('Picture Naming: condition-difference classes','fig3_picture_condition_difference_schematic.svg'),
    ('Lexical Delay: condition-difference classes','fig3_lexical_condition_difference_schematic.svg'),
    ('Picture Naming: selected electrode mean / heatmaps','fig3_picture_no_difference_average_time.svg'),
    ('Lexical Delay: selected electrode difference / heatmaps','fig3_lexical_decision_gt_repeat_difference_time.svg'),
    ('Picture Naming: current component weights','fig3_picture_weight_pair.svg'),
    ('Lexical Delay: current component weights','fig3_lexical_weight_pair.svg'),
    ('Picture Naming: current weight-density overlay','fig3_picture_weight_density_overlay.svg'),
    ('Lexical Delay: current weight-density overlay','fig3_lexical_weight_density_overlay.svg'),
]
canvas=Image.new('RGB',(2200,2800),'white');draw=ImageDraw.Draw(canvas)
for i,(title,name) in enumerate(panels):
    data=cairosvg.svg2png(url=str(ROOT/'img/fig3'/name),output_width=1600)
    im=Image.open(io.BytesIO(data)).convert('RGB');im.thumbnail((1060,640))
    x,y=(i%2)*1100,(i//2)*700
    draw.text((x+15,y+12),title,fill='black')
    canvas.paste(im,(x+(1100-im.width)//2,y+45+(640-im.height)//2))
DEST.parent.mkdir(parents=True,exist_ok=True);canvas.save(DEST);print(DEST)
