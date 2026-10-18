"""User-approved main-analysis conditions, windows, paths and figure colors."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONDITION='Repeat'
MODALITY='sound'
TASKS=('PhonemeSequence','LexicalDelay','PictureNaming','SentenceRep')
RESPONSE_WINDOWS={'stimulus':(0.,.6),'delay':(0.,1.),'response':(-.5,.5)}
NMF_WINDOWS={'stimulus':(0.,.6),'delay':(0.,1.),'response':(-.5,1.)}
GROUP_AVERAGE_UNIT='electrode'
HGA_ERROR_BAR='SEM'
PCA_VARIANCE_THRESHOLD=.05
COHORT_DIR=ROOT/'docs/evidence/sound_only_cohort_2026-10-17'
ROI_COLORS={'Insula':'#6A5ACD','STG':'#20B2AA','SMC':'#F39B35','IFG':'#2369BD','Other':'#D3D3D3'}
# Functional components and anatomical ROIs have distinct meanings and palettes.
COMPONENT_COLORS={'sustain':'#C4A35A','motor':'#A9373B','sensory':'#2369BD'}
