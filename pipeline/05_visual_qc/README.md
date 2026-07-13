# Stage 3 — Parcellation slice QC

Native MRI three-plane slices for **pure insula** bipolar electrodes after
Stage 2 parcellation.

## Filter rules

| Atlas | Include when |
|-------|----------------|
| `hammers` | `mix == False` and `roi ∈ {AIC, PIC}` | red dashed = MAPER Hammersmith insula |
| `aparc2009s` | `mix == False` and `roi ∈ {INS, Insula}` | blue solid = aparc2009s insula (`aparc.a2009s+aseg.mgz`) |

Only subjects with at least one pure insula electrode (`mix=False`) get an
output folder. Subjects with no matching electrodes are skipped entirely.

## Output

`results/qc/{atlas}/sub-{SUBJECT}/`

```text
index.csv
png/{channel}.png
D0094_hammers_insula_slices.pdf
```

## Usage

```bash
conda activate ieeg
python pipeline/05_visual_qc/plot_parcellation_slices.py \
  --parcellation-csv /cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/parcellation/sub-D0094/bipolar/sub-D0094_hammers.csv \
  --atlas hammers \
  --recon-dir /cwork/ns458/ECoG_Recon \
  --fused /cwork/ns458/maper_run/D0094/output/f30-seg95-D0094.nii.gz
```

SLURM (single subject):

```bash
sbatch --export=SUBJ=D0094,BIDS=/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS \
  pipeline/05_visual_qc/run_parcellation_slices.sbatch
```

Stage 2 batch (`scripts/slurm/run_hammers_parcellation.sbatch`) chains Hammers QC
automatically after successful Hammers parcellation.

All subjects (either atlas):

```bash
# Hammers (default)
sbatch scripts/slurm/run_parcellation_slices_all.sbatch

# aparc2009s
sbatch --job-name=parc_slices_aparc --export=ALL,ATLAS=aparc2009s \
  scripts/slurm/run_parcellation_slices_all.sbatch
```

## Legacy

`plot_atlas_conflict_slices.py` is the earlier aparc/MAPER conflict QC script;
new work should use `plot_parcellation_slices.py`.
