#!/bin/bash
#SBATCH --job-name=paper_fig6
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig6_refresh_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig6_refresh_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/cwork/ns458/insula-paper
export PYVISTA_OFF_SCREEN=true PYVISTA_USE_PANEL=false MNE_3D_BACKEND=notebook
export MPLBACKEND=Agg MESA_GL_VERSION_OVERRIDE=3.3
export INSULA_HGA_INPUT_ROOT=/hpc/group/coganlab/nanlinshi/insula-functional/results/hga
PYTHON=/hpc/home/ns458/miniconda3/envs/ieeg/bin/python
if [[ -f config/figure6_manual14_2026-11-07.json ]]; then
    # The current 67-subject projection/spatial caches were rebuilt separately.
    # Preserve legacy whole-brain results and redraw from this explicit version.
    exec xvfb-run -a -n "$((SLURM_JOB_ID % 20000 + 100))" -s '-screen 0 1600x1600x24' \
        "$PYTHON" -u scripts/execute_paper_notebook.py 6
fi
"$PYTHON" -u scripts/run_nmf_whole_brain_projection.py \
 --exclude-subject D0121 --tasks PhonemeSequence LexicalDelay PictureNaming SentenceRep \
 --phases stimulus delay response --significance-mode mask-any --min-coverage 0.95 --min-tasks 1
xvfb-run -a -s '-screen 0 1600x1600x24' "$PYTHON" -u scripts/plot_nmf_whole_brain_projection.py
bash scripts/slurm/refresh_paper_notebook.sh 6
