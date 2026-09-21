#!/bin/bash
#SBATCH --job-name=lda_bench
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_bench_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_bench_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#
# Sizes n_perm for decoding_resolved_lda_lexicality.sh: measures the permutation
# rate on the widest pool (STGl, 85 channels) and the narrowest (Sustain).

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

python -u -c "
import time
import joblib
from sklearn.model_selection import StratifiedGroupKFold

from src.decoding.decoder import decode_permutation_auc_pooled
from src.decoding.pooled_io import load_pooled_roi
from src.decoding.run_decoding_resolved_lda import build_classifier, build_transformer

N_JOBS = int('${SLURM_CPUS_PER_TASK:-16}')
N = 800
print('cores visible:', joblib.cpu_count(), 'n_jobs:', N_JOBS, flush=True)

for roi in ['STGl', 'Sustain', 'Sensory', 'Motor']:
    X, y, cond, _, _, _, _ = load_pooled_roi(
        '/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/', 'bipolar', roi,
        'lexicality', 'Repeat', 'Stimulus', 'highgamma', -0.5, 1.5)
    segment = X[..., 60:98].copy()
    start = time.time()
    decode_permutation_auc_pooled(
        segment, y,
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
        build_classifier(), transformer=build_transformer(5), groups=cond,
        n_jobs=N_JOBS, n_permutations=N, random_state=42)
    rate = N / (time.time() - start)
    for n_perm in (2000, 5000):
        print('%-8s %3d ch : %6.1f perms/s | n_perm=%4d -> %5.2f h for 57 windows'
              % (roi, X.shape[1], rate, n_perm, 57 * n_perm / rate / 3600), flush=True)
"
