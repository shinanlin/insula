#!/bin/bash
#SBATCH --job-name=wpli_smoke_D0092
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/connectivity/wpli_smoke_D0092_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/connectivity/wpli_smoke_D0092_%j.err

set -eo pipefail

REPOSITORY=/hpc/group/coganlab/nanlinshi/insula-functional
MANIFEST="$REPOSITORY/results/connectivity/manifests/LexicalNoDelay_ready.tsv"
# One-subject smoke: do not overwrite formal results/
OUTPUT_ROOT="$REPOSITORY/results/connectivity_wpli_smoke"
ROW_INDEX=95   # D0092 / LexicalNoDelay / Stimulus / Repeat
N_PERM=200     # enough to verify non-degenerate null

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd "$REPOSITORY"

export PYTHONDONTWRITEBYTECODE=1
export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$OUTPUT_ROOT" \
  "$REPOSITORY/logs/connectivity"

echo "start=$(date -u +%FT%TZ) host=$(hostname) job=${SLURM_JOB_ID}"
echo "manifest=${MANIFEST} row=${ROW_INDEX} n_perm=${N_PERM} out=${OUTPUT_ROOT}"

python -m src.connectivity.pairwise.cli run-row \
  --manifest "$MANIFEST" \
  --row-index "$ROW_INDEX" \
  --metrics wpli \
  --output-root "$OUTPUT_ROOT" \
  --scratch-dir "${SLURM_TMPDIR:-/tmp}" \
  --n-perm "$N_PERM" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --pair-block-size 32 \
  --permutation-chunk-size 100 \
  --require-parquet \
  --overwrite

python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("/hpc/group/coganlab/nanlinshi/insula-functional/results/connectivity_wpli_smoke")
paths = sorted(root.glob("LexicalNoDelay/sub-D0092/wpli/*Stimulus*Repeat*_pairs.parquet"))
assert paths, f"no pairs parquet under {root}"
df = pd.read_parquet(paths[0])
print("file:", paths[0])
print("rows:", len(df))
print("bands:", sorted(df["band"].astype(str).unique().tolist()) if "band" in df.columns else None)
finite_p = np.isfinite(df["p_uncorrected"]).sum()
print(f"finite p_uncorrected: {finite_p} / {len(df)} ({100*finite_p/len(df):.1f}%)")
print(f"null_std==0: {(df['null_std']==0).sum()}")
print(f"null_std>0: {(df['null_std']>0).sum()}")
print(f"null_std median: {df['null_std'].median():.6g}")
print("by band:")
for band, g in df.groupby("band"):
    print(
        f"  {band}: finite_p={np.isfinite(g['p_uncorrected']).mean():.1%} "
        f"null_std_med={g['null_std'].median():.4g} "
        f"null_mean_med={g['null_mean'].median():.4g} "
        f"stat_med={g['stat'].median():.4g}"
    )
# secondary bands should NOT all share the same null_mean
means = {
    b: float(g["null_mean"].median())
    for b, g in df.groupby("band")
    if b in {"theta", "alpha", "beta"}
}
print("secondary null_mean medians:", means)
if len(means) == 3:
    vals = list(means.values())
    print(
        "secondary null_means identical?",
        np.allclose(vals[0], vals[1]) and np.allclose(vals[1], vals[2]),
    )
print("PASS_SMOKE" if finite_p > 0 and (df["null_std"] > 0).any() else "FAIL_SMOKE")
PY

echo "finish=$(date -u +%FT%TZ) job=${SLURM_JOB_ID}"
