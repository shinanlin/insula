#!/bin/bash
#SBATCH --job-name=wpli_memprof_D0092
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/connectivity/wpli_memprof_D0092_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/connectivity/wpli_memprof_D0092_%j.err

set -eo pipefail

# Memory-profile float32-null + parallel wPLI on D0092 Stimulus Repeat.
# Writes to results/connectivity_wpli_memprof/ (does not overwrite formal results).

REPOSITORY=/hpc/group/coganlab/nanlinshi/insula-functional
MANIFEST="$REPOSITORY/results/connectivity/manifests/LexicalNoDelay_ready.tsv"
OUTPUT_ROOT="$REPOSITORY/results/connectivity_wpli_memprof"
ROW_INDEX=95
N_PERM=1000

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

echo "start=$(date -u +%FT%TZ) host=$(hostname) job=${SLURM_JOB_ID} mem=64G n_jobs=8 n_perm=${N_PERM} null=float32"

# Sample RSS of python child every 10s while it runs
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
  --overwrite &
PID=$!
MAX_RSS_KB=0
while kill -0 "$PID" 2>/dev/null; do
  if [[ -r /proc/$PID/status ]]; then
    RSS_KB=$(awk '/VmRSS:/{print $2}' /proc/$PID/status)
    if [[ -n "$RSS_KB" && "$RSS_KB" -gt "$MAX_RSS_KB" ]]; then
      MAX_RSS_KB=$RSS_KB
    fi
    # also sum thread? VmRSS is process RSS including threads
    echo "t=$(date -u +%FT%TZ) pid=$PID VmRSS_MB=$((RSS_KB/1024)) max_MB=$((MAX_RSS_KB/1024))"
  fi
  sleep 15
done
wait "$PID"
STATUS=$?

echo "sampler_max_rss_mb=$((MAX_RSS_KB/1024))"
echo "python_exit=$STATUS"

python - <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd
p = Path('/hpc/group/coganlab/nanlinshi/insula-functional/results/connectivity_wpli_memprof/LexicalNoDelay/sub-D0092/wpli')
prov = next(p.glob('*Stimulus*Repeat*_provenance.json'))
d = json.loads(prov.read_text())
print('provenance_peak_rss_mb', d.get('peak_rss_mb'))
print('elapsed_s', d.get('elapsed_seconds'))
print('implementation_hash', d.get('implementation_hash'))
print('n_pairs', d.get('runtime_metadata', {}).get('n_pairs'))
df = pd.read_parquet(Path(d['outputs']['pair_table']))
print('finite_p_frac', float(np.isfinite(df['p_uncorrected']).mean()))
print('null_std_gt0_frac', float((df['null_std'] > 0).mean()))
PY

echo "finish=$(date -u +%FT%TZ)"
exit $STATUS
