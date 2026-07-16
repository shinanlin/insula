#!/bin/bash
#SBATCH --job-name=sem_ridge_smoke
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm/sem_ridge_smoke_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm/sem_ridge_smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-semantic

mkdir -p /hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

SUBJECT=D0092

echo "=== Observed + fold-inner permutation significance (n_perm=50) ==="
python src/semantic/run_encoding.py --subject "${SUBJECT}" --n_perm 50 --n_jobs 2

echo "=== Verify H5 contains mask/p_values ==="
python - <<'PY'
import h5py
from pathlib import Path

h5 = next(Path("results/semantic/LexicalDelay").rglob(f"sub-{SUBJECT}/*_ridge_glove.h5"))
with h5py.File(h5, "r") as f:
    for key in ("r", "r_null", "mask", "p_values"):
        assert key in f, f"missing dataset: {key}"
    assert f["mask"].shape == f["r"].shape
    assert f["p_values"].shape == f["r"].shape
    assert f["r_null"].shape[:2] == f["r"].shape
    assert f.attrs.get("significance_method") == "channel_time_cluster"
print(f"OK: {h5}")
PY
