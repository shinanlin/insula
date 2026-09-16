#!/bin/bash
# Submit Hammers-insula TFR (AF multitaper recipe, SHI-87).
set -eo pipefail
cd /hpc/group/coganlab/nanlinshi/insula-functional
mkdir -p logs/slurm
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR=/work/ns458/numba_cache
export MNE_CONFIG_PATH=/work/ns458/mne_config
SBATCH=scripts/slurm/run_insula_tfr.sbatch
N=$(python -c "from src.tfr.run import job_table; print(len(job_table()))")
LAST=$((N - 1))
TEST=$(sbatch --parsable --job-name=insula_tfr_test --time=00:15:00 --mem=8G --cpus-per-task=2 "$SBATCH" test)
SMOKE=$(sbatch --parsable --job-name=insula_tfr_smoke --depend=afterok:"$TEST" --time=01:00:00 --mem=128G --cpus-per-task=4 "$SBATCH" smoke)
RUN=$(sbatch --parsable --job-name=insula_tfr_run --depend=afterok:"$SMOKE" --array=0-"$LAST"%8 --time=06:00:00 --mem=128G --cpus-per-task=4 "$SBATCH" run)
echo "n_jobs=$N test=$TEST smoke=$SMOKE run=$RUN"
echo "logs: logs/slurm/insula_tfr_<jobid>_<task>.out"
