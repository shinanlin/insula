#!/bin/bash

#SBATCH --job-name=connectivity
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/connectivity.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/connectivity.err
#SBATCH --time=7-00:00:00
#SBATCH --mem=18G
#SBATCH --cpus-per-task=24
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0

module purge
module load CUDA/11.4
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

# Create logs directory if it doesn't exist
mkdir -p logs

# Parameters
BANDS='highgamma'
WINDOW=0.2
STEP=0.01
MODEL_ORDER=4
N_PERMUTATIONS=500
N_JOBS=20

# BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"
# SUBJECTS=(
#     D0019 D0022 D0023 D0024 \
#     D0025 D0028 D0029 D0031 \
#     D0035 D0040 D0041 D0042 \
#     D0045 D0049 D0052 D0053 \
#     D0054 D0055 D0056 D0057 \
#     D0058 D0059 D0060 D0061 \
#     D0063 D0064 D0066 D0067 \
#     D0068 D0069 D0070 D0071 \
#     D0073 D0075 D0077 D0079 \
#     D0084 D0085 D0086 D0088 \
#     D0091 D0092 D0093 D0094 \
#     D0095 D0096 D0100 D0102 \
#     D0103
# )
SUBJECTS=(
    D0103
)

# BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"
# SUBJECTS=(
#     D0023 D0024 D0026 D0027 \
#     D0028 D0029 D0032 D0035 \
#     D0038 D0042 D0044 D0047 \
#     D0053 D0054 D0055 D0057 \
#     D0059 D0063 D0065 D0066 \
#     D0068 D0069 D0070 D0071 \
#     D0077 D0079 D0080 D0081 \
#     D0084 D0086 D0090 D0092 \
#     D0094 D0096 D0100 D0101 \
#     D0102 D0103 D0107 D0115 \
#     D0117
# )

# Get the subject for this array job
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing subject: ${SUBJECT}"

# Run connectivity analysis for this ROI
for BAND in $BANDS; do
    echo "Processing: ${SUBJECT} (band=${BAND})"
    python src/run_connectivity.py \
        --bids_root "${BIDS_ROOT}" \
        --subject "${SUBJECT}" \
        --band "${BAND}" \
        --window "${WINDOW}" \
        --step "${STEP}" \
        --model_order "${MODEL_ORDER}" \
        --n_permutations "${N_PERMUTATIONS}" \
        --n_jobs "${N_JOBS}" \
        > logs/connectivity_${SUBJECT}_${BAND}.out \
        2> logs/connectivity_${SUBJECT}_${BAND}.err
    
    echo "Exit code: $?"
done

echo "Completed processing for subject: ${SUBJECT}"
echo "Job finished at: $(date)"