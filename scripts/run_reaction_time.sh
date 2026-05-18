#!/bin/bash
#SBATCH --job-name=rt_pred_phonemesequence
#SBATCH --output=/hpc/home/ns458/coganlab/nanlinshi/insula/logs/rt_phonemesequence.out
#SBATCH --error=/hpc/home/ns458/coganlab/nanlinshi/insula/logs/rt_phonemesequence.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --partition=coganlab-gpu,common,scavenger
#SBATCH --chdir=/hpc/home/ns458/coganlab/nanlinshi/insula
#SBATCH --array=0-50%5

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
module purge

# Parameters
WINDOW=0.2
STEP=0.02
N_PERM=500
N_FOLDS=10
N_JOBS=30
BANDS=('highgamma')
REF='bipolar'

# BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
# SUBJECTS=(
#     D0023 D0024 D0026 D0027 
#     D0028 D0029 D0032 D0035 
#     D0038 D0042 D0044 D0047 
#     D0053 D0054 D0055 D0057 
#     D0059 D0063 D0065 D0066
#     D0068 D0069 D0070 D0071 
#     D0077 D0079 D0080 D0081 
#     D0084 D0086 D0090 D0092 
#     D0094 D0096 D0100 D0101 
#     D0102 D0103 D0107 D0115
#     D0117
# )


BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/"
SUBJECTS=(
    D0024 D0026 D0027 D0028 \
    D0029 D0053 D0054 D0057 \
    D0063 D0065 D0069 D0071 \
    D0077 D0086 D0090 D0092 \
    D0094 D0100 D0121 D0128 \
    D0137
)

# BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/"
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


# Limit BLAS/OpenMP threads to avoid nested parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

for BAND in ${BANDS[@]}; do
    echo "Processing: ${SUBJECT} | ${BAND}"
    
    python src/run_reaction_time.py \
        --subject ${SUBJECT} \
        --bids_root "${BIDS_ROOT}" \
        --band ${BAND} \
        --ref ${REF} \
        --window ${WINDOW} \
        --step ${STEP} \
        --n_perm ${N_PERM} \
        --n_folds ${N_FOLDS} \
        --n_jobs ${N_JOBS} \
    > /hpc/home/ns458/coganlab/nanlinshi/insula/logs/rt_${SUBJECT}_${BAND}.out \
    2> /hpc/home/ns458/coganlab/nanlinshi/insula/logs/rt_${SUBJECT}_${BAND}.err
done

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
