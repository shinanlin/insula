#!/bin/bash
#SBATCH --job-name=filter_insula
#SBATCH --output=/hpc/home/ns458/coganlab/nanlinshi/insula/logs/filter_insula_%s.out
#SBATCH --error=/hpc/home/ns458/coganlab/nanlinshi/insula/logs/filter_insula_%s.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24
#SBATCH --partition=common
#SBATCH --chdir=/hpc/home/ns458/coganlab/nanlinshi/insula
#SBATCH --array=0-50  # Process all subjects, max 50 at a time

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
module purge
module load CUDA/11.4

# Configuration
RECON_DIR="/cwork/ns458/ECoG_Recon/"
SCRIPT_DIR="/hpc/home/ns458/coganlab/nanlinshi/insula/src"


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

BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"
SUBJECTS=(
    D0023 D0024 D0026 D0027 \
    D0028 D0029 D0032 D0035 \
    D0038 D0042 D0044 D0047 \
    D0053 D0054 D0055 D0057 \
    D0059 D0063 D0065 D0066 \
    D0068 D0069 D0070 D0071 \
    D0077 D0079 D0080 D0081 \
    D0084 D0086 D0090 D0092 \
    D0094 D0096 D0100 D0101 \
    D0102 D0103 D0107 D0115 \
    D0117
)

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
PROXIMITY_THRESHOLD=4.0
echo "Processing subject $SUBJECT (array task $SLURM_ARRAY_TASK_ID)"

python "${SCRIPT_DIR}/filter_insula_electrodes.py" \
    --bids_root "${BIDS_ROOT}" \
    --recon_dir "${RECON_DIR}" \
    --subject ${SUBJECT} \
    --proximity_threshold ${PROXIMITY_THRESHOLD} \
    > /hpc/home/ns458/coganlab/nanlinshi/insula/logs/filter_insula_${SUBJECT}.out \
    2> /hpc/home/ns458/coganlab/nanlinshi/insula/logs/filter_insula_${SUBJECT}.err

echo "Finished processing subject ${SUBJECT}"