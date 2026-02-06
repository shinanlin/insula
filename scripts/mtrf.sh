#!/bin/bash
#SBATCH --job-name=mtrf_lexicaldelay
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/mtrf_lexicaldelay.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/mtrf_lexicaldelay.err
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-48%10

# Ensure logs dir exists
mkdir -p /hpc/group/coganlab/nanlinshi/insula/logs

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
module purge
module load CUDA/11.4

# BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/"
# DESCRIPTIONS=('Repeat')
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

# BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/"
# DESCRIPTIONS=('Repeat' 'Decision' 'Passive')
# SUBJECTS=(
#     D0024 D0026 D0027 D0028 \
#     D0029 D0053 D0054 D0057 \
#     D0063 D0065 D0069 D0071 \
#     D0077 D0086 D0090 D0092 \
#     D0094 D0100 D0121 D0128 \
#     D0137
# )

BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
DESCRIPTIONS=('Repeat' 'Decision')
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


BANDS=('highgamma')
PHASES=('Stimulus')
TYPE=(
  envelope
  mel
)
# TYPE=(
#   'mel'
# )


N_FOLDS=10
N_JOBS=18
TMIN=-0.2
TMAX=0.5
N_PERM=200
CONCAT=True

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Subject: ${SUBJECT}"

for DESCRIPTION in ${DESCRIPTIONS[@]}; do
  for PHASE in ${PHASES[@]}; do
    for BAND in ${BANDS[@]}; do
      echo "Processing: ${SUBJECT} (${TYPE}, ${BAND})"
      python -u src/encoder.py \
        --bids_root "${BIDS_ROOT}" \
        --subject "${SUBJECT}" \
        --band "${BAND}" \
        --description "${DESCRIPTION}" \
        --phase "${PHASE}" \
        --feature_type "${TYPE[@]}" \
        --n_jobs ${N_JOBS} \
        --n_folds ${N_FOLDS} \
        --tmin ${TMIN} \
        --tmax ${TMAX} \
        --n_perm ${N_PERM} \
        > /hpc/group/coganlab/nanlinshi/insula/logs/mtrf_lexicaldelay_${SUBJECT}_${BAND}_${PHASE}_${DESCRIPTION}.out \
        2> /hpc/group/coganlab/nanlinshi/insula/logs/mtrf_lexicaldelay_${SUBJECT}_${BAND}_${PHASE}_${DESCRIPTION}.err
      echo "Exit code: $?"
    done
  done
done
echo "Completed: ${SUBJECT}"