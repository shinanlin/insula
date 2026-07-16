#!/bin/bash
#SBATCH --job-name=sem_ridge_full
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm/sem_ridge_full_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm/sem_ridge_full_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-semantic
#SBATCH --array=0-415%20

# Full semantic ridge encoding sweep:
#   phases: Stimulus, Go, Response, Delay
#   descriptions: Decision, Repeat
# One subject x condition per array task (52 subjects x 8 = 416 tasks).
# Existing outputs are skipped.

mkdir -p /hpc/group/coganlab/nanlinshi/insula-semantic/logs/slurm

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

SUBJECTS=(
    D0023 D0024 D0026 D0027 D0028 D0029 D0032 D0035 D0038 D0042
    D0044 D0047 D0053 D0054 D0055 D0057 D0059 D0063 D0065 D0066
    D0068 D0069 D0070 D0071 D0077 D0079 D0080 D0081 D0084 D0086
    D0090 D0092 D0094 D0096 D0100 D0101 D0102 D0103 D0107 D0115
    D0117 D0121 D0127 D0128 D0129 D0132 D0135 D0137 D0138 D0139
    D0140 D0143
)

PHASES=(Stimulus Go Response Delay)
DESCRIPTIONS=(Decision Repeat)

N_CONDITIONS=8
SUBJECT_IDX=$((SLURM_ARRAY_TASK_ID / N_CONDITIONS))
COND_IDX=$((SLURM_ARRAY_TASK_ID % N_CONDITIONS))
PHASE_IDX=$((COND_IDX / 2))
DESC_IDX=$((COND_IDX % 2))

SUBJECT=${SUBJECTS[$SUBJECT_IDX]}
PHASE=${PHASES[$PHASE_IDX]}
DESCRIPTION=${DESCRIPTIONS[$DESC_IDX]}

OUT_DIR="results/semantic/LexicalDelay/sub-${SUBJECT}"
OUT_FILE="${OUT_DIR}/sub-${SUBJECT}_task-LexicalDelay_proc-${PHASE}_desc-${DESCRIPTION}_ridge_glove.h5"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Subject: ${SUBJECT}"
echo "Phase: ${PHASE}"
echo "Description: ${DESCRIPTION}"
echo "Output: ${OUT_FILE}"

if [[ -f "${OUT_FILE}" ]]; then
    echo "Skip: output already exists"
    exit 0
fi

python src/semantic/run_encoding.py \
    --subject "${SUBJECT}" \
    --phase "${PHASE}" \
    --description "${DESCRIPTION}" \
    --n_perm 500 \
    --n_jobs 2
