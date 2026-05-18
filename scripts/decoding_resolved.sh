#!/bin/bash

#SBATCH --job-name=decoding_resolved
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/decoding_resolved.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/decoding_resolved.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --partition=coganlab-gpu,common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-39%10

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
module purge
module load CUDA/11.4


SUBJECTS=(
  AICl
  PICl
  SMCl
  IFGl
  STGl
)


# BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"
# DESCRIPTIONS=('Repeat')
# BANDS=('highgamma')
# # DATATYPES=('phoneme' 'token' 'articulator')
# DATATYPES=('token')
# PHASES=('Stimulus' 'Delay' 'Go' 'Response')


BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
DESCRIPTIONS=('Repeat' 'Decision')
BANDS=('highgamma')
DATATYPES=('phoneme')
PHASES=('Stimulus' 'Delay' 'Go' 'Response')

# BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/"
# DESCRIPTIONS=('Repeat' 'Decision' 'Passive')
# BANDS=('highgamma')
# DATATYPES=('phoneme' 'lexicality')
# PHASES=('Stimulus' 'Response')

# BIDS_ROOT="/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/"
# DESCRIPTIONS=('Repeat')
# BANDS=('highgamma')
# DATATYPES=('token')
# PHASES=('Stimulus' 'Delay' 'Go' 'Response')

REF='bipolar'
WINDOW=0.2
STEP=0.02
VARIANCE=0.9
N_PERMUTATIONS=200
N_FOLDS=5
N_JOBS=24

# ── Flatten all combinations into arrays ──────────────
declare -a ALL_SUBJ
declare -a ALL_BAND
declare -a ALL_DESC
declare -a ALL_DATA
declare -a ALL_PHASE

for subj in "${SUBJECTS[@]}"; do
    for band in "${BANDS[@]}"; do
        for desc in "${DESCRIPTIONS[@]}"; do
            for data in "${DATATYPES[@]}"; do
                for phase in "${PHASES[@]}"; do
                    ALL_SUBJ+=("$subj")
                    ALL_BAND+=("$band")
                    ALL_DESC+=("$desc")
                    ALL_DATA+=("$data")
                    ALL_PHASE+=("$phase")
                done
            done
        done
    done
done

TOTAL_JOBS=${#ALL_SUBJ[@]}

# Safety check: if TASK_ID is out of bounds, exit safely
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of bounds (Total combinations: $TOTAL_JOBS)."
    echo "Please update your #SBATCH --array directive to 0-$((TOTAL_JOBS - 1))"
    exit 0
fi

# Extract the specific combination for THIS job
SUBJECT=${ALL_SUBJ[$SLURM_ARRAY_TASK_ID]}
BAND=${ALL_BAND[$SLURM_ARRAY_TASK_ID]}
TYPE=${ALL_DESC[$SLURM_ARRAY_TASK_ID]}
DATATYPE=${ALL_DATA[$SLURM_ARRAY_TASK_ID]}
PHASE=${ALL_PHASE[$SLURM_ARRAY_TASK_ID]}

LOG_BASE="/hpc/group/coganlab/nanlinshi/insula/logs/decoding_resolved_${SUBJECT}_${BAND}_${TYPE}_${PHASE}_${DATATYPE}"

echo "Array task ${SLURM_ARRAY_TASK_ID} out of $((TOTAL_JOBS - 1)) "
echo "Combination: subject=${SUBJECT} band=${BAND} desc=${TYPE} datatype=${DATATYPE} phase=${PHASE}"
echo "Current working directory: $(pwd)"
echo "Python: $(which python) ($(python --version 2>&1))"
echo "Conda env: $CONDA_DEFAULT_ENV"

python src/run_decoding_resolved.py \
    --bids_root "${BIDS_ROOT}" \
    --subject    "${SUBJECT}" \
    --ref        "${REF}" \
    --description "${TYPE}" \
    --phase      "${PHASE}" \
    --band       "${BAND}" \
    --datatype   "${DATATYPE}" \
    --variance   "${VARIANCE}" \
    --window     "${WINDOW}" \
    --step       "${STEP}" \
    --n_perm     "${N_PERMUTATIONS}" \
    --n_folds    "${N_FOLDS}" \
    --n_jobs     "${N_JOBS}" \
    > "${LOG_BASE}.out" \
    2> "${LOG_BASE}.err"

echo "Exit code: $?"