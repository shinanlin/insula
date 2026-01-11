#!/bin/bash

#SBATCH --job-name=decoding_resolved
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/decoding_resolved.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/decoding_resolved.err
#SBATCH --time=7-00:00:00
#SBATCH --mem=36G
#SBATCH --cpus-per-task=24
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-11

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
module purge
module load CUDA/11.4

SUBJECTS=(
  AICl AICr
  PICl PICr
  SMCl SMCr
  IFGl IFGr
  STGl STGr
)

# BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"
# DESCRIPTIONS=('Repeat')
# BANDS=('highgamma')
# DATATYPES=('phoneme')
# PHASES=('Stimulus' 'Go' 'Response')


# BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
# DESCRIPTIONS=('Repeat')
# BANDS=('highgamma')
# DATATYPES=('word')

BIDS_ROOT="/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/"
DESCRIPTIONS=('Repeat' 'Passive')
BANDS=('highgamma')
DATATYPES=('token')
PHASES=('Stimulus' 'Go' 'Response')

REF='bipolar'
WINDOW=0.2
STEP=0.02
VARIANCE=0.8
N_PERMUTATIONS=200
N_FOLDS=10
N_JOBS=24

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
echo "Processing subject $SUBJECT (array task $SLURM_ARRAY_TASK_ID)"

# Debug environment
echo "Current working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "BIDS_ROOT: ${BIDS_ROOT}"
echo "ROI: ${SUBJECT}"

for BAND in ${BANDS[@]}; do
    for TYPE in ${DESCRIPTIONS[@]}; do
        for DATATYPE in ${DATATYPES[@]}; do
            for PHASE in ${PHASES[@]}; do
                python src/run_decoding_resolved.py \
                --bids_root "${BIDS_ROOT}" \
                --subject ${SUBJECT} \
                --ref ${REF} \
                --description ${TYPE} \
                --phase ${PHASE} \
                --band ${BAND} \
                --datatype ${DATATYPE} \
                --variance ${VARIANCE} \
                --window ${WINDOW} \
                --step ${STEP} \
                --n_perm ${N_PERMUTATIONS} \
                --n_folds ${N_FOLDS} \
                --n_jobs ${N_JOBS} \
                > /hpc/group/coganlab/nanlinshi/insula/logs/decoding_resolved_${SUBJECT}_${BAND}_${TYPE}_${PHASE}_${DATATYPE}.out \
                2> /hpc/group/coganlab/nanlinshi/insula/logs/decoding_resolved_${SUBJECT}_${BAND}_${TYPE}_${PHASE}_${DATATYPE}.err
                echo "Exit code: $?"
            done
        done
    done
done