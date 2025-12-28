#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=/hpc/group/coganlab/nanlinshi/sharedspace/logs/decoding.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/sharedspace/logs/decoding.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/sharedspace
#SBATCH --array=0-21

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
module purge
module load CUDA/11.4

SUBJECTS=(
  SMCl SMCr
  HGl  HGr
  IFGl IFGr
  INSl INSr
  IPLl IPLr
  ITGl ITGr
  MFGl MFGr
  MTGl MTGr
  SFGl SFGr
  STGl STGr
  STSl STSr
)

# SUBJECTS=(
#   'perception'
#   'production'
#   'sensorimotor'
#   'perceptionOnly'
#   'productionOnly'
# )


SUBJECTS=(
    S14 S16 S18 S22 
    S23 S26 S32 S33 
    S36 S39 S57 S58 
    S62
)

# BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"
BIDS_ROOT="/cwork/ns458/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS/"
DESCRIPTIONS=('perception' 'production')
# DATATYPES=('phoneme')
# BANDS=('highgamma')

DATATYPES=('phoneme(acoustic)')
BANDS=('mfcc')

REF='car'

# BIDS_ROOT="/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/"
# DESCRIPTIONS=('JL' 'LM' 'LS')
# DATATYPES=('word')

VARIANCE=0.85
N_PERM=200
N_FOLDS=10
N_JOBS=10

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "Processing subject $SUBJECT (array task $SLURM_ARRAY_TASK_ID)"

for BAND in ${BANDS[@]}; do
    for TYPE in ${DESCRIPTIONS[@]}; do
        for DATATYPE in ${DATATYPES[@]}; do
            python src/decoding.py \
                --bids_root "${BIDS_ROOT}" \
                --subject ${SUBJECT} \
                --description ${TYPE} \
                --band ${BAND} \
                --datatype ${DATATYPE} \
                --variance ${VARIANCE} \
                --n_perm ${N_PERM} \
                --n_folds ${N_FOLDS} \
                --n_jobs ${N_JOBS} \
                --ref ${REF} \
            > /hpc/group/coganlab/nanlinshi/sharedspace/logs/decoding_${SUBJECT}_${BAND}_${TYPE}_${DATATYPE}.out \
            2> /hpc/group/coganlab/nanlinshi/sharedspace/logs/decoding_${SUBJECT}_${BAND}_${TYPE}_${DATATYPE}.err
        done
    done
done
