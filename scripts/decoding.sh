#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=/hpc/home/ns458/coganlab/nanlinshi/insula/logs/decoding.out
#SBATCH --error=/hpc/home/ns458/coganlab/nanlinshi/insula/logs/decoding.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common
#SBATCH --chdir=/hpc/home/ns458/coganlab/nanlinshi/insula
#SBATCH --array=0-10

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
module purge
module load CUDA/11.4


VARIANCE=0.85
N_PERM=200
N_FOLDS=10
N_JOBS=10
BANDS=('highgamma')
REF='bipolar'
DESCRIPTIONS=('production' 'perception')


BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"
SUBJECTS=(
  AICl AICr
  PICl PICr
  SMCl SMCr
  IFGl IFGr
  STGl STGr
)
DATATYPES=('phoneme')

# BIDS_ROOT="/cwork/ns458/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS/"
# SUBJECTS=(
#     S14 S16 S18 S22 
#     S23 S26 S32 S33 
#     S36 S39 S57 S58 
#     S62
# )
# DATATYPES=('phoneme(acoustic)')
# BANDS=('mfcc')
# REF='car'

# BIDS_ROOT="/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/"
# DESCRIPTIONS=('JL' 'LM' 'LS')
# DATATYPES=('word')


SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "Processing subject $SUBJECT (array task $SLURM_ARRAY_TASK_ID)"

for BAND in ${BANDS[@]}; do
    for TYPE in ${DESCRIPTIONS[@]}; do
        for DATATYPE in ${DATATYPES[@]}; do
            python src/run_decoding.py \
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
            > /hpc/home/ns458/coganlab/nanlinshi/insula/logs/decoding_${SUBJECT}_${BAND}_${TYPE}_${DATATYPE}.out \
            2> /hpc/home/ns458/coganlab/nanlinshi/insula/logs/decoding_${SUBJECT}_${BAND}_${TYPE}_${DATATYPE}.err
        done
    done
done
