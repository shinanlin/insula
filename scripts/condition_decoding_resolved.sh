#!/bin/bash

#SBATCH --job-name=condition_decoding_resolved
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/condition_decoding_resolved.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/condition_decoding_resolved.err
#SBATCH --time=7-00:00:00
#SBATCH --mem=36G
#SBATCH --cpus-per-task=24
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-6

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
module purge
module load CUDA/11.4

ROIS=(
  AIC
  PIC
  SMC
  IFG
  STG
)

HEMIS=(L R B)

BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
BANDS=('highgamma')
PHASES=('Stimulus' 'Go' 'Response')

REF='bipolar'
WINDOW=0.2
STEP=0.02
VARIANCE=0.8
N_PERMUTATIONS=200
N_FOLDS=10
N_JOBS=24

ROI=${ROIS[$SLURM_ARRAY_TASK_ID]}

echo "Processing ROI ${ROI} (array task $SLURM_ARRAY_TASK_ID)"

for HEMI in ${HEMIS[@]}; do
    for BAND in ${BANDS[@]}; do
        for PHASE in ${PHASES[@]}; do
                python src/decoding/condition_decoding.py \
                --bids_root "${BIDS_ROOT}" \
                --roi ${ROI} \
                --hemi ${HEMI} \
                --ref ${REF} \
                --phase ${PHASE} \
                --band ${BAND} \
                --variance ${VARIANCE} \
                --window ${WINDOW} \
                --step ${STEP} \
                --n_perm ${N_PERMUTATIONS} \
                --n_folds ${N_FOLDS} \
                --n_jobs ${N_JOBS} \
                > /hpc/group/coganlab/nanlinshi/insula/logs/condition_decoding_resolved_${ROI}_${HEMI}_${BAND}_${PHASE}.out \
                2> /hpc/group/coganlab/nanlinshi/insula/logs/condition_decoding_resolved_${ROI}_${HEMI}_${BAND}_${PHASE}.err
                echo "Exit code: $?"
        done
    done
done
