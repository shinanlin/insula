#!/bin/bash
#SBATCH --job-name=package
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/package_%s.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/package_%s.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula/src
#SBATCH --array=0  # Process all 5 bands, max 5 at a time

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

# Configuration
BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"
SCRIPT_DIR="/hpc/group/coganlab/nanlinshi/insula/src"

BANDS=('highgamma')

BAND=${BANDS[$SLURM_ARRAY_TASK_ID]}

echo "Processing band $BAND (array task $SLURM_ARRAY_TASK_ID)"

DESCRIPTIONS=('production' 'perception')

for DESCRIPTION in ${DESCRIPTIONS[@]}; do
    python "${SCRIPT_DIR}/package_HGA.py" \
        --bids_root "${BIDS_ROOT}" \
        --band ${BAND} \
        --description ${DESCRIPTION} \
        > /hpc/group/coganlab/nanlinshi/insula/logs/package_HGA_${BAND}_${DESCRIPTION}.out \
        2> /hpc/group/coganlab/nanlinshi/insula/logs/package_HGA_${BAND}_${DESCRIPTION}.err

    python "${SCRIPT_DIR}/package_stats.py" \
        --bids_root "${BIDS_ROOT}" \
        --band ${BAND} \
        --description ${DESCRIPTION} \
        > /hpc/group/coganlab/nanlinshi/insula/logs/package_stats_${BAND}_${DESCRIPTION}.out \
        2> /hpc/group/coganlab/nanlinshi/insula/logs/package_stats_${BAND}_${DESCRIPTION}.err

    python "${SCRIPT_DIR}/package_ave_cord.py" \
        --bids_root "${BIDS_ROOT}" \
        --band ${BAND} \
        --description ${DESCRIPTION} \
        > /hpc/group/coganlab/nanlinshi/insula/logs/package_ave_cord_${BAND}_${DESCRIPTION}.out \
        2> /hpc/group/coganlab/nanlinshi/insula/logs/package_ave_cord_${BAND}_${DESCRIPTION}.err
done

echo "Finished processing band ${BAND}"