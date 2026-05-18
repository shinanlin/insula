#!/bin/bash

# Load environment
source ~/.bashrc
conda activate ieeg
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Define parameters
ROIS=('AICl' 'SMCl' 'STGl' 'PICl')
PHASES=('Stimulus' 'Delay' 'Go' 'Response')
DESCS=('Repeat' 'Decision')

cnt=1
total=$(( ${#ROIS[@]} * ${#PHASES[@]} * ${#DESCS[@]} ))

for roi in "${ROIS[@]}"; do
    for phase in "${PHASES[@]}"; do
        for desc in "${DESCS[@]}"; do
            echo "=========================================="
            echo "Running Pattern Extraction [$cnt/$total]"
            echo "ROI: $roi"
            echo "Phase: $phase"
            echo "Description: $desc"
            echo "=========================================="

            python src/run_decoder_patterns_resolved.py \
                --roi $roi \
                --phase $phase \
                --description $desc \
                --window 0.2 \
                --step 0.1 \
                --n_folds 10
            
            cnt=$((cnt+1))
        done
    done
done
echo "All pattern extraction jobs completed."
