#!/bin/bash
# Submit prepare -> validate -> smoke -> five formal arrays -> result census.

set -euo pipefail
cd /hpc/group/coganlab/nanlinshi/insula-functional
mkdir -p logs/slurm

PREP_JOB=$(sbatch --parsable scripts/slurm/prepare_decoding_functional.sh)
VALIDATE_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" scripts/slurm/validate_decoding_functional.sh)
SMOKE_JOB=$(sbatch --parsable --dependency="afterok:${VALIDATE_JOB}" scripts/slurm/smoke_decoding_functional.sh)

WIN_LEX=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" scripts/slurm/decoding_functional_window_lexical_delay.sh)
WIN_PHON=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" scripts/slurm/decoding_functional_window_phoneme_sequence.sh)
RES_LEX=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" scripts/slurm/decoding_functional_resolved_lexical_delay.sh)
RES_PHON=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" scripts/slurm/decoding_functional_resolved_phoneme_sequence.sh)
CROSS=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" scripts/slurm/decoding_functional_cross_generalized.sh)
CENSUS=$(sbatch --parsable --dependency="afterok:${WIN_LEX}:${WIN_PHON}:${RES_LEX}:${RES_PHON}:${CROSS}" scripts/slurm/validate_decoding_functional_results.sh)

echo "prepare=${PREP_JOB} validate=${VALIDATE_JOB} smoke=${SMOKE_JOB}"
echo "window_lexical=${WIN_LEX} window_phoneme=${WIN_PHON}"
echo "resolved_lexical=${RES_LEX} resolved_phoneme=${RES_PHON} cross=${CROSS} census=${CENSUS}"
