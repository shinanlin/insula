#!/bin/bash
# Submit INS whole-window Haufe pattern production (LexicalDelay + PhonemeSequence).

set -eo pipefail
cd /hpc/group/coganlab/nanlinshi/insula-functional
mkdir -p logs/slurm results/decoding

LEX_JOB=$(sbatch --parsable scripts/slurm/decoding_insula_patterns_lexical_delay.sh)
PHON_JOB=$(sbatch --parsable scripts/slurm/decoding_insula_patterns_phoneme_sequence.sh)
CENSUS_JOB=$(sbatch --parsable --dependency=afterok:${LEX_JOB}:${PHON_JOB} \
  scripts/slurm/validate_insula_pattern_results.sh)

echo "lexical_delay_patterns=${LEX_JOB}"
echo "phoneme_sequence_patterns=${PHON_JOB}"
echo "pattern_census=${CENSUS_JOB}"
echo "Expected outputs: 48 LexicalDelay + 16 PhonemeSequence pattern H5 files"
