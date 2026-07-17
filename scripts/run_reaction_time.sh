#!/bin/bash
# Legacy entrypoint — prefer the per-task Slurm wrappers under scripts/slurm/:
#   sbatch scripts/slurm/rt_lexical_delay.sh
#   sbatch scripts/slurm/rt_lexical_nodelay.sh
#   sbatch scripts/slurm/rt_phoneme_sequence.sh
#   sbatch scripts/slurm/rt_picture_naming.sh
#
# Output convention (hammers): results/{Task}(bipolar)(hammers)/RT/...
set -eo pipefail
TASK="${TASK:-LexicalNoDelay}"
exec bash "$(dirname "$0")/slurm/run_reaction_time.sh"
