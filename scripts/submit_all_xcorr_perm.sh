#!/bin/bash
# Submit permutation xcorr for all 5 datasets.
# Auto-counts subjects per dataset and submits a subject-parallel array job.

set -euo pipefail

BAND=${BAND:-highgamma}
REFERENCE=${REFERENCE:-bipolar}
CONCURRENCY=${CONCURRENCY:-10}

# Must match the DATASETS array in run_xcorr_pair_permutation.sh (same order).
BIDS_ROOTS=(
  "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
  "/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/"
  "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/"
  "/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/"
  "/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/"
)
NAMES=(LexicalDelay LexicalNoDelay PhonemeSequence PictureNaming SentenceRep)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for i in "${!BIDS_ROOTS[@]}"; do
  ROOT="${BIDS_ROOTS[$i]}"
  NAME="${NAMES[$i]}"

  N=$(find "${ROOT}derivatives/epoch(${REFERENCE})" \
        -path "*/epoch(band)(sig)(effective)/*${BAND}*.h5" 2>/dev/null \
      | sed -E 's|.*/sub-([^/]+)/.*|\1|' \
      | sort -u | wc -l)

  if [ "$N" -eq 0 ]; then
    echo "[$NAME] no subjects found, skipping."
    continue
  fi

  LAST=$((N - 1))
  echo "[$NAME] idx=$i N=$N  submitting array 0-${LAST}%${CONCURRENCY}"
  sbatch --array=0-${LAST}%${CONCURRENCY} \
    --export=ALL,DATASET_IDX=$i,BAND=$BAND,REFERENCE=$REFERENCE \
    "${SCRIPT_DIR}/run_xcorr_pair_permutation.sh"
done
