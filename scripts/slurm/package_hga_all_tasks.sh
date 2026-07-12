#!/bin/bash
#SBATCH --job-name=pkg_hga_atlas
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/package_hga_atlas_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/package_hga_atlas_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

run_one() {
  local bids_root="$1"
  local name="$2"
  echo "===== ${name} ====="
  echo "bids_root=${bids_root}"
  python src/hga/package_highgamma.py --bids_root "${bids_root}" --band highgamma --ref bipolar --atlas aparc2009s
  python src/hga/package_highgamma.py --bids_root "${bids_root}" --band highgamma --ref bipolar --atlas hammers
}

run_one "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/" "LexicalDelay"
run_one "/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/" "LexicalNoDelay"
run_one "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/" "PhonemeSequencing"
run_one "/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/" "PictureNaming"
run_one "/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/" "SentenceRep"

echo "All done."
