#!/bin/bash
#SBATCH --job-name=export_img_svg
#SBATCH --output=logs/slurm/export_notebook_svgs_%j.out
#SBATCH --error=logs/slurm/export_notebook_svgs_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -eo pipefail

PROJECT_ROOT="/hpc/group/coganlab/nanlinshi/insula-functional"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm img/{nmf,functional_hga,insula_patterns,univariate,decode,decode_functional,modulation,connectivity}

source ~/.bashrc
conda activate ieeg

export PYVISTA_OFF_SCREEN=true
export MNE_3D_BACKEND=notebook
export MPLBACKEND=Agg

python scripts/export_notebook_svgs.py

# Pre-fix invalid stream outputs (missing name) from prior notebook edits.
python - <<'PY'
import json
from pathlib import Path
for name in [
    "notebooks/nmf.ipynb",
    "notebooks/modulation.ipynb",
    "notebooks/decode.ipynb",
    "notebooks/decode_functional.ipynb",
    "notebooks/connectivity.ipynb",
]:
    path = Path(name)
    nb = json.loads(path.read_text())
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []) or []:
            if out.get("output_type") == "stream" and "name" not in out:
                out["name"] = "stdout"
            if out.get("output_type") in {"display_data", "execute_result"} and "metadata" not in out:
                out["metadata"] = {}
    path.write_text(json.dumps(nb, indent=2) + "\n")
print("pre-fixed notebook outputs")
PY

# Execute notebooks that write SVG (does not overwrite source notebooks).
# Failures in one notebook should not abort the rest.
set +e
OUT_NB_DIR="logs/nbconvert_exports"
mkdir -p "$OUT_NB_DIR"
for nb in \
  notebooks/nmf.ipynb \
  notebooks/modulation.ipynb \
  notebooks/decode.ipynb \
  notebooks/decode_functional.ipynb \
  notebooks/connectivity.ipynb
do
  base="$(basename "$nb" .ipynb)"
  echo "===== EXECUTE $nb ====="
  jupyter nbconvert --to notebook --execute \
    --output="${base}_executed.ipynb" \
    --output-dir="$OUT_NB_DIR" \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=python3 \
    "$nb"
  if [[ $? -eq 0 ]]; then echo "OK $nb"; else echo "FAIL $nb"; fi
done
set -e

echo "===== IMG TREE ====="
find img -type f | sort
