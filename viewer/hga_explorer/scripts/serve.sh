#!/bin/bash
#SBATCH --job-name=hga_explorer_web
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/hga_explorer_web_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/hga_explorer_web_%j.err
#SBATCH --partition=coganlab-gpu
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -eo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/package.json" ]]; then
  VIEWER_ROOT="${SLURM_SUBMIT_DIR}"
else
  VIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
PROJECT_ROOT="$(cd "${VIEWER_ROOT}/../.." && pwd)"
DIST_DIR="${VIEWER_ROOT}/dist"
PORT="${HGA_EXPLORER_PORT:-18081}"

source ~/.bashrc
conda activate ieeg

mkdir -p "${PROJECT_ROOT}/logs/slurm"

if [[ ! -f "${DIST_DIR}/index.html" ]]; then
  echo "dist/ not found — building production bundle..."
  cd "${VIEWER_ROOT}"
  npm run build
fi

if [[ ! -f "${DIST_DIR}/index.html" ]]; then
  echo "ERROR: ${DIST_DIR}/index.html missing after build" >&2
  exit 1
fi

NODE="$(hostname -s)"
echo "HGA Explorer serving ${DIST_DIR}"
echo "Node: ${NODE}"
echo "Port: ${PORT}"
echo ""
echo "From your laptop (via login node):"
echo "  ssh -L ${PORT}:${NODE}:${PORT} ns458@dcc-login.oit.duke.edu"
echo "Or: bash scripts/connect_tunnel.sh  (run on laptop from viewer/hga_explorer)"
echo "Then open: http://localhost:${PORT}/"
echo "See docs/ACCESS.md"
echo ""

cd "${DIST_DIR}"
exec python -m http.server "${PORT}" --bind 0.0.0.0
