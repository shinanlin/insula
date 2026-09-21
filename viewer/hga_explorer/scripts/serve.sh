#!/bin/bash
#SBATCH --job-name=hga_explorer_web
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/hga_explorer_web_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/hga_explorer_web_%j.err
#SBATCH --partition=coganlab-gpu
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
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
NPM_CACHE="${NPM_CACHE:-/hpc/group/coganlab/nanlinshi/.npm}"
EXPORTED_DATA="${HGA_EXPLORER_DATA:-/hpc/group/coganlab/nanlinshi/insula/viewer/hga_explorer/public/data}"
EXPORTED_ASSETS="${HGA_EXPLORER_ASSETS:-/hpc/group/coganlab/nanlinshi/insula/viewer/hga_explorer/public/assets}"

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

mkdir -p "${PROJECT_ROOT}/logs/slurm"
export npm_config_cache="${NPM_CACHE}"
mkdir -p "${NPM_CACHE}"

cd "${VIEWER_ROOT}"

if [[ ! -d "${VIEWER_ROOT}/node_modules" ]]; then
  echo "node_modules/ missing — installing..."
  npm ci
fi

if [[ ! -f "${DIST_DIR}/index.html" ]]; then
  echo "dist/ not found — building production bundle..."
  npm run build
fi

if [[ ! -f "${DIST_DIR}/index.html" ]]; then
  echo "ERROR: ${DIST_DIR}/index.html missing after build" >&2
  exit 1
fi

if [[ -f "${VIEWER_ROOT}/public/data/manifest.json" ]]; then
  echo "Using local public/data/manifest.json"
elif [[ -f "${EXPORTED_DATA}/manifest.json" ]]; then
  echo "Linking exported dataset: ${EXPORTED_DATA} -> ${DIST_DIR}/data"
  rm -rf "${DIST_DIR}/data"
  ln -sfn "${EXPORTED_DATA}" "${DIST_DIR}/data"
else
  echo "WARNING: no manifest.json; viewer will fall back to mock data"
fi

if [[ -d "${EXPORTED_ASSETS}" ]]; then
  echo "Overlaying GLB meshes from ${EXPORTED_ASSETS}"
  mkdir -p "${DIST_DIR}/assets/native"
  find "${EXPORTED_ASSETS}" -maxdepth 1 -name '*.glb' -exec ln -sfn {} "${DIST_DIR}/assets/" \;
  if [[ -d "${EXPORTED_ASSETS}/native" ]]; then
    find "${EXPORTED_ASSETS}/native" -maxdepth 1 -name '*.glb' -exec ln -sfn {} "${DIST_DIR}/assets/native/" \;
  fi
fi

if ss -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${PORT}$"; then
  echo "ERROR: port ${PORT} is already in use on $(hostname -s)" >&2
  ss -lntp | grep -E "[:.]${PORT}\\b" || true
  exit 1
fi

NODE="$(hostname -s)"
echo "HGA Explorer serving ${DIST_DIR}"
echo "Node: ${NODE}"
echo "Port: ${PORT}"
echo "Bind: 0.0.0.0"
echo ""
echo "From your laptop (via login node):"
echo "  ssh -L ${PORT}:${NODE}:${PORT} ns458@dcc-login.oit.duke.edu"
echo "Or: export HGA_EXPLORER_NODE=${NODE}; bash scripts/connect_tunnel.sh"
echo "Then open: http://localhost:${PORT}/"
echo "See docs/ACCESS.md"
echo ""

cd "${DIST_DIR}"
exec python -m http.server "${PORT}" --bind 0.0.0.0
