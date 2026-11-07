#!/usr/bin/env bash
# Vite dev server plus the anatomy review API. The API stays on 127.0.0.1;
# Vite proxies /api, so one SSH tunnel to the Vite port is enough.

set -eo pipefail

VIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${ANATOMY_API_PORT:-18082}"
DEV_PORT="${HGA_EXPLORER_DEV_PORT:-5173}"

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

cd "${VIEWER_ROOT}"

if [[ ! -d node_modules ]]; then
  echo "node_modules/ missing — installing..."
  npm ci
fi

if ss -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${API_PORT}$"; then
  echo "ERROR: anatomy API port ${API_PORT} is already in use" >&2
  exit 1
fi
if ss -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${DEV_PORT}$"; then
  echo "ERROR: Vite port ${DEV_PORT} is already in use" >&2
  exit 1
fi

python scripts/serve_anatomy.py --host 127.0.0.1 --port "${API_PORT}" &
API_PID=$!
cleanup() {
  kill "${API_PID}" 2>/dev/null || true
}
trap cleanup EXIT

export ANATOMY_API_PORT="${API_PORT}"
echo "Anatomy API: http://127.0.0.1:${API_PORT}"
echo "Explorer:    http://$(hostname -s):${DEV_PORT}/"
exec npm run dev -- --host 0.0.0.0 --port "${DEV_PORT}"
