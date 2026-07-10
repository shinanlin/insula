#!/usr/bin/env bash
# Run on your laptop from viewer/hga_explorer after sbatch scripts/serve.sh.
# Forwards localhost:18081 to the compute node running the viewer.

set -eo pipefail

PORT="${HGA_EXPLORER_PORT:-18081}"
LOGIN_NODE="${HGA_EXPLORER_LOGIN:-dcc-login.oit.duke.edu}"
USER_NAME="${HGA_EXPLORER_USER:-ns458}"

if [[ -z "${HGA_EXPLORER_NODE:-}" ]]; then
  echo "Set HGA_EXPLORER_NODE to the compute node from the Slurm serve log, e.g.:"
  echo "  export HGA_EXPLORER_NODE=dcc-cognac-01"
  echo "  bash scripts/connect_tunnel.sh"
  exit 1
fi

echo "Tunneling localhost:${PORT} -> ${HGA_EXPLORER_NODE}:${PORT} via ${LOGIN_NODE}"
exec ssh -N -L "${PORT}:${HGA_EXPLORER_NODE}:${PORT}" "${USER_NAME}@${LOGIN_NODE}"
