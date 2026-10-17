#!/bin/bash
# Refresh the existing Globus CLI session for Duke's DCC collection.
# Complete browser authentication and enter its authorization code in this
# terminal. Do not paste credentials or authorization codes into chat.
set -euo pipefail
export PYTHONPATH=/cwork/ns458/.globus-cli
exec /hpc/home/ns458/miniconda3/bin/python -m globus_cli session update duke.edu --no-local-server
