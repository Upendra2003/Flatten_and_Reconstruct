#!/usr/bin/env bash
# Run all flatten/reconstruct experiments.  Extra args go to run_experiments.py
#   ./scripts/run.sh                 # default B=8, loop + vectorized
#   ./scripts/run.sh --batch 16 --mode loop
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -x .venv/bin/python ]] || ./scripts/setup_venv.sh
mkdir -p logs results
export PYTHONUNBUFFERED=1
.venv/bin/python src/run_experiments.py "$@" 2>&1 | tee logs/run.log
