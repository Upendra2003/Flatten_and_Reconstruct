#!/usr/bin/env bash
# Create ./.venv and install dependencies (run once).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "venv ready: source .venv/bin/activate"
