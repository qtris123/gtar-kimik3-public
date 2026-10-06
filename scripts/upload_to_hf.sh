#!/usr/bin/env bash
# Helper wrapper to run upload_to_hf.py inside the project virtual environment
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "/home/gpuuser/.venv/bin/activate" ]; then
    source /home/gpuuser/.venv/bin/activate
fi

python scripts/upload_to_hf.py "$@"
