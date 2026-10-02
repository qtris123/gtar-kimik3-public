#!/usr/bin/env bash
# Tokenize FineWeb-Edu (sample-10BT) with the Qwen3 tokenizer into data/fineweb_edu.
# Extra arguments are passed on, e.g. `bash scripts/prepare_data.sh --max-tokens 100000000`.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

python -m scripts.prepare_data --out data/fineweb_edu "$@"
