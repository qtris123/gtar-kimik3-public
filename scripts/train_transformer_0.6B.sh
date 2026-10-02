#!/usr/bin/env bash
# Train the 0.6B Transformer baseline on GPUS GPUs (default 8). Extra arguments are passed on to scripts.train,
# e.g. `GPUS=4 bash scripts/train_transformer_0.6B.sh --wandb transformer-0.6B`.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

torchrun --standalone --nproc_per_node="${GPUS:-8}" -m scripts.train \
    --config configs/transformer_0.6B.json --data data/fineweb_edu "$@"
