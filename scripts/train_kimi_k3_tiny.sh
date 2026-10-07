#!/usr/bin/env bash
# Train the tiny KimiK3 (KDA + GQA hybrid) on GPUS GPUs (default 2). Extra arguments are passed on to scripts.train,
# e.g. `GPUS=1 bash scripts/train_kimi_k3_tiny.sh --wandb kimi-k3-tiny`.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

torchrun --standalone --nproc_per_node="${GPUS:-2}" -m scripts.train \
    --config configs/kimi_k3_tiny.json --data data/fineweb_edu "$@"
