#!/usr/bin/env bash
# Train the 0.6B KimiK3 with MTP on GPUS GPUs (default 8). Extra arguments are passed on to scripts.train,
# e.g. `GPUS=4 bash scripts/train_kimi_k3_0.6B_mtp.sh --wandb kimi-k3-0.6B_mtp`.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

torchrun --standalone --nproc_per_node="${GPUS:-8}" -m scripts.train \
    --config configs/kimi_k3_0.6B_mtp.json --data data/fineweb_edu "$@"
