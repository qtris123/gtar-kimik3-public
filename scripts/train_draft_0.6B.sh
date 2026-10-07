#!/usr/bin/env bash
# Fine-tune the 0.6B MTP draft layer against a frozen target checkpoint on GPUS GPUs (default 8).
# Usage:
#   bash scripts/train_draft_0.6B.sh out/kimi_k3_0.6B_mtp/ckpt_019073.pt
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

if [ $# -eq 0 ]; then
    echo "Error: Stage-1 checkpoint path required"
    echo "Usage: bash scripts/train_draft_0.6B.sh <target_checkpoint_path> [extra args...]"
    exit 1
fi

TARGET_CKPT="$1"
shift

torchrun --standalone --nproc_per_node="${GPUS:-8}" -m scripts.train_draft \
    --config configs/kimi_k3_0.6B_mtp.json \
    --target-checkpoint "${TARGET_CKPT}" \
    --data data/fineweb_edu "$@"
