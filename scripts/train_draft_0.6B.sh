#!/usr/bin/env bash
# Fine-tune the 0.6B MTP draft layer against a frozen target checkpoint on GPUS GPUs (default 4).
# Usage:
#   bash scripts/train_draft_0.6B.sh out/kimi_k3_0.6B_mtp/ckpt_012000.pt
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

if [ $# -eq 0 ]; then
    echo "Error: Stage-1 checkpoint path required"
    echo "Usage: bash scripts/train_draft_0.6B.sh <target_checkpoint_path>"
    echo "Example: bash scripts/train_draft_0.6B.sh out/kimi_k3_0.6B_mtp/ckpt_012000.pt"
    exit 1
fi

TARGET_CKPT="$1"
shift

unset NCCL_NET
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//\/usr\/local\/gib\/lib64:/}"
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

torchrun --standalone --nproc_per_node="${GPUS:-4}" -m scripts.train_stage2 \
    --config configs/kimi_k3_0.6B_mtp.json \
    --target-checkpoint "${TARGET_CKPT}" \
    --data data/fineweb_edu \
    --out out/kimi_k3_0.6B_mtp_draft \
    --wandb kimi-k3-0.6B_mtp_draft \
    "$@"
