#!/usr/bin/env bash
# Fine-tune the 0.6B MTP draft layer against a frozen target checkpoint on GPUS GPUs (default 4).
# Usage:
#   bash scripts/train_draft_0.6B.sh out/kimi_k3_0.6B_mtp/ckpt_012000.pt
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

TARGET_CKPT="${1:-out/kimi_k3_0.6B_mtp/ckpt_012000.pt}"
shift || true

unset NCCL_NET
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//\/usr\/local\/gib\/lib64:/}"
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

torchrun --standalone --nproc_per_node="${GPUS:-4}" -m scripts.train_draft \
    --config configs/kimi_k3_0.6B_mtp.json \
    --target-checkpoint "${TARGET_CKPT}" \
    --data data/fineweb_edu \
    --out out/kimi_k3_0.6B_mtp_draft \
    --wandb kimi-k3-0.6B_mtp_draft \
    "$@"
