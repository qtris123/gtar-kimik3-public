#!/usr/bin/env bash
# Train the 0.6B KimiK3 with MTP on GPUS GPUs (default 4). Extra arguments are passed on to scripts.train,
# e.g. `GPUS=4 bash scripts/train_kimi_k3_0.6B_mtp.sh --wandb kimi-k3-0.6B_mtp`.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Google Cloud's VM defaults NCCL_NET to gIB, which is Google's proprietary InfiniBand/Falcon NIC network plugin
# However, this machine doesn't have it - returns Last error: Failed to initialize any NET plugin
# => Disable it, to fallback to the NVLink
unset NCCL_NET
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//\/usr\/local\/gib\/lib64:/}"
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

torchrun --standalone --nproc_per_node="${GPUS:-4}" -m scripts.train \
    --config configs/kimi_k3_0.6B_mtp.json \
    --data data/fineweb_edu \
    --out out/kimi_k3_0.6B_mtp \
    --save-steps 0,1000,2000,5000,8000,9000,10000,11000,12000 \
    --save-total-limit 0 \
    --wandb kimi-k3-0.6B_mtp \
    "$@"
