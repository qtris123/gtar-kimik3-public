#!/usr/bin/env bash
# Train the 0.6B KimiK3 (KDA + GQA hybrid) on GPUS GPUs (default 8). Extra arguments are passed on to scripts.train,
# e.g. `GPUS=4 bash scripts/train_kimi_k3_0.6B.sh --wandb kimi-k3-0.6B`.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/envs/kimi/bin/activate

# Google Cloud's VM defaults NCCL_NET to gIB, which is Google's proprietary InfiniBand/Falcon NIC network plugin
# However, this machine doesn't have it - returns Last error: Failed to initialize any NET plugin
# => Disable it, to fallback to the NV18
unset NCCL_NET
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//\/usr\/local\/gib\/lib64:/}"
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

nvidia-smi

torchrun --standalone --nproc_per_node="${GPUS:-2}" -m scripts.train \
    --config configs/kimi_k3_tiny.json --data data/fineweb_edu "$@" --wandb kimi-k3-tiny-adjusted-redo
