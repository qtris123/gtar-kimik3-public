#!/usr/bin/env bash
# Fine-tune the 0.6B MTP draft layer with custom fused layers [0, 4, 27] on 4 GPUs.
#
# Usage:
#   bash scripts/train_draft_0.6B_fused_0_4_27.sh
#   bash scripts/train_draft_0.6B_fused_0_4_27.sh /scratch/out/kimi_k3_0.6B_mtp/ckpt_019073.pt
#
set -euo pipefail
cd "$(dirname "$0")/.."

# Auto-link virtual environment and data/scratch directories if missing
if [ ! -d ".venv" ] && [ -d "/home/gpuuser/.venv" ]; then
    ln -sfn /home/gpuuser/.venv .venv
fi
if [ ! -e "data" ] && [ -d "/scratch/data" ]; then
    ln -sfn /scratch/data data
fi
if [ ! -e "out" ] && [ -d "/scratch/out" ]; then
    ln -sfn /scratch/out out
fi

source .venv/bin/activate

if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
    TARGET_CKPT="$1"
    shift
else
    TARGET_CKPT="${TARGET_CKPT:-/scratch/out/kimi_k3_0.6B_mtp/ckpt_019073.pt}"
fi
OUT_DIR="${OUT_DIR:-/scratch/out/kimi_k3_0.6B_mtp_draft_0_4_27}"
WANDB_RUN="${WANDB_RUN:-kimi-k3-0.6B_mtp_draft_0_4_27}"

# Google Cloud VM NCCL settings (fallback to NVLink)
unset NCCL_NET
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//\/usr\/local\/gib\/lib64:/}"
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

GPUS="${GPUS:-4}"

echo "======================================================================"
echo " Starting Stage 2 Draft Training with Fused Layers [0, 4, 27]"
echo " System:            gstar-kimi3-public (latest clean implementation)"
echo " Target Checkpoint: ${TARGET_CKPT}"
echo " Output Directory:  ${OUT_DIR}"
echo " WandB Run Name:    ${WANDB_RUN}"
echo " GPUs:              ${GPUS}"
echo "======================================================================"

torchrun --standalone --nproc_per_node="${GPUS}" -m scripts.train_draft \
    --config configs/kimi_k3_0.6B_mtp.json \
    --target-checkpoint "${TARGET_CKPT}" \
    --feature-layers "0,4,27" \
    --data data/fineweb_edu \
    --out "${OUT_DIR}" \
    --wandb "${WANDB_RUN}" \
    "$@"
