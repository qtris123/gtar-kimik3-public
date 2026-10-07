"""
Stage 2: Fine-tune the MTP block as an EAGLE-style recursive drafter against a frozen target.

python -m scripts.train_draft --config configs/kimi_k3_tiny_mtp.json --data data/test \
    --target-checkpoint out/kimi_k3_tiny_mtp/ckpt_000020.pt --no-compile

torchrun --standalone --nproc_per_node=4 -m scripts.train_draft \
    --config configs/kimi_k3_tiny_mtp.json --data data/test \
    --target-checkpoint out/kimi_k3_tiny_mtp/ckpt_final.pt

The target model is loaded, frozen, and set to eval mode. Only the MTP block
and feature projection parameters are trained using LK loss.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from src.dataset import TokenLoader
from src.common import (
    all_reduce_mean,
    init_distributed,
    print0,
)
from src.models import ARCHITECTURES
from src.models.mtp import MTPBlock
from src.training.draft import DraftConfig, DraftTrainer
from src.optim import get_lr

parser = argparse.ArgumentParser(description="Stage 2: Fine-tune MTP block as drafter")
parser.add_argument("--config", required=True, help="json config (same arch as Stage 1)")
parser.add_argument("--target-checkpoint", required=True, help="Stage 1 checkpoint with target + MTP weights")
parser.add_argument("--data", default="data/fineweb_edu")
parser.add_argument("--out", default=None, help="checkpoint directory (default: out/<config>_draft)")
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--total-batch-size", type=int, default=524_288, help="tokens per optimizer step")
parser.add_argument("--total-tokens", type=float, default=1e9, help="training horizon in tokens")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--min-lr-ratio", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--warmup-steps", type=int, default=200)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--eval-every", type=int, default=200)
parser.add_argument("--eval-tokens", type=int, default=10 * 524_288)
parser.add_argument("--save-every", type=int, default=1000)
parser.add_argument("--save-total-limit", type=int, default=3)
parser.add_argument("--log-every", type=int, default=10)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument("--draft-steps", type=int, default=4, help="number of recursive unroll steps")
parser.add_argument("--step-loss-weighting", default="decay", choices=["uniform", "decay"])
parser.add_argument("--step-loss-decay", type=float, default=0.8)
parser.add_argument("--feature-layers", type=str, default=None, help="comma-separated feature layer indices for fusion (e.g. '0,4,27')")
parser.add_argument("--resume", default=None, help="draft checkpoint to resume from")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--wandb", default=None, help="wandb run name")
parser.add_argument("--seed", type=int, default=42)

config_file = json.loads(Path(parser.parse_known_args()[0].config).read_text())
parser.set_defaults(**config_file.get("draft_train", {}))
args = parser.parse_args()

rank, world_size, device = init_distributed()
master_process = rank == 0
torch.manual_seed(args.seed)
torch.set_float32_matmul_precision("high")
autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
out_dir = Path(args.out or f"out/{Path(args.config).stem}_draft")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Load target model from Stage 1 checkpoint ---
print0("Loading target model from Stage 1 checkpoint...")
target_ckpt = torch.load(args.target_checkpoint, map_location="cpu")
target_config_dict = target_ckpt["config"]
target_step = target_ckpt.get("step", "unknown")
print0(f"Target checkpoint step: {target_step}")

# Build vocab size from data
train_loader = TokenLoader(args.data, "train", args.device_batch_size, args.seq_len, rank, world_size, device)
build_val_loader = lambda: TokenLoader(args.data, "val", args.device_batch_size, args.seq_len, rank, world_size, device)
vocab_size = math.ceil(train_loader.meta["vocab_size"] / 128) * 128

# Build target model config
Config, ForCausalLM = ARCHITECTURES[config_file["arch"]]
target_config = Config(**target_config_dict)

if args.feature_layers:
    target_config.feature_layer_indices = [int(x.strip()) for x in args.feature_layers.split(",")]
    print0(f"Feature layer indices (from --feature-layers): {target_config.feature_layer_indices}")
elif not target_config.feature_layer_indices:
    n = target_config.num_hidden_layers
    target_config.feature_layer_indices = [
        max(0, n // 6),
        n // 2,
        n - 1,
    ]
    print0(f"Auto-selected feature layer indices: {target_config.feature_layer_indices}")
else:
    print0(f"Feature layer indices from config/checkpoint: {target_config.feature_layer_indices}")

with torch.device(device):
    target_model = ForCausalLM(target_config)
target_model.load_state_dict(target_ckpt["model"], strict=True)

# Freeze target model
target_model.eval()
for p in target_model.parameters():
    p.requires_grad_(False)
print0(f"Target model loaded and frozen. Params: {sum(p.numel() for p in target_model.parameters()):,}")

# --- Extract MTP block from target and create DraftTrainer ---
assert target_model.mtp_block is not None, (
    "Target checkpoint does not have an MTP block. "
    "Stage 2 requires a Stage 1 checkpoint with mtp_enabled=True."
)

draft_mtp = MTPBlock(
    hidden_size=target_config.hidden_size,
    intermediate_size=target_config.intermediate_size,
    num_attention_heads=target_config.num_attention_heads,
    num_key_value_heads=target_config.num_key_value_heads,
    head_dim=target_config.head_dim,
    rms_norm_eps=target_config.rms_norm_eps,
).to(device)

# Initialize from Stage 1 MTP weights
mtp_state = {k.replace("mtp_block.", ""): v for k, v in target_ckpt["model"].items() if k.startswith("mtp_block.")}
draft_mtp.load_state_dict(mtp_state, strict=True)
print0(f"Draft MTP block initialized from Stage 1. Params: {sum(p.numel() for p in draft_mtp.parameters()):,}")

draft_config = DraftConfig(
    draft_steps=args.draft_steps,
    feature_layer_indices=target_config.feature_layer_indices,
    step_loss_weighting=args.step_loss_weighting,
    step_loss_decay=args.step_loss_decay,
)
draft_trainer = DraftTrainer(draft_mtp, draft_config).to(device)
print0(f"Draft trainer created. Trainable params: {sum(p.numel() for p in draft_trainer.get_trainable_parameters()):,}")

trainable_params = draft_trainer.get_trainable_parameters()
decay_params = [p for p in trainable_params if p.dim() >= 2]
no_decay_params = [p for p in trainable_params if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=args.lr,
    betas=(0.9, 0.95),
    fused=device.type == "cuda",
)

step = 0
if args.resume:
    print0(f"Resuming draft training from {args.resume}")
    draft_ckpt = torch.load(args.resume, map_location="cpu")
    draft_trainer.load_state_dict(draft_ckpt["draft_trainer"])
    optimizer.load_state_dict(draft_ckpt["optimizer"])
    train_loader.load_state_dict(draft_ckpt["loader"])
    step = draft_ckpt["step"]
    print0(f"Resumed at step {step}")

raw_draft_trainer = draft_trainer
if world_size > 1:
    draft_trainer = DistributedDataParallel(
        draft_trainer,
        device_ids=[device.index] if device.type == "cuda" else None,
    )
if args.compile:
    draft_trainer = torch.compile(draft_trainer)

tokens_per_micro_step = args.device_batch_size * args.seq_len * world_size
assert args.total_batch_size % tokens_per_micro_step == 0
grad_accum_steps = args.total_batch_size // tokens_per_micro_step
total_iterations = int(args.total_tokens) // args.total_batch_size
max_iterations = min(total_iterations, args.max_steps) if args.max_steps is not None else total_iterations
eval_steps = max(1, args.eval_tokens // tokens_per_micro_step)
print0(f"Tokens/step: {args.total_batch_size:,} | grad accum: {grad_accum_steps} | iterations: {max_iterations:,}")

wandb_run = None
if args.wandb and master_process:
    import wandb
    wandb_run = wandb.init(
        name=args.wandb,
        config={**vars(args), **asdict(target_config), **asdict(draft_config)},
    )


def save_draft_checkpoint(path, draft_trainer_module, optimizer, loader, step):
    checkpoint = {
        "draft_trainer": draft_trainer_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loader": loader.state_dict(),
        "step": step,
        "target_config": asdict(target_config),
        "draft_config": asdict(draft_config),
        "target_checkpoint": str(args.target_checkpoint),
    }
    torch.save(checkpoint, path)


@torch.no_grad()
def evaluate_draft(draft_trainer_module, target_model, loader, steps):
    draft_trainer_module.eval()
    total_loss = torch.zeros((), device=next(draft_trainer_module.parameters()).device)
    actual_steps = 0

    for _ in range(steps):
        inputs, targets = next(loader)
        with autocast:
            result = target_model(inputs, targets=None, return_features=True)
            target_logits = result["logits"]
            target_features = result["features"]

            draft_result = draft_trainer_module(
                target_features=target_features,
                target_logits=target_logits,
                targets=targets,
                embed_fn=target_model.model.embed_tokens,
                lm_head_fn=target_model.lm_head,
            )

        total_loss += draft_result["loss"].detach()
        actual_steps += 1

    if actual_steps == 0:
        draft_trainer_module.train()
        return 0.0

    avg_loss = all_reduce_mean(total_loss / actual_steps).item()
    draft_trainer_module.train()
    return avg_loss


while True:
    last_step = step == max_iterations

    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        val_loss = evaluate_draft(
            raw_draft_trainer, target_model, build_val_loader(), eval_steps
        )
        print0(f"step {step:6d} | val draft loss {val_loss:.4f}")
        if wandb_run:
            wandb_run.log({"step": step, "val/draft_loss": val_loss})

    should_save = step > 0 and args.save_every > 0 and step % args.save_every == 0
    if master_process and (last_step or should_save):
        ckpt_path = out_dir / f"draft_{step:06d}.pt"
        save_draft_checkpoint(ckpt_path, raw_draft_trainer, optimizer, train_loader, step)
        print0(f"Saved draft checkpoint to {ckpt_path}")
        if args.save_total_limit and args.save_total_limit > 0:
            ckpts = sorted(out_dir.glob("draft_*.pt"))
            if len(ckpts) > args.save_total_limit:
                for old_ckpt in ckpts[:-args.save_total_limit]:
                    old_ckpt.unlink(missing_ok=True)

    if last_step:
        break

    synchronize()
    t0 = time.time()
    lr = get_lr(step, args.lr, args.lr * args.min_lr_ratio, args.warmup_steps, total_iterations)
    for group in optimizer.param_groups:
        group["lr"] = lr

    accum_loss = torch.zeros((), device=device)

    for micro_step in range(grad_accum_steps):
        inputs, targets = next(train_loader)
        if world_size > 1:
            draft_trainer.require_backward_grad_sync = micro_step == grad_accum_steps - 1

        with autocast:
            with torch.no_grad():
                result = target_model(inputs, targets=None, return_features=True)
                target_logits = result["logits"].detach()
                target_features = {k: v.detach() for k, v in result["features"].items()}

            draft_result = draft_trainer(
                target_features=target_features,
                target_logits=target_logits,
                targets=targets,
                embed_fn=target_model.model.embed_tokens,
                lm_head_fn=target_model.lm_head,
            )

        loss = draft_result["loss"] / grad_accum_steps
        loss.backward()
        accum_loss += loss.detach()

    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
    if torch.isfinite(grad_norm):
        optimizer.step()
    else:
        print0(f"step {step:6d} | non-finite grad norm ({grad_norm:.2f}), skipping")
    optimizer.zero_grad(set_to_none=True)

    step += 1
    synchronize()
    dt = time.time() - t0

    if step % args.log_every == 0:
        step_loss = all_reduce_mean(accum_loss).item()
        tok_sec = args.total_batch_size / dt
        print0(
            f"step {step:6d}/{max_iterations} | draft loss {step_loss:.4f} | "
            f"lr {lr:.2e} | grad norm {grad_norm:.2f} | "
            f"{dt * 1000:.0f} ms/step | {tok_sec:,.0f} tok/s"
        )
        if wandb_run:
            wandb_run.log({
                "step": step,
                "train/draft_loss": step_loss,
                "train/lr": lr,
                "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                "train/tok_sec": tok_sec,
            })

if device.type == "cuda":
    print0(f"Peak memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
if wandb_run:
    wandb_run.finish()
if world_size > 1:
    torch.distributed.destroy_process_group()
