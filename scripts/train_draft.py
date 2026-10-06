"""
Stage 2: Fine-tune the MTP block as a drafter against a frozen target.

python -m scripts.train_draft --config configs/kimi_k3_tiny_mtp.json --data data/test \\
    --target-checkpoint out/kimi_k3_tiny_mtp/ckpt_000020.pt --no-compile

torchrun --standalone --nproc_per_node=4 -m scripts.train_draft \\
    --config configs/kimi_k3_tiny_mtp.json --data data/test \\
    --target-checkpoint out/kimi_k3_tiny_mtp/ckpt_final.pt

The target model is loaded, frozen, and set to eval mode. Only the MTP block
and feature projection parameters are trained.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from src.dataset import TokenLoader
from src.common import (
    all_reduce_mean,
    get_peak_flops,
    init_distributed,
    print0,
)
from src.models import ARCHITECTURES
from src.models.kimi_k3 import KimiK3Config, KimiK3ForCausalLM
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
parser.add_argument("--resume", default=None, help="draft checkpoint to resume from")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--wandb", default=None, help="wandb run name")
parser.add_argument("--wandb-id", default=None)
parser.add_argument("--wandb-resume", default="allow")
parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "GStar"))
parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "vqtri-purdue-university"))
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

# Compute target checkpoint SHA256 once by streaming the file in chunks
print0("Computing target checkpoint SHA256...")
sha = hashlib.sha256()
with open(args.target_checkpoint, "rb") as f:
    while chunk := f.read(65536):
        sha.update(chunk)
target_checkpoint_sha256 = sha.hexdigest()
print0(f"Target checkpoint step: {target_step} | SHA256: {target_checkpoint_sha256[:16]}...")

# Build vocab size from data
train_loader = TokenLoader(args.data, "train", args.device_batch_size, args.seq_len, rank, world_size, device)
build_val_loader = lambda: TokenLoader(args.data, "val", args.device_batch_size, args.seq_len, rank, world_size, device)
vocab_size = math.ceil(train_loader.meta["vocab_size"] / 128) * 128

# Build target model config -- use checkpoint config but override vocab_size
# and ensure feature extraction is configured
Config, ForCausalLM = ARCHITECTURES[config_file["arch"]]
target_config = Config(**target_config_dict)
# Ensure feature layers are set for extraction
if not target_config.feature_layer_indices:
    # Auto-select representative depths: early (1/6), middle (1/2), final (last)
    n = target_config.num_hidden_layers
    target_config.feature_layer_indices = [
        max(0, n // 6),
        n // 2,
        n - 1,
    ]
    print0(f"Auto-selected feature layer indices: {target_config.feature_layer_indices}")

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

# Create a fresh MTP block for the draft (separate from target's frozen copy)
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

# Create draft trainer
draft_config = DraftConfig(
    draft_steps=args.draft_steps,
    feature_layer_indices=target_config.feature_layer_indices,
    step_loss_weighting=args.step_loss_weighting,
    step_loss_decay=args.step_loss_decay,
)
draft_trainer = DraftTrainer(draft_mtp, draft_config).to(device)

# Feature projection is initialized with [0, 0, I] in DraftTrainer.__init__
print0(f"Draft trainer created. Trainable params: {sum(p.numel() for p in draft_trainer.get_trainable_parameters()):,}")
print0(f"Draft config: steps={draft_config.draft_steps}, features={draft_config.feature_layer_indices}, "
       f"weighting={draft_config.step_loss_weighting}")

# --- Optimizer: only draft/projection parameters ---
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

# --- Resume draft checkpoint ---
step = 0
if args.resume:
    print0(f"Resuming draft training from {args.resume}")
    draft_ckpt = torch.load(args.resume, map_location="cpu")
    draft_trainer.load_state_dict(draft_ckpt["draft_trainer"])
    optimizer.load_state_dict(draft_ckpt["optimizer"])
    train_loader.load_state_dict(draft_ckpt["loader"])
    step = draft_ckpt["step"]
    # Verify target checkpoint identity
    saved_target_id = draft_ckpt.get("target_checkpoint")
    if saved_target_id and saved_target_id != str(args.target_checkpoint):
        print0(f"WARNING: Resuming with different target checkpoint. Saved: {saved_target_id}, Current: {args.target_checkpoint}")
    print0(f"Resumed at step {step}")

# --- Wrap for DDP ---
raw_draft_trainer = draft_trainer
if world_size > 1:
    draft_trainer = DistributedDataParallel(
        draft_trainer,
        device_ids=[device.index] if device.type == "cuda" else None,
    )
if args.compile:
    draft_trainer = torch.compile(draft_trainer)

# --- Training loop setup ---
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
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb,
        id=args.wandb_id,
        resume=args.wandb_resume if args.wandb_id else None,
        config={**vars(args), **asdict(target_config), **asdict(draft_config)},
    )


def save_draft_checkpoint(
    path,
    draft_trainer_module,
    optimizer,
    loader,
    step,
    target_config,
    draft_config,
    target_ckpt_path,
    target_step="unknown",
    target_checkpoint_sha256="unknown",
):
    """Save a draft checkpoint with all state needed for resume."""
    # Get git commit SHA for reproducibility
    git_sha = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], 
            capture_output=True, text=True, cwd=os.path.dirname(path)
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except:
        pass
    
    checkpoint = {
        "draft_trainer": draft_trainer_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loader": loader.state_dict(),
        "step": step,
        "target_config": asdict(target_config),
        "draft_config": asdict(draft_config),
        "target_checkpoint": str(target_ckpt_path),
        "target_step": target_step,
        "target_checkpoint_sha256": target_checkpoint_sha256,
        "git_commit_sha": git_sha,
        "creation_timestamp": time.time(),
    }
    torch.save(checkpoint, path)


@torch.no_grad()
def evaluate_draft(draft_trainer_module, target_model, loader, steps):
    """Evaluate draft on validation data and return mean loss + metrics."""
    draft_trainer_module.eval()
    total_loss = torch.zeros((), device=next(draft_trainer_module.parameters()).device)
    total_overlap = torch.zeros((), device=next(draft_trainer_module.parameters()).device)
    total_agree = torch.zeros((), device=next(draft_trainer_module.parameters()).device)
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
        if draft_result["per_step_overlap"]:
            step_overlap = sum(draft_result["per_step_overlap"]) / len(draft_result["per_step_overlap"])
            step_agree = sum(draft_result["per_step_agreement"]) / len(draft_result["per_step_agreement"])
            total_overlap += step_overlap
            total_agree += step_agree
        actual_steps += 1

    if actual_steps == 0:
        draft_trainer_module.train()
        return 0.0, 0.0, 0.0

    # All-reduce across DDP ranks
    avg_loss = all_reduce_mean(total_loss / actual_steps).item()
    avg_overlap = all_reduce_mean(total_overlap / actual_steps).item() 
    avg_agree = all_reduce_mean(total_agree / actual_steps).item()

    draft_trainer_module.train()
    return avg_loss, avg_overlap, avg_agree


# --- Training loop ---
while True:
    last_step = step == max_iterations

    # Evaluation
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        val_loss, val_overlap, val_agree = evaluate_draft(
            raw_draft_trainer, target_model, build_val_loader(), eval_steps
        )
        print0(f"step {step:6d} | val loss {val_loss:.4f} | overlap {val_overlap:.4f} | top1 agree {val_agree:.4f}")
        if wandb_run:
            wandb_run.log({"step": step, "val/loss": val_loss, "val/overlap": val_overlap, "val/top1_agreement": val_agree})

    # Save
    should_save = step > 0 and args.save_every > 0 and step % args.save_every == 0
    if master_process and (last_step or should_save):
        save_draft_checkpoint(
            out_dir / f"draft_{step:06d}.pt", raw_draft_trainer, optimizer, train_loader,
            step, target_config, draft_config, args.target_checkpoint,
            target_step=target_step, target_checkpoint_sha256=target_checkpoint_sha256,
        )
        ckpts = sorted(out_dir.glob("draft_*.pt"))
        if args.save_total_limit and len(ckpts) > args.save_total_limit:
            for old in ckpts[:-args.save_total_limit]:
                old.unlink(missing_ok=True)

    if last_step:
        break

    synchronize()
    t0 = time.time()
    lr = get_lr(step, args.lr, args.lr * args.min_lr_ratio, args.warmup_steps, total_iterations)
    for group in optimizer.param_groups:
        group["lr"] = lr

    accum_loss = torch.zeros((), device=device)
    accum_per_step_overlap = torch.zeros(args.draft_steps, device=device)
    accum_per_step_agree = torch.zeros(args.draft_steps, device=device)
    step_counts = torch.zeros(args.draft_steps, device=device)

    for micro_step in range(grad_accum_steps):
        inputs, targets = next(train_loader)
        if world_size > 1:
            draft_trainer.require_backward_grad_sync = micro_step == grad_accum_steps - 1

        with autocast:
            # Target forward (no grad)
            with torch.no_grad():
                result = target_model(inputs, targets=None, return_features=True)
                target_logits = result["logits"].detach()
                target_features = {k: v.detach() for k, v in result["features"].items()}

            # Draft forward (with grad)
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

        # Accumulate per-step metrics across all microsteps
        if draft_result["per_step_overlap"]:
            for i, (ov, ag) in enumerate(zip(draft_result["per_step_overlap"], draft_result["per_step_agreement"])):
                if i < args.draft_steps:
                    ov_t = ov.detach() if isinstance(ov, torch.Tensor) else torch.tensor(ov, device=device)
                    ag_t = ag.detach() if isinstance(ag, torch.Tensor) else torch.tensor(ag, device=device)
                    accum_per_step_overlap[i] += ov_t
                    accum_per_step_agree[i] += ag_t
                    step_counts[i] += 1

    # Average per-step metrics across microsteps
    mask = step_counts > 0
    accum_per_step_overlap = torch.where(mask, accum_per_step_overlap / step_counts.clamp_min(1.0), accum_per_step_overlap)
    accum_per_step_agree = torch.where(mask, accum_per_step_agree / step_counts.clamp_min(1.0), accum_per_step_agree)

    # Optimizer step
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
    if torch.isfinite(grad_norm):
        optimizer.step()
    else:
        print0(f"step {step:6d} | non-finite gradient norm, skipping")
    optimizer.zero_grad(set_to_none=True)

    step += 1
    synchronize()
    dt = time.time() - t0

    if step % args.log_every == 0:
        train_loss = all_reduce_mean(accum_loss).item()
        tokens_per_sec = args.total_batch_size / dt

        # All-reduce N-element metric tensors across DDP ranks
        reduced_overlap = all_reduce_mean(accum_per_step_overlap)
        reduced_agree = all_reduce_mean(accum_per_step_agree)

        step_metrics = {}
        for i in range(args.draft_steps):
            ov_val = reduced_overlap[i].item()
            ag_val = reduced_agree[i].item()
            step_metrics[f"draft/step{i+1}_overlap"] = ov_val
            step_metrics[f"draft/step{i+1}_agreement"] = ag_val

        overlap_str = " ".join(f"{reduced_overlap[i].item():.3f}" for i in range(args.draft_steps))
        agree_str = " ".join(f"{reduced_agree[i].item():.3f}" for i in range(args.draft_steps))
        print0(
            f"step {step:6d}/{max_iterations} | loss {train_loss:.4f} | lr {lr:.2e} | "
            f"grad_norm {grad_norm:.2f} | overlap [{overlap_str}] | agree [{agree_str}] | "
            f"{dt * 1000:.0f} ms/step | {tokens_per_sec:,.0f} tok/s | epoch {train_loader.epoch}"
        )

        if wandb_run:
            wandb_run.log({
                "step": step,
                "train/loss": train_loss,
                "train/lr": lr,
                "train/grad_norm": grad_norm.item(),
                "train/tokens_per_sec": tokens_per_sec,
                **step_metrics,
            })

if device.type == "cuda":
    print0(f"Peak memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
if wandb_run:
    wandb_run.finish()
if world_size > 1:
    torch.distributed.destroy_process_group()
