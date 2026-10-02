"""
Pretrain a language model. From the repo root:

python -m scripts.train --config configs/transformer_0.6B.json --data data/fineweb_edu
torchrun --standalone --nproc_per_node=8 -m scripts.train --config configs/kimi_k3_0.6B.json --data data/fineweb_edu

The config's "train" section sets the defaults below; any flag on the command line overrides it.

CPU smoke test:
python -m scripts.train --config configs/transformer_tiny.json --data data/test --no-compile
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from src.dataset import TokenLoader
from src.common import (
    all_reduce_mean,
    evaluate,
    get_peak_flops,
    init_distributed,
    load_checkpoint,
    print0,
    save_checkpoint,
)
from src.models import ARCHITECTURES
from src.optim import build_optimizer, get_lr

parser = argparse.ArgumentParser(description="Pretrain a language model")
parser.add_argument("--config", required=True, help="json with arch, model fields and training defaults")
parser.add_argument("--data", default="data/fineweb_edu")
parser.add_argument("--out", default=None, help="checkpoint directory (default: out/<config name>)")
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--device-batch-size", type=int, default=4, help="sequences per device per micro-step")
parser.add_argument("--total-batch-size", type=int, default=524_288, help="tokens per optimizer step")
parser.add_argument("--total-tokens", type=float, default=10e9, help="training horizon in tokens")
parser.add_argument("--lr", type=float, default=6e-4)
parser.add_argument("--min-lr-ratio", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--warmup-steps", type=int, default=1000)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--eval-every", type=int, default=500)
parser.add_argument("--eval-tokens", type=int, default=20 * 524_288)
parser.add_argument("--save-every", type=int, default=2000)
parser.add_argument("--save-total-limit", type=int, default=3, help="max checkpoints to keep (older ones deleted)")
parser.add_argument("--log-every", type=int, default=10)
parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--wandb", default=None, help="wandb run name (unset = disabled)")
parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "GStar"), help="wandb project name")
parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "vqtri-purdue-university"), help="wandb entity name")
parser.add_argument("--seed", type=int, default=42)
config_file = json.loads(Path(parser.parse_known_args()[0].config).read_text())
parser.set_defaults(**config_file.get("train", {}))
args = parser.parse_args()

rank, world_size, device = init_distributed()
master_process = rank == 0
torch.manual_seed(args.seed)
torch.set_float32_matmul_precision("high")
autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
out_dir = Path(args.out or f"out/{Path(args.config).stem}")
out_dir.mkdir(parents=True, exist_ok=True)

train_loader = TokenLoader(args.data, "train", args.device_batch_size, args.seq_len, rank, world_size, device)
build_val_loader = lambda: TokenLoader(args.data, "val", args.device_batch_size, args.seq_len, rank, world_size, device)

vocab_size = math.ceil(train_loader.meta["vocab_size"] / 128) * 128
Config, ForCausalLM = ARCHITECTURES[config_file["arch"]]
config = Config(vocab_size=vocab_size, **config_file["model"])
with torch.device(device):
    model = ForCausalLM(config)
num_params = sum(p.numel() for p in model.parameters())
flops_per_token = model.estimate_flops_per_token(args.seq_len)
print0(f"Model config:\n{json.dumps(asdict(config), indent=2)}")
print0(f"Parameters: {num_params:,} | FLOPs/token: {flops_per_token:.3e}")

optimizer = build_optimizer(model, args.lr, args.weight_decay, betas=(0.9, 0.95))
step = load_checkpoint(args.resume, model, optimizer, train_loader) if args.resume else 0

raw_model = model
if world_size > 1:
    model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
if args.compile:
    model = torch.compile(model)

tokens_per_micro_step = args.device_batch_size * args.seq_len * world_size
assert args.total_batch_size % tokens_per_micro_step == 0, "total batch size must be a multiple of the micro-step"
grad_accum_steps = args.total_batch_size // tokens_per_micro_step
num_iterations = int(args.total_tokens) // args.total_batch_size
eval_steps = max(1, args.eval_tokens // tokens_per_micro_step)
peak_flops = get_peak_flops(device) * world_size
print0(f"Tokens/step: {args.total_batch_size:,} | grad accum steps: {grad_accum_steps} | iterations: {num_iterations:,}")

wandb_run = None
if args.wandb and master_process:
    import wandb

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb,
        config={**vars(args), **asdict(config)},
    )

while True:
    last_step = step == num_iterations

    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        with autocast:
            val_loss = evaluate(model, build_val_loader(), eval_steps)
        print0(f"step {step:6d} | val loss {val_loss:.4f}")
        if wandb_run:
            wandb_run.log({"step": step, "val/loss": val_loss})

    if master_process and (last_step or (step > 0 and step % args.save_every == 0)):
        save_checkpoint(out_dir / f"ckpt_{step:06d}.pt", raw_model, optimizer, train_loader, step, config)
        if args.save_total_limit and args.save_total_limit > 0:
            ckpts = sorted(out_dir.glob("ckpt_*.pt"))
            if len(ckpts) > args.save_total_limit:
                for old_ckpt in ckpts[:-args.save_total_limit]:
                    old_ckpt.unlink(missing_ok=True)

    if last_step:
        break

    synchronize()
    t0 = time.time()
    lr = get_lr(step, args.lr, args.lr * args.min_lr_ratio, args.warmup_steps, num_iterations)
    for group in optimizer.param_groups:
        group["lr"] = lr
    train_loss = torch.zeros((), device=device)
    for micro_step in range(grad_accum_steps):
        inputs, targets = next(train_loader)
        if world_size > 1:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        with autocast:
            loss = model(inputs, targets) / grad_accum_steps
        loss.backward()
        train_loss += loss.detach()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    if torch.isfinite(grad_norm):
        optimizer.step()
    else:
        print0(f"step {step:6d} | non-finite gradient norm, skipping the update")
    optimizer.zero_grad(set_to_none=True)
    step += 1
    synchronize()
    dt = time.time() - t0

    if step % args.log_every == 0:
        train_loss = all_reduce_mean(train_loss).item()
        tokens_per_sec = args.total_batch_size / dt
        mfu = 100 * flops_per_token * tokens_per_sec / peak_flops
        print0(
            f"step {step:6d}/{num_iterations} | loss {train_loss:.4f} | lr {lr:.2e} | grad norm {grad_norm:.2f} | "
            f"{dt * 1000:.0f} ms/step | {tokens_per_sec:,.0f} tok/s | mfu {mfu:.1f}% | epoch {train_loader.epoch}"
        )
        if wandb_run:
            wandb_run.log({
                "step": step,
                "train/loss": train_loss,
                "train/lr": lr,
                "train/grad_norm": grad_norm.item(),
                "train/tokens_per_sec": tokens_per_sec,
                "train/mfu": mfu,
            })

if device.type == "cuda":
    print0(f"Peak memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
if wandb_run:
    wandb_run.finish()
if world_size > 1:
    torch.distributed.destroy_process_group()
