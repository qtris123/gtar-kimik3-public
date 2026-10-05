"""
Pretrain a language model. From the repo root:

python -m scripts.train --config configs/transformer_0.6B.json --data data/fineweb_edu
torchrun --standalone --nproc_per_node=8 -m scripts.train --config configs/kimi_k3_0.6B.json --data data/fineweb_edu

The config's "train" section sets the defaults below; any flag on the command line overrides it.

CPU smoke test:
python -m scripts.train --config configs/transformer_tiny.json --data data/test --no-compile

MTP-enabled training:
python -m scripts.train --config configs/kimi_k3_tiny_mtp.json --data data/test --no-compile

To resume a target-only checkpoint with MTP enabled (initializes MTP fresh):
python -m scripts.train --config configs/kimi_k3_tiny_mtp.json --data data/test --resume out/.../ckpt.pt --no-strict-resume --no-compile
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
parser.add_argument("--save-steps", type=str, default=None, help="comma-separated list of explicit steps to save, e.g. '0,200,500,1000'")
parser.add_argument("--max-steps", type=int, default=None, help="maximum steps to train before exiting")
parser.add_argument("--save-total-limit", type=int, default=3, help="max checkpoints to keep (older ones deleted)")
parser.add_argument("--log-every", type=int, default=10)
parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
parser.add_argument("--strict-resume", action=argparse.BooleanOptionalAction, default=True,
                    help="require exact key match on resume (disable to load target-only into MTP model)")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--wandb", default=None, help="wandb run name (unset = disabled)")
parser.add_argument("--wandb-id", default=None, help="wandb run id to resume or attach to")
parser.add_argument("--wandb-resume", default="allow", help="wandb resume mode (default: allow)")
parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "GStar"), help="wandb project name")
parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "vqtri-purdue-university"), help="wandb entity name")
parser.add_argument("--profile-start-step", type=int, default=-1, help="step to start nsys profiler (-1 = disabled)")
parser.add_argument("--profile-end-step", type=int, default=-1, help="step to stop nsys profiler and exit (-1 = disabled)")
parser.add_argument("--seed", type=int, default=42)
config_file = json.loads(Path(parser.parse_known_args()[0].config).read_text())
parser.set_defaults(**config_file.get("train", {}))
args = parser.parse_args()

save_steps_set = set(int(s.strip()) for s in args.save_steps.split(",")) if args.save_steps else set()
if args.save_steps and args.save_total_limit is not None and args.save_total_limit > 0 and args.save_total_limit < len(save_steps_set):
    args.save_total_limit = len(save_steps_set) + 1

rank, world_size, device = init_distributed()
master_process = rank == 0
torch.manual_seed(args.seed)
torch.set_float32_matmul_precision("high")
autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
out_dir = Path(args.out or f"out/{Path(args.config).stem}")
out_dir.mkdir(parents=True, exist_ok=True)
initial_ckpts = {p.resolve() for p in out_dir.glob("ckpt_*.pt")}

train_loader = TokenLoader(args.data, "train", args.device_batch_size, args.seq_len, rank, world_size, device)
build_val_loader = lambda: TokenLoader(args.data, "val", args.device_batch_size, args.seq_len, rank, world_size, device)

vocab_size = math.ceil(train_loader.meta["vocab_size"] / 128) * 128
Config, ForCausalLM = ARCHITECTURES[config_file["arch"]]
config = Config(vocab_size=vocab_size, **config_file["model"])
mtp_enabled = getattr(config, "mtp_enabled", False)
with torch.device(device):
    model = ForCausalLM(config)
num_params = sum(p.numel() for p in model.parameters())
flops_per_token = model.estimate_flops_per_token(args.seq_len)
print0(f"Model config:\n{json.dumps(asdict(config), indent=2)}")
print0(f"Parameters: {num_params:,} | FLOPs/token: {flops_per_token:.3e}")
if mtp_enabled:
    mtp_params = sum(p.numel() for p in model.mtp_block.parameters())
    print0(f"MTP enabled: weight={config.mtp_loss_weight}, MTP block params: {mtp_params:,}")

optimizer = build_optimizer(model, args.lr, args.weight_decay, betas=(0.9, 0.95))
step = load_checkpoint(args.resume, model, optimizer, train_loader, strict=args.strict_resume) if args.resume else 0
start_step = step

raw_model = model
if world_size > 1:
    model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
if args.compile:
    model = torch.compile(model)

tokens_per_micro_step = args.device_batch_size * args.seq_len * world_size
assert args.total_batch_size % tokens_per_micro_step == 0, "total batch size must be a multiple of the micro-step"
grad_accum_steps = args.total_batch_size // tokens_per_micro_step
total_iterations = int(args.total_tokens) // args.total_batch_size
max_iterations = min(total_iterations, args.max_steps) if args.max_steps is not None else total_iterations
eval_steps = max(1, args.eval_tokens // tokens_per_micro_step)
peak_flops = get_peak_flops(device) * world_size
print0(f"Tokens/step: {args.total_batch_size:,} | grad accum steps: {grad_accum_steps} | iterations: {max_iterations:,} (target horizon: {total_iterations:,})")

wandb_run = None
if args.wandb and master_process:
    import wandb

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb,
        id=args.wandb_id,
        resume=args.wandb_resume if args.wandb_id else None,
        config={**vars(args), **asdict(config)},
    )

while True:
    last_step = step == max_iterations

    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        with autocast:
            val_metrics = evaluate(model, build_val_loader(), eval_steps, return_metrics=True)
        val_loss = val_metrics["val/loss"]
        val_ppl = val_metrics["val/ppl"]
        val_str = f"step {step:6d} | val loss {val_loss:.4f} | val ppl {val_ppl:.2f}"
        if "val/mtp_loss" in val_metrics:
            val_mtp_loss = val_metrics["val/mtp_loss"]
            val_mtp_agree = val_metrics.get("val/mtp_agreement", 0.0)
            val_str += f" | val mtp loss {val_mtp_loss:.4f} | val mtp top1 {val_mtp_agree:.4f}"
        print0(val_str)
        if wandb_run:
            wandb_run.log({"step": step, **val_metrics})

    should_save = (step in save_steps_set) or (step > 0 and args.save_every > 0 and step % args.save_every == 0)

    if master_process and (last_step or should_save):
        ckpt_path = out_dir / f"ckpt_{step:06d}.pt"
        if not (args.resume and step == start_step and ckpt_path.exists()):
            save_checkpoint(ckpt_path, raw_model, optimizer, train_loader, step, config)
            if args.save_total_limit and args.save_total_limit > 0:
                ckpts = sorted(out_dir.glob("ckpt_*.pt"))
                # Protect explicit milestones in save_steps_set and pre-existing checkpoints from rolling pruning
                prunable = [
                    c for c in ckpts
                    if int(c.stem.split("_")[1]) not in save_steps_set and c.resolve() not in initial_ckpts
                ]
                if len(prunable) > args.save_total_limit:
                    for old_ckpt in prunable[:-args.save_total_limit]:
                        old_ckpt.unlink(missing_ok=True)

    if last_step:
        break

    if args.profile_start_step >= 0 and step == args.profile_start_step:
        synchronize()
        print0(f"\n>>> [NSYS] Starting CUDA profiler capture at step {step}...")
        if device.type == "cuda":
            torch.cuda.cudart().cudaProfilerStart()

    synchronize()
    t0 = time.time()
    lr = get_lr(step, args.lr, args.lr * args.min_lr_ratio, args.warmup_steps, total_iterations)
    for group in optimizer.param_groups:
        group["lr"] = lr
    train_loss = torch.zeros((), device=device)
    train_main_loss = torch.zeros((), device=device)
    train_mtp_loss = torch.zeros((), device=device)
    train_mtp_agree = torch.zeros((), device=device)
    train_main_agree = torch.zeros((), device=device)

    if device.type == "cuda":
        torch.cuda.nvtx.range_push(f"step_{step}")
    for micro_step in range(grad_accum_steps):
        if device.type == "cuda":
            torch.cuda.nvtx.range_push("dataloader")
        inputs, targets = next(train_loader)
        if device.type == "cuda":
            torch.cuda.nvtx.range_pop()
        if world_size > 1:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        with autocast:
            if device.type == "cuda":
                torch.cuda.nvtx.range_push("forward")
            result = model(inputs, targets)
            if device.type == "cuda":
                torch.cuda.nvtx.range_pop()

        # Handle both scalar loss and MTP dict loss
        if isinstance(result, dict):
            loss = result["loss"] / grad_accum_steps
            train_main_loss += result["main_loss"].detach() / grad_accum_steps
            train_mtp_loss += result["mtp_loss"].detach() / grad_accum_steps
            if "mtp_agreement" in result:
                train_mtp_agree += result["mtp_agreement"].detach() / grad_accum_steps
            if "main_agreement" in result:
                train_main_agree += result["main_agreement"].detach() / grad_accum_steps
        else:
            loss = result / grad_accum_steps
            train_main_loss += result.detach() / grad_accum_steps

        if device.type == "cuda":
            torch.cuda.nvtx.range_push("backward")
        loss.backward()
        if device.type == "cuda":
            torch.cuda.nvtx.range_pop()
        train_loss += loss.detach()

    if device.type == "cuda":
        torch.cuda.nvtx.range_push("optimizer")

    kda_metrics = {}
    if (step + 1) % args.log_every == 0:
        with torch.no_grad():
            kda_grad_sq = torch.zeros((), device=device)
            attn_grad_sq = torch.zeros((), device=device)
            mlp_grad_sq = torch.zeros((), device=device)
            beta_grad_sq = torch.zeros((), device=device)
            decay_grad_sq = torch.zeros((), device=device)
            mtp_grad_sq = torch.zeros((), device=device)

            for name, p in raw_model.named_parameters():
                if p.grad is not None:
                    g_sq = p.grad.detach().pow(2).sum()

                    # MTP block gradient tracking
                    if "mtp_block." in name:
                        mtp_grad_sq += g_sq
                        continue

                    if "beta_proj" in name:
                        beta_grad_sq += g_sq
                    elif "decay_proj" in name or "A_log" in name:
                        decay_grad_sq += g_sq

                    if ".mixer." in name:
                        parts = name.split(".")
                        if "layers" in parts:
                            idx = int(parts[parts.index("layers") + 1])
                            if (idx + 1) % getattr(config, "attention_interval", 4) == 0:
                                attn_grad_sq += g_sq
                            else:
                                kda_grad_sq += g_sq
                        else:
                            kda_grad_sq += g_sq
                    elif ".mlp." in name:
                        mlp_grad_sq += g_sq

            if kda_grad_sq > 0 or attn_grad_sq > 0:
                kda_metrics["kda/grad_norm"] = kda_grad_sq.sqrt().item()
                kda_metrics["kda/grad_norm_beta"] = beta_grad_sq.sqrt().item()
                kda_metrics["kda/grad_norm_decay"] = decay_grad_sq.sqrt().item()
                kda_metrics["attn/grad_norm"] = attn_grad_sq.sqrt().item()
                kda_metrics["mlp/grad_norm"] = mlp_grad_sq.sqrt().item()

            if mtp_enabled:
                kda_metrics["mtp/grad_norm"] = mtp_grad_sq.sqrt().item()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    if torch.isfinite(grad_norm):
        optimizer.step()
    else:
        print0(f"step {step:6d} | non-finite gradient norm, skipping the update")
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.nvtx.range_pop() # optimizer
        torch.cuda.nvtx.range_pop() # step_{step}

    step += 1
    synchronize()
    dt = time.time() - t0

    if args.profile_end_step >= 0 and step >= args.profile_end_step:
        synchronize()
        print0(f">>> [NSYS] Completed profiling target steps ({args.profile_start_step} to {args.profile_end_step}). Stopping capture...")
        if device.type == "cuda":
            torch.cuda.cudart().cudaProfilerStop()
        break

    if step % args.log_every == 0:
        train_loss = all_reduce_mean(train_loss).item()
        tokens_per_sec = args.total_batch_size / dt
        mfu = 100 * flops_per_token * tokens_per_sec / peak_flops

        kda_layers = [
            layer.mixer for layer in getattr(raw_model.model, "layers", [])
            if hasattr(layer, "mixer") and getattr(layer.mixer, "stats", None) is not None
        ]

        # Build loss string
        if mtp_enabled:
            main_loss_val = all_reduce_mean(train_main_loss).item()
            mtp_loss_val = all_reduce_mean(train_mtp_loss).item()
            mtp_agree_val = all_reduce_mean(train_mtp_agree).item()
            loss_str = f"loss {train_loss:.4f} (main {main_loss_val:.4f} + mtp {mtp_loss_val:.4f} | mtp_top1 {mtp_agree_val:.3f})"
            kda_metrics["train/main_loss"] = main_loss_val
            kda_metrics["train/mtp_loss"] = mtp_loss_val
            kda_metrics["train/mtp_agreement"] = mtp_agree_val
            main_agree_val = all_reduce_mean(train_main_agree).item()
            if main_agree_val > 0:
                kda_metrics["train/main_agreement"] = main_agree_val
        else:
            loss_str = f"loss {train_loss:.4f}"

        if kda_layers:
            stats_stack = torch.stack([l.stats for l in kda_layers])
            beta_mean = all_reduce_mean(stats_stack[:, 0].mean()).item()
            decay_gate_mean = all_reduce_mean(stats_stack[:, 1].mean()).item()
            state_norm = all_reduce_mean(stats_stack[:, 2].mean()).item()
            out_gate_mean = all_reduce_mean(stats_stack[:, 3].mean()).item()

            A_vals = torch.cat([l.A_log.detach().exp() for l in kda_layers])
            A_val_mean = A_vals.mean().item()

            kda_metrics.update({
                "kda/beta_mean": beta_mean,
                "kda/decay_gate_mean": decay_gate_mean,
                "kda/state_norm": state_norm,
                "kda/output_gate_mean": out_gate_mean,
                "kda/A_val_mean": A_val_mean,
            })

            print0(
                f"step {step:6d}/{max_iterations} | {loss_str} | lr {lr:.2e} | grad norm {grad_norm:.2f} | "
                f"kda_g {kda_metrics.get('kda/grad_norm', 0.0):.2f} | beta {beta_mean:.3f} | decay {decay_gate_mean:.3f} | S_norm {state_norm:.2f} | "
                f"{dt * 1000:.0f} ms/step | {tokens_per_sec:,.0f} tok/s | mfu {mfu:.1f}% | epoch {train_loader.epoch}"
            )
        else:
            print0(
                f"step {step:6d}/{max_iterations} | {loss_str} | lr {lr:.2e} | grad norm {grad_norm:.2f} | "
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
                **kda_metrics,
            })

if device.type == "cuda":
    print0(f"Peak memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
if wandb_run:
    wandb_run.finish()
if world_size > 1:
    torch.distributed.destroy_process_group()
