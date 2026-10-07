import math
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
def init_distributed() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, world_size, local_rank = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world_size, device


def print0(*args, **kwargs) -> None:
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(tensor)
        tensor /= dist.get_world_size()
    return tensor


@torch.no_grad()
def evaluate(model: nn.Module, loader, steps: int, return_metrics: bool = False) -> float | dict[str, float]:
    """Evaluate model and return the main (next-token) loss or a dict of metrics.

    Args:
        model: model to evaluate.
        loader: data loader.
        steps: evaluation steps.
        return_metrics: if True, returns a dict with 'val/loss', 'val/ppl',
            and optional 'val/mtp_loss'. If False (default), returns float val_loss.
    """
    model.eval()
    total_loss = 0.0
    total_mtp_loss = 0.0
    has_mtp = False

    for _ in range(steps):
        inputs, targets = next(loader)
        result = model(inputs, targets)
        if isinstance(result, dict):
            total_loss = total_loss + result["main_loss"]
            if "mtp_loss" in result:
                has_mtp = True
                total_mtp_loss = total_mtp_loss + result["mtp_loss"]
        else:
            total_loss = total_loss + result

    model.train()
    main_loss_val = all_reduce_mean(total_loss / steps).item()

    if not return_metrics:
        return main_loss_val

    metrics = {
        "val/loss": main_loss_val,
        "val/ppl": math.exp(main_loss_val) if main_loss_val < 20 else float("inf"),
    }
    if has_mtp:
        metrics["val/mtp_loss"] = all_reduce_mean(total_mtp_loss / steps).item()

    return metrics


def save_checkpoint(path: Path, model: nn.Module, optimizer, loader, step: int, config) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loader": loader.state_dict(),
        "step": step,
        "config": asdict(config),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer, loader, strict: bool = True) -> int:
    """Load a checkpoint into the model.

    Args:
        path: checkpoint file.
        model: model to load into.
        optimizer: optimizer to load state into.
        loader: data loader to restore iteration state.
        strict: if False, allow missing keys (e.g. loading a target-only
                checkpoint into an MTP-enabled model and initializing only
                the missing MTP parameters fresh).

    Returns:
        The training step number stored in the checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if strict and (missing or unexpected):
        # Re-raise with the standard strict error
        model.load_state_dict(checkpoint["model"], strict=True)
    elif missing:
        print0(f"[checkpoint] Loaded with {len(missing)} missing keys (initialized fresh):")
        for k in missing[:20]:
            print0(f"  {k}")
        if len(missing) > 20:
            print0(f"  ... and {len(missing) - 20} more")
    if unexpected:
        print0(f"[checkpoint] {len(unexpected)} unexpected keys ignored:")
        for k in unexpected[:10]:
            print0(f"  {k}")
    optimizer.load_state_dict(checkpoint["optimizer"])
    loader.load_state_dict(checkpoint["loader"])
    return checkpoint["step"]
