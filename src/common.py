import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

PEAK_FLOPS = {"B200": 2250e12, "H200": 989e12, "H100": 989e12, "A100": 312e12}


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


def get_peak_flops(device: torch.device) -> float:
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else ""
    return next((flops for key, flops in PEAK_FLOPS.items() if key in name), float("inf"))


@torch.no_grad()
def evaluate(model: nn.Module, loader, steps: int) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(steps):
        inputs, targets = next(loader)
        total_loss = total_loss + model(inputs, targets)
    model.train()
    return all_reduce_mean(total_loss / steps).item()


def save_checkpoint(path: Path, model: nn.Module, optimizer, loader, step: int, config) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loader": loader.state_dict(),
        "step": step,
        "config": asdict(config),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer, loader) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    loader.load_state_dict(checkpoint["loader"])
    return checkpoint["step"]
