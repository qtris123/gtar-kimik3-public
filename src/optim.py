import math

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, betas: tuple[float, float]) -> torch.optim.AdamW:
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, fused=True)


def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
