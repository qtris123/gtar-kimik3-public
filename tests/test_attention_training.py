"""
Train a small Transformer for a few AdamW steps with the Triton attention kernel and with PyTorch SDPA, from the same initialisation and data, and compare the loss curves.

python -m tests.test_attention_training
"""

from pathlib import Path

import torch
import torch.nn.functional as F

import src.layers.attention as attention_module
from src.dataset import TokenLoader
from src.models import TransformerConfig, TransformerForCausalLM
from src.ops import flash_attention
from src.optim import build_optimizer


def sdpa_attention(query, key, value):
    return F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)


def batches(batch_size, seq_len, device):
    if Path("data/test").exists():
        yield from TokenLoader("data/test", "train", batch_size, seq_len, device=device)
    generator = torch.Generator(device=device).manual_seed(1)
    while True:
        tokens = torch.randint(0, 4096, (batch_size, seq_len + 1), device=device, generator=generator)
        yield tokens[:, :-1], tokens[:, 1:]


def train(attention, steps, device="cuda"):
    attention_module.flash_attention = attention
    torch.manual_seed(0)
    config = TransformerConfig(
        vocab_size=151680, hidden_size=512, intermediate_size=1408, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=2, head_dim=64,
    )
    with torch.device(device):
        model = TransformerForCausalLM(config)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
    losses = []
    for _, (inputs, targets) in zip(range(steps), batches(8, 512, device)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return losses


if __name__ == "__main__":
    steps = 30
    triton_losses = train(flash_attention, steps)
    sdpa_losses = train(sdpa_attention, steps)
    gaps = [abs(a - b) for a, b in zip(triton_losses, sdpa_losses)]
    for step in range(0, steps, 5):
        print(f"step {step:2d} | triton {triton_losses[step]:.4f} | sdpa {sdpa_losses[step]:.4f} | gap {gaps[step]:.4f}")
    print(f"final  | triton {triton_losses[-1]:.4f} | sdpa {sdpa_losses[-1]:.4f} | max gap {max(gaps):.4f}")
    assert max(gaps) < 0.05, "loss curves diverged"
    print("ok  training with triton attention matches sdpa")
