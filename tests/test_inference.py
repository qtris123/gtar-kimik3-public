"""
Check that decoding with the cache (prefill, then one token at a time) reproduces the full forward.
From the repo root:

python -m tests.test_inference
"""

import json
from pathlib import Path

import torch

from src.engine import Engine
from src.models.kimi_k3 import KimiK3Config, KimiK3ForCausalLM


def check(config_name, device, seq_len=40, prefill_len=25):
    config_file = json.loads(Path(f"configs/{config_name}.json").read_text())
    torch.manual_seed(0)
    with torch.device(device):
        model = KimiK3ForCausalLM(KimiK3Config(vocab_size=512, **config_file["model"])).eval()
    tokens = torch.randint(0, 512, (2, seq_len), device=device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        full = model(tokens).float()
        cache = model.make_cache(2, seq_len, dtype)
        logits = [model(tokens[:, :prefill_len], cache=cache).float()]
        for t in range(prefill_len, seq_len):
            logits.append(model(tokens[:, t : t + 1], cache=cache).float())
    error = (torch.cat(logits, 1) - full).abs().max().item()
    assert cache.seq_len == seq_len
    assert error < (5e-2 if device == "cuda" else 1e-3), f"{config_name}: cached vs full logits differ by {error:.2e}"

    generated = Engine(model).generate(tokens[:, :10], max_new_tokens=5, temperature=0.0)
    assert generated.shape == (2, 5)
    print(f"ok  {config_name} on {device}: cached vs full logits max err {error:.1e}, greedy generation runs")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check("kimi_k3_tiny", device)
