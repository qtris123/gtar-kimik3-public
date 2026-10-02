"""
Time and measure peak memory of the Triton FlashAttention-2 kernels against PyTorch SDPA at the 0.6B training shape.

python -m tests.bench_flash_attention_2
"""

import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.ops.flash_attention_2 import FlashAttention


def benchmark(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000


def peak_memory_mib(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / 2**20


if __name__ == "__main__":
    batch_size, num_heads, num_kv_heads, seq_len, head_dim = 8, 16, 8, 2048, 128
    dtype = torch.bfloat16
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    grad_out = torch.randn_like(query)
    forward_flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim / 2

    def sdpa(backend):
        def attention():
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)
        return attention

    candidates = {
        "sdpa math": sdpa(SDPBackend.MATH),
        "sdpa flash": sdpa(SDPBackend.FLASH_ATTENTION),
        "triton": lambda: FlashAttention.apply(query, key, value, head_dim**-0.5),
    }
    print(f"B={batch_size} H={num_heads} KV={num_kv_heads} T={seq_len} D={head_dim} {dtype} on {torch.cuda.get_device_name()}")
    for name, attention in candidates.items():
        def forward_backward():
            query.grad = key.grad = value.grad = None
            attention().backward(grad_out)
        forward_ms = benchmark(attention)
        forward_backward_ms = benchmark(forward_backward)
        query.grad = key.grad = value.grad = None
        memory = peak_memory_mib(forward_backward)
        print(
            f"{name:11s} forward {forward_ms:5.2f} ms ({forward_flops / forward_ms / 1e9:4.0f} TFLOPS) | "
            f"forward+backward {forward_backward_ms:5.2f} ms ({3.5 * forward_flops / forward_backward_ms / 1e9:4.0f} TFLOPS) | "
            f"peak memory {memory:6.0f} MiB"
        )
