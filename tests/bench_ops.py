"""
Time and peak memory of the chunkwise kernels against fla and against a naive PyTorch recurrence (an fp32 loop over tokens, the definition of each op).
The naive form only fits at batch size 1.

python -m tests.bench_ops
"""

import time

import torch
import torch.nn.functional as F

from src.ops import (
    chunk_deltanet,
    chunk_gated_deltanet,
    chunk_gated_linear_attention,
    chunk_kimi_delta_attention,
    chunk_linear_attention,
    flash_attention,
)

try:
    from fla.ops.delta_rule import chunk_delta_rule
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gla import chunk_gla
    from fla.ops.kda import chunk_kda
    from fla.ops.linear_attn import chunk_linear_attn
except ImportError:
    chunk_delta_rule = chunk_gated_delta_rule = chunk_gla = chunk_kda = chunk_linear_attn = None


def benchmark(fn, iters):
    for _ in range(2):
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


def naive(query, key, value, gate=None, beta=None):
    scale = query.shape[-1] ** -0.5
    batch_size, num_heads, seq_len, head_dim = query.shape
    query, key, value = (x.float() for x in (query, key, value))
    state = query.new_zeros(batch_size, num_heads, head_dim, value.shape[-1])
    outputs = []
    for t in range(seq_len):
        if gate is not None:
            state = state * gate[:, :, t].exp().view(batch_size, num_heads, -1, 1)
        k, v = key[:, :, t], value[:, :, t]
        if beta is not None:
            v = beta[:, :, t, None] * (v - torch.einsum("bhk,bhkv->bhv", k, state))
        state = state + k[..., :, None] * v[..., None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, :, t] * scale, state))
    return torch.stack(outputs, 2)


def transposed(*tensors):
    return [x.transpose(1, 2).contiguous() for x in tensors]


def report(batch_size, num_heads, seq_len, head_dim, include_naive):
    dtype = torch.bfloat16
    query, key = (F.normalize(torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda"), dim=-1).to(dtype).requires_grad_() for _ in range(2))
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    gate_k = (F.logsigmoid(torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")) / 16).requires_grad_()
    gate = (F.logsigmoid(torch.randn(batch_size, num_heads, seq_len, device="cuda")) / 16).requires_grad_()
    beta = torch.sigmoid(torch.randn(batch_size, num_heads, seq_len, device="cuda")).requires_grad_()
    grad_out = torch.randn_like(value)
    tq, tk, tv, tgate_k, tgate, tbeta = transposed(query, key, value, gate_k, gate, beta)
    inputs = (query, key, value, gate_k, gate, beta)

    rows = {
        "linear attention": {
            "ours": lambda: chunk_linear_attention(query, key, value),
            "fla": lambda: chunk_linear_attn(tq, tk, tv, normalize=False)[0].transpose(1, 2),
            "naive": lambda: naive(query, key, value),
        },
        "gated linear attention": {
            "ours": lambda: chunk_gated_linear_attention(query, key, value, gate_k),
            "fla": lambda: chunk_gla(tq, tk, tv, tgate_k)[0].transpose(1, 2),
            "naive": lambda: naive(query, key, value, gate=gate_k),
        },
        "deltanet": {
            "ours": lambda: chunk_deltanet(query, key, value, beta),
            "fla": lambda: chunk_delta_rule(tq, tk, tv, tbeta)[0].transpose(1, 2),
            "naive": lambda: naive(query, key, value, beta=beta),
        },
        "gated deltanet": {
            "ours": lambda: chunk_gated_deltanet(query, key, value, gate, beta),
            "fla": lambda: chunk_gated_delta_rule(tq, tk, tv, tgate, tbeta)[0].transpose(1, 2),
            "naive": lambda: naive(query, key, value, gate=gate, beta=beta),
        },
        "kimi delta attention": {
            "ours": lambda: chunk_kimi_delta_attention(query, key, value, gate_k, beta),
            "fla": lambda: chunk_kda(tq, tk, tv, tgate_k, tbeta)[0].transpose(1, 2),
            "naive": lambda: naive(query, key, value, gate=gate_k, beta=beta),
        },
        "softmax attention": {
            "ours (flash attention 2)": lambda: flash_attention(query, key, value),
            "sdpa flash": lambda: F.scaled_dot_product_attention(query, key, value, is_causal=True),
        },
    }

    print(f"\nB={batch_size} H={num_heads} T={seq_len} D={head_dim} {dtype} on {torch.cuda.get_device_name()}")
    print(f"{'op':24s} {'implementation':26s} {'forward':>10s} {'fwd+bwd':>10s} {'peak memory':>13s}")
    for op, implementations in rows.items():
        for name, forward in implementations.items():
            if name == "naive" and not include_naive or name == "fla" and chunk_gla is None:
                continue
            iters = 3 if name == "naive" else 20

            def forward_backward():
                for x in inputs:
                    x.grad = None
                forward().backward(grad_out)

            forward_ms = benchmark(forward, iters)
            forward_backward_ms = benchmark(forward_backward, iters)
            for x in inputs:
                x.grad = None
            memory = peak_memory_mib(forward_backward)
            print(f"{op:24s} {name:26s} {forward_ms:8.2f} ms {forward_backward_ms:8.2f} ms {memory:9.0f} MiB")


if __name__ == "__main__":
    report(8, 16, 2048, 128, include_naive=False)
    report(1, 16, 2048, 128, include_naive=True)
