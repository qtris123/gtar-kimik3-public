"""
Check the Triton FlashAttention-2 kernels against PyTorch SDPA, and against fla's parallel_attn.

python -m tests.test_flash_attention_2
"""

import torch
import torch.nn.functional as F

from src.ops.flash_attention_2 import FlashAttention, flash_attention

try:
    from fla.ops.attn import parallel_attn
except ImportError:
    parallel_attn = None

NAMES = ("out", "dq", "dk", "dv")


def sdpa(query, key, value):
    return F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)


def triton(query, key, value):
    return FlashAttention.apply(query, key, value, query.shape[-1] ** -0.5)


def fla(query, key, value):
    out = parallel_attn(*(x.transpose(1, 2).contiguous() for x in (query, key, value)))
    return out.transpose(1, 2)


def forward_backward(attention, query, key, value, grad_out):
    inputs = [x.detach().clone().requires_grad_() for x in (query, key, value)]
    out = attention(*inputs)
    return (out, *torch.autograd.grad(out, inputs, grad_out.to(out.dtype)))


def max_errors(results, exact):
    return [(a.float() - b).abs().max().item() for a, b in zip(results, exact)]


def check(batch_size, num_heads, num_kv_heads, seq_len, head_dim, dtype, device="cuda"):
    torch.manual_seed(0)
    shape_q, shape_kv = (batch_size, num_heads, seq_len, head_dim), (batch_size, num_kv_heads, seq_len, head_dim)
    query, key, value = (torch.randn(shape, device=device, dtype=dtype) for shape in (shape_q, shape_kv, shape_kv))
    grad_out = torch.randn(shape_q, device=device, dtype=dtype)

    exact = forward_backward(sdpa, query.float(), key.float(), value.float(), grad_out.float())
    baselines = {"sdpa": sdpa} | ({"fla": fla} if parallel_attn is not None else {})
    errors = {name: max_errors(forward_backward(fn, query, key, value, grad_out), exact) for name, fn in baselines.items()}
    errors["triton"] = max_errors(forward_backward(triton, query, key, value, grad_out), exact)

    for i, tensor in enumerate(NAMES):
        for baseline in baselines:
            assert errors["triton"][i] <= 2 * errors[baseline][i] + 1e-4, (
                f"{tensor}: triton error {errors['triton'][i]:.2e} vs {baseline} {errors[baseline][i]:.2e}"
            )
    summary = " | ".join(f"{t} " + "/".join(f"{errors[n][i]:.1e}" for n in errors) for i, t in enumerate(NAMES))
    print(f"ok  B={batch_size} H={num_heads:2d} KV={num_kv_heads:2d} T={seq_len:4d} D={head_dim:3d} {str(dtype)[6:]:8s} {summary}")


def check_autocast(device="cuda"):
    query = torch.randn(2, 8, 256, 64, device=device, requires_grad=True)
    key = torch.randn(2, 4, 256, 64, device=device, requires_grad=True)
    value = torch.randn(2, 4, 256, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = flash_attention(query, key, value)
        ref = sdpa(query, key, value)
    assert out.dtype == ref.dtype == torch.bfloat16
    assert (out.float() - ref.float()).abs().max().item() < 2e-2
    torch.autograd.grad(out.sum(), (query, key, value))
    print("ok  autocast with fp32 q/k and bf16 v")


if __name__ == "__main__":
    print(f"max abs error vs fp32 reference, columns: {'/'.join(['sdpa'] + (['fla'] if parallel_attn else []) + ['triton'])}")
    for dtype in (torch.bfloat16, torch.float16):
        check(1, 8, 8, 128, 64, dtype)
        check(2, 8, 8, 512, 64, dtype)
        check(2, 16, 8, 2048, 128, dtype)
        check(2, 8, 2, 1000, 128, dtype)
        check(3, 12, 4, 777, 64, dtype)
        check(1, 32, 8, 256, 128, dtype)
        check(1, 16, 16, 4096, 128, dtype)
    check(1, 4, 1, 64, 32, torch.bfloat16)
    check_autocast()
