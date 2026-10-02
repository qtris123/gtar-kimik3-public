"""
Check the linear attention kernels (recurrent and chunkwise) against an fp64 recurrence, and against fla's kernels when installed.

python -m tests.test_linear_attention
"""

import torch

from src.ops.linear_attention import chunk_linear_attention, recurrent_linear_attention

try:
    from fla.ops.linear_attn import chunk_linear_attn, fused_recurrent_linear_attn
except ImportError:
    chunk_linear_attn = fused_recurrent_linear_attn = None

NAMES = ("out", "final", "dq", "dk", "dv", "dinit")


def reference(query, key, value, scale=None, initial_state=None, output_final_state=True):
    scale = scale or query.shape[-1] ** -0.5
    query, key, value = (x.double() for x in (query, key, value))
    state = initial_state.double()
    outputs = []
    for t in range(query.shape[2]):
        state = state + key[:, :, t, :, None] * value[:, :, t, None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, :, t] * scale, state))
    return torch.stack(outputs, 2), state


def from_fla(fn, **fixed):
    def attention(query, key, value, scale=None, initial_state=None, output_final_state=True):
        out, final_state = fn(
            *(x.transpose(1, 2).contiguous() for x in (query, key, value)),
            scale=scale, initial_state=initial_state, output_final_state=True, **fixed,
        )
        return out.transpose(1, 2), final_state
    return attention


def run(attention, query, key, value, grad_out, initial_state, forward_only=False):
    if forward_only:
        with torch.no_grad():
            return attention(query, key, value, initial_state=initial_state, output_final_state=True)
    inputs = [x.detach().clone().requires_grad_() for x in (query, key, value, initial_state)]
    out, final_state = attention(*inputs[:3], initial_state=inputs[3], output_final_state=True)
    loss = (out.float() * grad_out).sum() + 0.1 * final_state.float().sum()
    return (out, final_state, *torch.autograd.grad(loss, inputs))


def max_errors(results, exact):
    return [(a.float() - b.float()).abs().max().item() for a, b in zip(results, exact)]


def check(batch_size, num_heads, seq_len, head_dim, value_dim, dtype=torch.bfloat16, device="cuda"):
    torch.manual_seed(0)
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, seq_len, value_dim, device=device, dtype=dtype)
    grad_out = torch.randn(batch_size, num_heads, seq_len, value_dim, device=device)
    initial_state = torch.randn(batch_size, num_heads, head_dim, value_dim, device=device)

    exact = run(reference, query, key, value, grad_out, initial_state)
    candidates = {"chunk": chunk_linear_attention, "recurrent": recurrent_linear_attention}
    if chunk_linear_attn is not None:
        candidates["fla chunk"] = from_fla(chunk_linear_attn, normalize=False)
        candidates["fla recurrent"] = from_fla(fused_recurrent_linear_attn, normalize=False)
    errors = {name: max_errors(run(fn, query, key, value, grad_out, initial_state, forward_only="recurrent" in name), exact) for name, fn in candidates.items()}

    for ours, theirs in (("chunk", "fla chunk"), ("recurrent", "fla recurrent")):
        if theirs in errors:
            for i, tensor in enumerate(NAMES[: len(errors[ours])]):
                assert errors[ours][i] <= 2 * errors[theirs][i] + 1e-3, (
                    f"{ours} {tensor}: error {errors[ours][i]:.2e} vs {theirs} {errors[theirs][i]:.2e}"
                )
    summary = " | ".join(f"{t} " + "/".join(f"{errors[n][i]:.1e}" if i < len(errors[n]) else "-" for n in errors) for i, t in enumerate(NAMES))
    print(f"ok  B={batch_size} H={num_heads} T={seq_len:4d} K={head_dim:3d} V={value_dim:3d} {summary}")


def check_state_passing(seq_len=256, device="cuda"):
    torch.manual_seed(1)
    query, key, value = (torch.randn(2, 2, seq_len, 64, device=device, dtype=torch.bfloat16) for _ in range(3))
    half = seq_len // 2
    for attention in (chunk_linear_attention, recurrent_linear_attention):
        full = attention(query, key, value)
        _, state = attention(query[:, :, :half], key[:, :, :half], value[:, :, :half], output_final_state=True)
        second = attention(query[:, :, half:], key[:, :, half:], value[:, :, half:], initial_state=state)
        assert (full[:, :, half:].float() - second.float()).abs().max().item() < 5e-2
    print("ok  state passing between segments")


if __name__ == "__main__":
    print(f"max abs error vs fp64 recurrence, columns: chunk/recurrent" + ("/fla chunk/fla recurrent" if chunk_linear_attn else ""))
    check(2, 4, 256, 64, 64)
    check(2, 4, 256, 128, 64)
    check(1, 2, 200, 64, 128)
    check(2, 2, 512, 128, 128)
    check_state_passing()
