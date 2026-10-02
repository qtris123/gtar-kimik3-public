"""
Check the gated linear attention kernels (recurrent and chunkwise) against an fp64 recurrence, and against fla's kernels.

python -m tests.test_gated_linear_attention
"""

import math

import torch
import torch.nn.functional as F

from src.ops.gated_linear_attention import chunk_gated_linear_attention, recurrent_gated_linear_attention

try:
    from fla.ops.gla import chunk_gla, fused_recurrent_gla
except ImportError:
    chunk_gla = fused_recurrent_gla = None

NAMES = ("out", "final", "dq", "dk", "dv", "dg", "dinit")
TOLERANCE = {"dg": 1e-2}  # dg = sum(q dq - k dk) cancels, so it carries bf16-level absolute error


def reference(query, key, value, gate, scale=None, initial_state=None, output_final_state=True):
    scale = scale or query.shape[-1] ** -0.5
    query, key, value, gate = (x.double() for x in (query, key, value, gate))
    state = initial_state.double()
    outputs = []
    for t in range(query.shape[2]):
        state = state * gate[:, :, t].exp()[..., None] + key[:, :, t, :, None] * value[:, :, t, None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, :, t] * scale, state))
    return torch.stack(outputs, 2), state


def from_fla(fn, gate_kwarg=None):
    def attention(query, key, value, gate, scale=None, initial_state=None, output_final_state=True):
        query, key, value, gate = (x.transpose(1, 2).contiguous() for x in (query, key, value, gate))
        gate_args = {gate_kwarg: gate} if gate_kwarg else {}
        out, final_state = fn(query, key, value, *(() if gate_kwarg else (gate,)), scale=scale, initial_state=initial_state, output_final_state=True, **gate_args)
        return out.transpose(1, 2), final_state
    return attention


def run(attention, query, key, value, gate, grad_out, initial_state, forward_only=False):
    if forward_only:
        with torch.no_grad():
            return attention(query, key, value, gate, initial_state=initial_state, output_final_state=True)
    inputs = [x.detach().clone().requires_grad_() for x in (query, key, value, gate, initial_state)]
    out, final_state = attention(*inputs[:4], initial_state=inputs[4], output_final_state=True)
    loss = (out.float() * grad_out).sum() + 0.1 * final_state.float().sum()
    return (out, final_state, *torch.autograd.grad(loss, inputs))


def max_errors(results, exact):
    return [(a.float() - b.float()).abs().max().item() for a, b in zip(results, exact)]


def check(batch_size, num_heads, seq_len, head_dim, value_dim, gate_scale, dtype=torch.bfloat16, device="cuda"):
    torch.manual_seed(0)
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, seq_len, value_dim, device=device, dtype=dtype)
    gate = F.logsigmoid(torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)) / gate_scale
    grad_out = torch.randn(batch_size, num_heads, seq_len, value_dim, device=device)
    initial_state = torch.randn(batch_size, num_heads, head_dim, value_dim, device=device)

    exact = run(reference, query, key, value, gate, grad_out, initial_state)
    candidates = {"chunk": chunk_gated_linear_attention, "recurrent": recurrent_gated_linear_attention}
    if chunk_gla is not None:
        candidates["fla chunk"] = from_fla(chunk_gla)
        candidates["fla recurrent"] = from_fla(fused_recurrent_gla, gate_kwarg="gk")
    errors = {name: max_errors(run(fn, query, key, value, gate, grad_out, initial_state, forward_only="recurrent" in name), exact) for name, fn in candidates.items()}

    for ours, theirs in (("chunk", "fla chunk"), ("recurrent", "fla recurrent")):
        if theirs in errors:
            for i, tensor in enumerate(NAMES[: len(errors[ours])]):
                bound = 2 * errors[theirs][i] + TOLERANCE.get(tensor, 1e-3) if math.isfinite(errors[theirs][i]) else 0.1
                assert errors[ours][i] <= bound, (
                    f"{ours} {tensor}: error {errors[ours][i]:.2e} vs {theirs} {errors[theirs][i]:.2e}"
                )
    summary = " | ".join(f"{t} " + "/".join(f"{errors[n][i]:.1e}" if i < len(errors[n]) else "-" for n in errors) for i, t in enumerate(NAMES))
    print(f"ok  B={batch_size} H={num_heads} T={seq_len:4d} K={head_dim:3d} V={value_dim:3d} gate/{gate_scale:<3} {summary}")


def check_state_passing(seq_len=256, device="cuda"):
    torch.manual_seed(1)
    query, key, value = (torch.randn(2, 2, seq_len, 64, device=device, dtype=torch.bfloat16) for _ in range(3))
    gate = F.logsigmoid(torch.randn(2, 2, seq_len, 64, device=device))
    half = seq_len // 2
    for attention in (chunk_gated_linear_attention, recurrent_gated_linear_attention):
        full = attention(query, key, value, gate)
        _, state = attention(*(x[:, :, :half] for x in (query, key, value, gate)), output_final_state=True)
        second = attention(*(x[:, :, half:] for x in (query, key, value, gate)), initial_state=state)
        assert (full[:, :, half:].float() - second.float()).abs().max().item() < 5e-2
    print("ok  state passing between segments")


if __name__ == "__main__":
    print("max abs error vs fp64 recurrence, columns: chunk/recurrent" + ("/fla chunk/fla recurrent" if chunk_gla else ""))
    check(2, 4, 256, 64, 64, gate_scale=16)
    check(2, 4, 256, 64, 64, gate_scale=1)
    check(2, 4, 256, 128, 64, gate_scale=16)
    check(1, 2, 200, 64, 128, gate_scale=4)
    check(2, 2, 512, 128, 128, gate_scale=16)
    check(2, 4, 256, 64, 64, gate_scale=0.1)
    check_state_passing()
