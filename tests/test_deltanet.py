"""
Check the DeltaNet kernels (recurrent and chunkwise) against an fp64 recurrence, and against fla's delta-rule kernels.
Keys and queries are L2-normalised as the layer would do.

python -m tests.test_deltanet
"""

import torch
import torch.nn.functional as F

from src.ops.deltanet import chunk_deltanet, recurrent_deltanet

try:
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
except ImportError:
    chunk_delta_rule = fused_recurrent_delta_rule = None

NAMES = ("out", "final", "dq", "dk", "dv", "dbeta", "dinit")


def reference(query, key, value, beta, scale=None, initial_state=None, output_final_state=True):
    scale = scale or query.shape[-1] ** -0.5
    query, key, value, beta = (x.double() for x in (query, key, value, beta))
    state = initial_state.double()
    outputs = []
    for t in range(query.shape[2]):
        k, v = key[:, :, t], value[:, :, t]
        error = v - torch.einsum("bhk,bhkv->bhv", k, state)
        state = state + k[..., :, None] * (beta[:, :, t, None] * error)[..., None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, :, t] * scale, state))
    return torch.stack(outputs, 2), state


def from_fla(fn):
    def attention(query, key, value, beta, scale=None, initial_state=None, output_final_state=True):
        query, key, value, beta = (x.transpose(1, 2).contiguous() for x in (query, key, value, beta))
        out, final_state = fn(query, key, value, beta, scale=scale, initial_state=initial_state, output_final_state=True)
        return out.transpose(1, 2), final_state
    return attention


def run(attention, query, key, value, beta, grad_out, initial_state, forward_only=False):
    if forward_only:
        with torch.no_grad():
            return attention(query, key, value, beta, initial_state=initial_state, output_final_state=True)
    inputs = [x.detach().clone().requires_grad_() for x in (query, key, value, beta, initial_state)]
    out, final_state = attention(*inputs[:4], initial_state=inputs[4], output_final_state=True)
    loss = (out.float() * grad_out).sum() + 0.1 * final_state.float().sum()
    return (out, final_state, *torch.autograd.grad(loss, inputs))


def max_errors(results, exact):
    return [(a.float() - b.float()).abs().max().item() for a, b in zip(results, exact)]


def check(batch_size, num_heads, seq_len, head_dim, value_dim, dtype=torch.bfloat16, device="cuda", correlated=False):
    torch.manual_seed(0)
    query = F.normalize(torch.randn(batch_size, num_heads, seq_len, head_dim, device=device), dim=-1).to(dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    if correlated:  # consecutive keys nearly parallel, as with repeated tokens in real text
        key = torch.randn(batch_size, num_heads, 1, head_dim, device=device) + 0.2 * key
    key = F.normalize(key, dim=-1).to(dtype)
    value = torch.randn(batch_size, num_heads, seq_len, value_dim, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(batch_size, num_heads, seq_len, device=device))
    grad_out = torch.randn(batch_size, num_heads, seq_len, value_dim, device=device)
    initial_state = torch.randn(batch_size, num_heads, head_dim, value_dim, device=device)

    exact = run(reference, query, key, value, beta, grad_out, initial_state)
    candidates = {"chunk": chunk_deltanet, "recurrent": recurrent_deltanet}
    if chunk_delta_rule is not None:
        candidates["fla chunk"] = from_fla(chunk_delta_rule)
        candidates["fla recurrent"] = from_fla(fused_recurrent_delta_rule)
    errors = {name: max_errors(run(fn, query, key, value, beta, grad_out, initial_state, forward_only="recurrent" in name), exact) for name, fn in candidates.items()}

    for ours, theirs in (("chunk", "fla chunk"), ("recurrent", "fla recurrent")):
        if theirs in errors:
            for i, tensor in enumerate(NAMES[: len(errors[ours])]):
                assert errors[ours][i] <= 2 * errors[theirs][i] + 1e-3, (
                    f"{ours} {tensor}: error {errors[ours][i]:.2e} vs {theirs} {errors[theirs][i]:.2e}"
                )
    summary = " | ".join(f"{t} " + "/".join(f"{errors[n][i]:.1e}" if i < len(errors[n]) else "-" for n in errors) for i, t in enumerate(NAMES))
    print(f"ok  B={batch_size} H={num_heads} T={seq_len:4d} K={head_dim:3d} V={value_dim:3d} {'correlated' if correlated else 'random    '} {summary}")


def check_state_passing(seq_len=256, device="cuda"):
    torch.manual_seed(1)
    query, key = (F.normalize(torch.randn(2, 2, seq_len, 64, device=device), dim=-1).to(torch.bfloat16) for _ in range(2))
    value = torch.randn(2, 2, seq_len, 64, device=device, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(2, 2, seq_len, device=device))
    half = seq_len // 2
    for attention in (chunk_deltanet, recurrent_deltanet):
        full = attention(query, key, value, beta)
        _, state = attention(*(x[:, :, :half] for x in (query, key, value, beta)), output_final_state=True)
        second = attention(*(x[:, :, half:] for x in (query, key, value, beta)), initial_state=state)
        assert (full[:, :, half:].float() - second.float()).abs().max().item() < 5e-2
    print("ok  state passing between segments")


if __name__ == "__main__":
    print("max abs error vs fp64 recurrence, columns: chunk/recurrent" + ("/fla chunk/fla recurrent" if chunk_delta_rule else ""))
    check(2, 4, 256, 64, 64)
    check(2, 4, 256, 128, 64)
    check(1, 2, 200, 64, 128)
    check(2, 2, 512, 128, 128)
    check(2, 4, 256, 128, 128, correlated=True)
    check_state_passing()
