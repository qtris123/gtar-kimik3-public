"""
Causal linear attention, o_t = q_t · Σ_{s<=t} k_s^T v_s, in two hardware-aware forms.

Recurrent form:
    - One program per (batch, head, value block) walks the sequence and keeps the
      (head_dim x value block) state in registers.
    - Sequential and memory-bound; this is the decoding form and has no backward.

Chunkwise parallel form:
    - The sequence is cut into chunks of CHUNK_SIZE tokens.
    - One short sequential pass computes the state entering every chunk, then every chunk is processed independently with matmuls on tensor cores: inter-chunk through the state, intra-chunk through a masked (chunk x chunk) score matrix.

All tensors are contiguous (batch, heads, seq_len, dim).
"""

import torch
import triton
import triton.language as tl

from .utils import autocast_inputs, pad_sequence, tile_ptrs

CHUNK_SIZE = 64
BLOCK_K = 64
BLOCK_V = 64


@triton.jit
def _recurrent_forward(
    Q, K, V, Out, InitialState, FinalState,
    scale: tl.constexpr, seq_len,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, BLOCK_V: tl.constexpr,
):
    value_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)

    q_ptrs = Q + batch_head * seq_len * HEAD_DIM + key_offsets
    k_ptrs = K + batch_head * seq_len * HEAD_DIM + key_offsets
    v_ptrs = V + batch_head * seq_len * VALUE_DIM + value_offsets
    out_ptrs = Out + batch_head * seq_len * VALUE_DIM + value_offsets

    state = tl.load(tile_ptrs(InitialState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM))
    for t in range(seq_len):
        q = tl.load(q_ptrs + t * HEAD_DIM).to(tl.float32) * scale
        k = tl.load(k_ptrs + t * HEAD_DIM).to(tl.float32)
        v = tl.load(v_ptrs + t * VALUE_DIM).to(tl.float32)
        state += k[:, None] * v[None, :]
        out = tl.sum(q[:, None] * state, axis=0)
        tl.store(out_ptrs + t * VALUE_DIM, out.to(Out.dtype.element_ty))
    tl.store(tile_ptrs(FinalState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@triton.jit
def _chunk_state_forward(
    K, V, States,
    seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    key_block = tl.program_id(0)
    value_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    chunk_offsets = tl.arange(0, CHUNK)

    # States[b, h, i] is the state entering chunk i; States[b, h, 0] holds the initial state on entry
    state_row_base = batch_head * (num_chunks + 1) * HEAD_DIM
    state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))
    for chunk in range(num_chunks):
        rows = chunk * CHUNK + chunk_offsets
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        state += tl.dot(tl.trans(k), v)
        tl.store(tile_ptrs(States, state_row_base + (chunk + 1) * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@triton.jit
def _chunk_output_forward(
    Q, K, V, States, Out,
    scale: tl.constexpr, seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    value_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    state_row_base = (batch_head * (num_chunks + 1) + chunk) * HEAD_DIM

    v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
    inter = tl.zeros([CHUNK, BLOCK_V], tl.float32)
    scores = tl.zeros([CHUNK, CHUNK], tl.float32)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))
        inter += tl.dot(q, state.to(q.dtype))
        scores += tl.dot(q, tl.trans(k))

    scores = tl.where(chunk_offsets[:, None] >= chunk_offsets[None, :], scores, 0.0)
    out = inter + tl.dot(scores.to(v.dtype), v)
    tl.store(tile_ptrs(Out, batch_head * seq_len, rows, value_offsets, VALUE_DIM), out.to(Out.dtype.element_ty))


@triton.jit
def _chunk_state_backward(
    Q, dOut, dStates,
    scale: tl.constexpr, seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    key_block = tl.program_id(0)
    value_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    chunk_offsets = tl.arange(0, CHUNK)

    # dStates[b, h, i] is the gradient of the state entering chunk i; dStates[b, h, num_chunks] is given
    state_row_base = batch_head * (num_chunks + 1) * HEAD_DIM
    dstate = tl.load(tile_ptrs(dStates, state_row_base + num_chunks * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM))
    for i in range(num_chunks):
        chunk = num_chunks - 1 - i
        rows = chunk * CHUNK + chunk_offsets
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
        dout = tl.load(tile_ptrs(dOut, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        dstate += tl.dot(tl.trans(q), dout)
        tl.store(tile_ptrs(dStates, state_row_base + chunk * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), dstate)


@triton.jit
def _chunk_backward_dqk(
    Q, K, V, dOut, States, dStates, dQ, dK,
    scale: tl.constexpr, seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    key_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    state_row_base = (batch_head * (num_chunks + 1) + chunk) * HEAD_DIM

    q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
    k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
    dq = tl.zeros([CHUNK, BLOCK_K], tl.float32)
    dk = tl.zeros([CHUNK, BLOCK_K], tl.float32)
    dscores = tl.zeros([CHUNK, CHUNK], tl.float32)
    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        dout = tl.load(tile_ptrs(dOut, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))
        next_dstate = tl.load(tile_ptrs(dStates, state_row_base + HEAD_DIM, key_offsets, value_offsets, VALUE_DIM))
        dq += tl.dot(dout, tl.trans(state.to(dout.dtype)))
        dk += tl.dot(v, tl.trans(next_dstate.to(v.dtype)))
        dscores += tl.dot(dout, tl.trans(v))

    dscores = tl.where(chunk_offsets[:, None] >= chunk_offsets[None, :], dscores, 0.0)
    dq += tl.dot(dscores.to(k.dtype), k)
    dk += tl.dot(tl.trans(dscores.to(q.dtype)), q)
    tl.store(tile_ptrs(dQ, batch_head * seq_len, rows, key_offsets, HEAD_DIM), (dq * scale).to(dQ.dtype.element_ty))
    tl.store(tile_ptrs(dK, batch_head * seq_len, rows, key_offsets, HEAD_DIM), dk.to(dK.dtype.element_ty))


@triton.jit
def _chunk_backward_dv(
    Q, K, dOut, dStates, dV,
    scale: tl.constexpr, seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    value_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    next_state_row_base = (batch_head * (num_chunks + 1) + chunk + 1) * HEAD_DIM

    dout = tl.load(tile_ptrs(dOut, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
    dv = tl.zeros([CHUNK, BLOCK_V], tl.float32)
    scores = tl.zeros([CHUNK, CHUNK], tl.float32)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        next_dstate = tl.load(tile_ptrs(dStates, next_state_row_base, key_offsets, value_offsets, VALUE_DIM))
        dv += tl.dot(k, next_dstate.to(k.dtype))
        scores += tl.dot(q, tl.trans(k))

    scores = tl.where(chunk_offsets[:, None] >= chunk_offsets[None, :], scores, 0.0)
    dv += tl.dot(tl.trans(scores.to(dout.dtype)), dout)
    tl.store(tile_ptrs(dV, batch_head * seq_len, rows, value_offsets, VALUE_DIM), dv.to(dV.dtype.element_ty))


def _chunk_states(key, value, initial_state):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = value.shape[-1]
    num_chunks = seq_len // CHUNK_SIZE
    states = torch.empty(batch_size, num_heads, num_chunks + 1, head_dim, value_dim, device=key.device, dtype=torch.float32)
    states[:, :, 0] = initial_state
    grid = (head_dim // BLOCK_K, value_dim // BLOCK_V, batch_size * num_heads)
    _chunk_state_forward[grid](
        key, value, states, seq_len, num_chunks,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )
    return states


class ChunkLinearAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, scale, initial_state):
        seq_len = query.shape[2]
        query, key, value = (pad_sequence(x, CHUNK_SIZE) for x in (query, key, value))
        batch_size, num_heads, padded_len, head_dim = query.shape
        value_dim = value.shape[-1]
        num_chunks = padded_len // CHUNK_SIZE

        states = _chunk_states(key, value, initial_state)
        out = torch.empty_like(value)
        grid = (num_chunks, value_dim // BLOCK_V, batch_size * num_heads)
        _chunk_output_forward[grid](
            query, key, value, states, out, scale, padded_len, num_chunks,
            HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        )

        ctx.save_for_backward(query, key, value, initial_state)
        ctx.scale = scale
        ctx.seq_len = seq_len
        return out[:, :, :seq_len], states[:, :, -1]

    @staticmethod
    def backward(ctx, grad_out, grad_final_state):
        query, key, value, initial_state = ctx.saved_tensors
        batch_size, num_heads, padded_len, head_dim = query.shape
        value_dim = value.shape[-1]
        num_chunks = padded_len // CHUNK_SIZE
        grad_out = pad_sequence(grad_out, CHUNK_SIZE)
        kernel_args = dict(HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V)

        states = _chunk_states(key, value, initial_state)
        dstates = torch.empty_like(states)
        dstates[:, :, -1] = 0.0 if grad_final_state is None else grad_final_state
        grid = (head_dim // BLOCK_K, value_dim // BLOCK_V, batch_size * num_heads)
        _chunk_state_backward[grid](query, grad_out, dstates, ctx.scale, padded_len, num_chunks, **kernel_args)

        grad_query, grad_key, grad_value = torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)
        grid = (num_chunks, head_dim // BLOCK_K, batch_size * num_heads)
        _chunk_backward_dqk[grid](
            query, key, value, grad_out, states, dstates, grad_query, grad_key, ctx.scale, padded_len, num_chunks, **kernel_args
        )
        grid = (num_chunks, value_dim // BLOCK_V, batch_size * num_heads)
        _chunk_backward_dv[grid](query, key, grad_out, dstates, grad_value, ctx.scale, padded_len, num_chunks, **kernel_args)

        seq_len = ctx.seq_len
        return grad_query[:, :, :seq_len], grad_key[:, :, :seq_len], grad_value[:, :, :seq_len], None, dstates[:, :, 0]


def _default_initial_state(query, value, initial_state):
    if initial_state is None:
        batch_size, num_heads, _, head_dim = query.shape
        return query.new_zeros(batch_size, num_heads, head_dim, value.shape[-1], dtype=torch.float32)
    return initial_state.float().contiguous()


@torch.compiler.disable
def chunk_linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    scale = scale or query.shape[-1] ** -0.5
    query, key, value = autocast_inputs(query, key, value)
    initial_state = _default_initial_state(query, value, initial_state)
    out, final_state = ChunkLinearAttention.apply(query, key, value, scale, initial_state)
    return (out, final_state) if output_final_state else out


def recurrent_linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    scale = scale or query.shape[-1] ** -0.5
    query, key, value = (x.contiguous() for x in (query, key, value))
    initial_state = _default_initial_state(query, value, initial_state)
    batch_size, num_heads, seq_len, head_dim = query.shape
    value_dim = value.shape[-1]

    out = torch.empty_like(value)
    final_state = torch.empty_like(initial_state)
    grid = (value_dim // BLOCK_V, batch_size * num_heads)
    _recurrent_forward[grid](
        query, key, value, out, initial_state, final_state, scale, seq_len,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, BLOCK_V=BLOCK_V,
    )
    return (out, final_state) if output_final_state else out
