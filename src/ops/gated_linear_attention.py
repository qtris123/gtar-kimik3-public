"""
Gated linear attention (Yang et al., 2023): linear attention whose state decays through a data-dependent gate on the key dimension:
    S_t = diag(exp(g_t)) S_{t-1} + k_t^T v_t,  o_t = q_t S_t,  with log gates g_t <= 0.

Recurrent form:
    - One program per (batch, head, value block) scans the sequence with the state in registers.
    - This is the decoding form and has no backward.

Chunkwise parallel form: 
    - Inside a chunk the gate acts through its cumulative sum b_t. 
    - Keys are decayed towards the chunk end (k_t exp(b_end - b_t)) to build the next state and queries towards the chunk start (q_t exp(b_t)) to read it, so both factors are <= 1. 
    - The intra-chunk scores q_t k_s exp(b_t - b_s) are assembled from SUB_CHUNK x SUB_CHUNK blocks, each anchored at the end of its key sub-chunk, so no exponent grows with the chunk size (the paper's secondary chunking).
    - The gate gradient uses dg_t = sum_{s>=t} (q_s dq_s - k_s dk_s).

All tensors are contiguous (batch, heads, seq_len, dim).
"""

import torch
import triton
import triton.language as tl

from .utils import autocast_inputs, chunk_cumsum, gated_scores_backward_key, gated_scores_backward_left, gated_scores_diagonal, pad_sequence, tile_ptrs

CHUNK_SIZE = 64
SUB_CHUNK = 16
BLOCK_K = 64
BLOCK_V = 64


@triton.jit
def _recurrent_forward(
    Q, K, V, Gate, Out, InitialState, FinalState,
    scale: tl.constexpr, seq_len,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, BLOCK_V: tl.constexpr,
):
    value_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)

    q_ptrs = Q + batch_head * seq_len * HEAD_DIM + key_offsets
    k_ptrs = K + batch_head * seq_len * HEAD_DIM + key_offsets
    gate_ptrs = Gate + batch_head * seq_len * HEAD_DIM + key_offsets
    v_ptrs = V + batch_head * seq_len * VALUE_DIM + value_offsets
    out_ptrs = Out + batch_head * seq_len * VALUE_DIM + value_offsets

    state = tl.load(tile_ptrs(InitialState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM))
    for t in range(seq_len):
        q = tl.load(q_ptrs + t * HEAD_DIM).to(tl.float32) * scale
        k = tl.load(k_ptrs + t * HEAD_DIM).to(tl.float32)
        v = tl.load(v_ptrs + t * VALUE_DIM).to(tl.float32)
        gate = tl.load(gate_ptrs + t * HEAD_DIM).to(tl.float32)
        state = state * tl.exp(gate)[:, None] + k[:, None] * v[None, :]
        out = tl.sum(q[:, None] * state, axis=0)
        tl.store(out_ptrs + t * VALUE_DIM, out.to(Out.dtype.element_ty))
    tl.store(tile_ptrs(FinalState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@triton.jit
def _chunk_state_forward(
    K, V, GateCumsum, States,
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
        gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        chunk_decay = tl.load(GateCumsum + (batch_head * seq_len + chunk * CHUNK + CHUNK - 1) * HEAD_DIM + key_offsets)
        k_decayed = (k * tl.exp(chunk_decay[None, :] - gate_cumsum)).to(k.dtype)
        state = state * tl.exp(chunk_decay)[:, None] + tl.dot(tl.trans(k_decayed), v)
        tl.store(tile_ptrs(States, state_row_base + (chunk + 1) * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@triton.jit
def _chunk_intra_forward(
    Q, K, GateCumsum, Scores,
    scale: tl.constexpr, seq_len,
    HEAD_DIM: tl.constexpr, CHUNK: tl.constexpr, SUB: tl.constexpr, BLOCK_K: tl.constexpr,
):
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1)
    sub_offsets = tl.arange(0, SUB)

    for i in tl.static_range(CHUNK // SUB):
        query_rows = chunk * CHUNK + i * SUB + sub_offsets
        for j in tl.static_range(i + 1):
            key_rows = chunk * CHUNK + j * SUB + sub_offsets
            anchor_row = chunk * CHUNK + j * SUB + SUB - 1
            scores = tl.zeros([SUB, SUB], tl.float32)
            for key_block in range(HEAD_DIM // BLOCK_K):
                key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
                q = tl.load(tile_ptrs(Q, batch_head * seq_len, query_rows, key_offsets, HEAD_DIM)) * scale
                k = tl.load(tile_ptrs(K, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM))
                query_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, query_rows, key_offsets, HEAD_DIM))
                key_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM))
                anchor = tl.load(GateCumsum + (batch_head * seq_len + anchor_row) * HEAD_DIM + key_offsets)
                if i == j:  # the diagonal block is built row by row in fp32, so no exponent is ever positive
                    scores += gated_scores_diagonal(q.to(tl.float32), k.to(tl.float32), query_cumsum, key_cumsum)
                else:
                    q_decayed = (q * tl.exp(query_cumsum - anchor[None, :])).to(q.dtype)
                    k_decayed = (k * tl.exp(anchor[None, :] - key_cumsum)).to(k.dtype)
                    scores += tl.dot(q_decayed, tl.trans(k_decayed))
            if i == j:
                scores = tl.where(sub_offsets[:, None] >= sub_offsets[None, :], scores, 0.0)
            tl.store(tile_ptrs(Scores, batch_head * seq_len, query_rows, j * SUB + sub_offsets, CHUNK), scores)


@triton.jit
def _chunk_output_forward(
    Q, V, GateCumsum, States, Scores, Out,
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
    scores = tl.load(tile_ptrs(Scores, batch_head * seq_len, rows, chunk_offsets, CHUNK))
    out = tl.dot(scores.to(v.dtype), v)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
        gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))
        q_decayed = (q * tl.exp(gate_cumsum)).to(q.dtype)
        out += tl.dot(q_decayed, state.to(q.dtype))
    tl.store(tile_ptrs(Out, batch_head * seq_len, rows, value_offsets, VALUE_DIM), out.to(Out.dtype.element_ty))


@triton.jit
def _chunk_state_backward(
    Q, dOut, GateCumsum, dStates,
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
        gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        chunk_decay = tl.load(GateCumsum + (batch_head * seq_len + chunk * CHUNK + CHUNK - 1) * HEAD_DIM + key_offsets)
        q_decayed = (q * tl.exp(gate_cumsum)).to(q.dtype)
        dstate = dstate * tl.exp(chunk_decay)[:, None] + tl.dot(tl.trans(q_decayed), dout)
        tl.store(tile_ptrs(dStates, state_row_base + chunk * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), dstate)


@triton.jit
def _chunk_intra_backward(
    V, dOut, dScores,
    seq_len,
    VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets

    dscores = tl.zeros([CHUNK, CHUNK], tl.float32)
    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        dout = tl.load(tile_ptrs(dOut, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        dscores += tl.dot(dout, tl.trans(v))
    dscores = tl.where(chunk_offsets[:, None] >= chunk_offsets[None, :], dscores, 0.0)
    tl.store(tile_ptrs(dScores, batch_head * seq_len, rows, chunk_offsets, CHUNK), dscores)


@triton.jit
def _chunk_backward_dq(
    K, dOut, GateCumsum, States, dScores, dQ,
    scale: tl.constexpr, seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, SUB: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    key_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    sub_offsets = tl.arange(0, SUB)
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    state_row_base = (batch_head * (num_chunks + 1) + chunk) * HEAD_DIM

    for i in tl.static_range(CHUNK // SUB):
        query_rows = chunk * CHUNK + i * SUB + sub_offsets
        query_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, query_rows, key_offsets, HEAD_DIM))

        dq = tl.zeros([SUB, BLOCK_K], tl.float32)
        for value_block in range(VALUE_DIM // BLOCK_V):
            value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
            dout = tl.load(tile_ptrs(dOut, batch_head * seq_len, query_rows, value_offsets, VALUE_DIM))
            state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))
            dq += tl.dot(dout, tl.trans(state.to(dout.dtype)))
        dq = dq * tl.exp(query_cumsum)

        for j in tl.static_range(i + 1):
            key_rows = chunk * CHUNK + j * SUB + sub_offsets
            anchor_row = chunk * CHUNK + j * SUB + SUB - 1
            k = tl.load(tile_ptrs(K, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM))
            key_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM))
            anchor = tl.load(GateCumsum + (batch_head * seq_len + anchor_row) * HEAD_DIM + key_offsets)
            dscores = tl.load(tile_ptrs(dScores, batch_head * seq_len, query_rows, j * SUB + sub_offsets, CHUNK))
            if i == j:
                dq += gated_scores_backward_left(dscores, k.to(tl.float32), query_cumsum, key_cumsum)
            else:
                k_decayed = (k * tl.exp(anchor[None, :] - key_cumsum)).to(k.dtype)
                dq += tl.dot(dscores.to(k.dtype), k_decayed) * tl.exp(query_cumsum - anchor[None, :])

        tl.store(tile_ptrs(dQ, batch_head * seq_len, query_rows, key_offsets, HEAD_DIM), (dq * scale).to(dQ.dtype.element_ty))


@triton.jit
def _chunk_backward_dk(
    Q, V, GateCumsum, dStates, dScores, dK,
    scale: tl.constexpr, seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, SUB: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    key_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    sub_offsets = tl.arange(0, SUB)
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    next_state_row_base = (batch_head * (num_chunks + 1) + chunk + 1) * HEAD_DIM
    chunk_decay = tl.load(GateCumsum + (batch_head * seq_len + chunk * CHUNK + CHUNK - 1) * HEAD_DIM + key_offsets)

    for j in tl.static_range(CHUNK // SUB):
        key_rows = chunk * CHUNK + j * SUB + sub_offsets
        anchor_row = chunk * CHUNK + j * SUB + SUB - 1
        key_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM))
        anchor = tl.load(GateCumsum + (batch_head * seq_len + anchor_row) * HEAD_DIM + key_offsets)

        dk = tl.zeros([SUB, BLOCK_K], tl.float32)
        for value_block in range(VALUE_DIM // BLOCK_V):
            value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
            v = tl.load(tile_ptrs(V, batch_head * seq_len, key_rows, value_offsets, VALUE_DIM))
            next_dstate = tl.load(tile_ptrs(dStates, next_state_row_base, key_offsets, value_offsets, VALUE_DIM))
            dk += tl.dot(v, tl.trans(next_dstate.to(v.dtype)))
        dk = dk * tl.exp(chunk_decay[None, :] - key_cumsum)

        for i in tl.static_range(j, CHUNK // SUB):
            query_rows = chunk * CHUNK + i * SUB + sub_offsets
            q = tl.load(tile_ptrs(Q, batch_head * seq_len, query_rows, key_offsets, HEAD_DIM)) * scale
            query_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, query_rows, key_offsets, HEAD_DIM))
            dscores = tl.load(tile_ptrs(dScores, batch_head * seq_len, query_rows, j * SUB + sub_offsets, CHUNK))
            if i == j:
                dk += gated_scores_backward_key(dscores, q.to(tl.float32), query_cumsum, key_cumsum)
            else:
                q_decayed = (q * tl.exp(query_cumsum - anchor[None, :])).to(q.dtype)
                dk += tl.dot(tl.trans(dscores.to(q.dtype)), q_decayed) * tl.exp(anchor[None, :] - key_cumsum)

        tl.store(tile_ptrs(dK, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM), dk.to(dK.dtype.element_ty))


@triton.jit
def _chunk_backward_dv(
    K, dOut, GateCumsum, dStates, Scores, dV,
    seq_len, num_chunks,
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
    scores = tl.load(tile_ptrs(Scores, batch_head * seq_len, rows, chunk_offsets, CHUNK))
    dv = tl.dot(tl.trans(scores.to(dout.dtype)), dout)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        chunk_decay = tl.load(GateCumsum + (batch_head * seq_len + chunk * CHUNK + CHUNK - 1) * HEAD_DIM + key_offsets)
        next_dstate = tl.load(tile_ptrs(dStates, next_state_row_base, key_offsets, value_offsets, VALUE_DIM))
        k_decayed = (k * tl.exp(chunk_decay[None, :] - gate_cumsum)).to(k.dtype)
        dv += tl.dot(k_decayed, next_dstate.to(k.dtype))
    tl.store(tile_ptrs(dV, batch_head * seq_len, rows, value_offsets, VALUE_DIM), dv.to(dV.dtype.element_ty))


@triton.jit
def _chunk_gate_gradient(
    Q, K, dQ, dK, States, dStates, dGate,
    seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    key_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    rows = chunk * CHUNK + tl.arange(0, CHUNK)
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    next_state_row_base = (batch_head * (num_chunks + 1) + chunk + 1) * HEAD_DIM

    # dg_t = sum_{s>=t in the chunk} (q_s dq_s - k_s dk_s) + rowsum(S ⊙ dS) at the chunk boundary
    q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)).to(tl.float32)
    k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM)).to(tl.float32)
    dq = tl.load(tile_ptrs(dQ, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
    dk = tl.load(tile_ptrs(dK, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
    dgate = tl.cumsum(q * dq - k * dk, axis=0, reverse=True)

    boundary = tl.zeros([BLOCK_K], tl.float32)
    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        next_state = tl.load(tile_ptrs(States, next_state_row_base, key_offsets, value_offsets, VALUE_DIM))
        next_dstate = tl.load(tile_ptrs(dStates, next_state_row_base, key_offsets, value_offsets, VALUE_DIM))
        boundary += tl.sum(next_state * next_dstate, axis=1)
    dgate = dgate + boundary[None, :]
    tl.store(tile_ptrs(dGate, batch_head * seq_len, rows, key_offsets, HEAD_DIM), dgate.to(dGate.dtype.element_ty))


def _chunk_states(key, value, gate_cumsum, initial_state):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = value.shape[-1]
    num_chunks = seq_len // CHUNK_SIZE
    states = torch.empty(batch_size, num_heads, num_chunks + 1, head_dim, value_dim, device=key.device, dtype=torch.float32)
    states[:, :, 0] = initial_state
    grid = (head_dim // BLOCK_K, value_dim // BLOCK_V, batch_size * num_heads)
    _chunk_state_forward[grid](
        key, value, gate_cumsum, states, seq_len, num_chunks,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )
    return states


def _chunk_scores(query, key, gate_cumsum, scale):
    batch_size, num_heads, seq_len, head_dim = query.shape
    scores = torch.zeros(batch_size, num_heads, seq_len, CHUNK_SIZE, device=query.device, dtype=torch.float32)
    grid = (seq_len // CHUNK_SIZE, batch_size * num_heads)
    _chunk_intra_forward[grid](
        query, key, gate_cumsum, scores, scale, seq_len,
        HEAD_DIM=head_dim, CHUNK=CHUNK_SIZE, SUB=SUB_CHUNK, BLOCK_K=BLOCK_K,
    )
    return scores


class ChunkGatedLinearAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, gate, scale, initial_state):
        seq_len = query.shape[2]
        query, key, value, gate = (pad_sequence(x, CHUNK_SIZE) for x in (query, key, value, gate))
        batch_size, num_heads, padded_len, head_dim = query.shape
        value_dim = value.shape[-1]
        num_chunks = padded_len // CHUNK_SIZE

        gate_cumsum = chunk_cumsum(gate, CHUNK_SIZE)
        states = _chunk_states(key, value, gate_cumsum, initial_state)
        scores = _chunk_scores(query, key, gate_cumsum, scale)
        out = torch.empty_like(value)
        grid = (num_chunks, value_dim // BLOCK_V, batch_size * num_heads)
        _chunk_output_forward[grid](
            query, value, gate_cumsum, states, scores, out, scale, padded_len, num_chunks,
            HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        )

        ctx.save_for_backward(query, key, value, gate, initial_state)
        ctx.scale = scale
        ctx.seq_len = seq_len
        return out[:, :, :seq_len], states[:, :, -1]

    @staticmethod
    def backward(ctx, grad_out, grad_final_state):
        query, key, value, gate, initial_state = ctx.saved_tensors
        batch_size, num_heads, padded_len, head_dim = query.shape
        value_dim = value.shape[-1]
        num_chunks = padded_len // CHUNK_SIZE
        grad_out = pad_sequence(grad_out, CHUNK_SIZE)
        kernel_args = dict(HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V)

        gate_cumsum = chunk_cumsum(gate, CHUNK_SIZE)
        states = _chunk_states(key, value, gate_cumsum, initial_state)
        scores = _chunk_scores(query, key, gate_cumsum, ctx.scale)
        dstates = torch.empty_like(states)
        dstates[:, :, -1] = 0.0 if grad_final_state is None else grad_final_state
        grid = (head_dim // BLOCK_K, value_dim // BLOCK_V, batch_size * num_heads)
        _chunk_state_backward[grid](query, grad_out, gate_cumsum, dstates, ctx.scale, padded_len, num_chunks, **kernel_args)

        dscores = torch.empty_like(scores)
        grid = (num_chunks, batch_size * num_heads)
        _chunk_intra_backward[grid](value, grad_out, dscores, padded_len, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_V=BLOCK_V)

        grad_query = torch.empty(query.shape, device=query.device, dtype=torch.float32)
        grad_key = torch.empty(key.shape, device=key.device, dtype=torch.float32)
        grad_value = torch.empty_like(value)
        grid = (num_chunks, head_dim // BLOCK_K, batch_size * num_heads)
        _chunk_backward_dq[grid](
            key, grad_out, gate_cumsum, states, dscores, grad_query, ctx.scale, padded_len, num_chunks, SUB=SUB_CHUNK, **kernel_args
        )
        _chunk_backward_dk[grid](
            query, value, gate_cumsum, dstates, dscores, grad_key, ctx.scale, padded_len, num_chunks, SUB=SUB_CHUNK, **kernel_args
        )
        grid = (num_chunks, value_dim // BLOCK_V, batch_size * num_heads)
        _chunk_backward_dv[grid](key, grad_out, gate_cumsum, dstates, scores, grad_value, padded_len, num_chunks, **kernel_args)

        grad_gate = torch.empty_like(gate)
        grid = (num_chunks, head_dim // BLOCK_K, batch_size * num_heads)
        _chunk_gate_gradient[grid](
            query, key, grad_query, grad_key, states, dstates, grad_gate, padded_len, num_chunks, **kernel_args
        )

        seq_len = ctx.seq_len
        return (
            grad_query[:, :, :seq_len].to(query.dtype),
            grad_key[:, :, :seq_len].to(key.dtype),
            grad_value[:, :, :seq_len],
            grad_gate[:, :, :seq_len],
            None,
            dstates[:, :, 0],
        )


def _default_initial_state(query, value, initial_state):
    if initial_state is None:
        batch_size, num_heads, _, head_dim = query.shape
        return query.new_zeros(batch_size, num_heads, head_dim, value.shape[-1], dtype=torch.float32)
    return initial_state.float().contiguous()


@torch.compiler.disable
def chunk_gated_linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    scale = scale or query.shape[-1] ** -0.5
    query, key, value = autocast_inputs(query, key, value)
    initial_state = _default_initial_state(query, value, initial_state)
    out, final_state = ChunkGatedLinearAttention.apply(query, key, value, gate, scale, initial_state)
    return (out, final_state) if output_final_state else out


def recurrent_gated_linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    scale = scale or query.shape[-1] ** -0.5
    query, key, value, gate = (x.contiguous() for x in (query, key, value, gate))
    initial_state = _default_initial_state(query, value, initial_state)
    batch_size, num_heads, seq_len, head_dim = query.shape
    value_dim = value.shape[-1]

    out = torch.empty_like(value)
    final_state = torch.empty_like(initial_state)
    grid = (value_dim // BLOCK_V, batch_size * num_heads)
    _recurrent_forward[grid](
        query, key, value, gate, out, initial_state, final_state, scale, seq_len,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, BLOCK_V=BLOCK_V,
    )
    return (out, final_state) if output_final_state else out
