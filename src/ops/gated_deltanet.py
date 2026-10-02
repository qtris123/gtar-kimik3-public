"""
Gated DeltaNet (Yang et al., 2024):
    - The delta rule of DeltaNet with a scalar per-head forget gate,
        S_t = exp(g_t) S_{t-1},   u_t = beta_t (v_t - k_t S_t),   S_t = S_t + k_t^T u_t,   o_t = q_t S_t,
    - With log gates g_t <= 0, write strengths beta_t in (0, 1) and L2-normalised keys (all produced by the layer).

Recurrent form:
    - One program per (batch, head, value block) scans the sequence with the state in registers.
    - This is the decoding form and has no backward.

Chunkwise parallel form:
    - With b_t the cumulative log gate inside a chunk, decay enters the WY representation
        A = (I + tril(diag(beta) (K K^T ⊙ exp(b_t - b_s)), -1))^{-1},  W = A diag(beta exp(b)) K,  U = A diag(beta) V,
    - Queries read the chunk state as q_t exp(b_t) and keys write it as k_t exp(b_end - b_t). All exponents are <= 0.
All tensors are contiguous (batch, heads, seq_len, dim); gate and beta are (batch, heads, seq_len).
"""

import torch
import triton
import triton.language as tl

from .utils import autocast_inputs, pad_sequence, tile_ptrs, triangular_inverse

CHUNK_SIZE = 64
BLOCK_K = 64
BLOCK_V = 64


@triton.jit
def _recurrent_forward(
    Q, K, V, Gate, Beta, Out, InitialState, FinalState,
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
    gate_ptrs = Gate + batch_head * seq_len
    beta_ptrs = Beta + batch_head * seq_len
    out_ptrs = Out + batch_head * seq_len * VALUE_DIM + value_offsets

    state = tl.load(tile_ptrs(InitialState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM))
    for t in range(seq_len):
        q = tl.load(q_ptrs + t * HEAD_DIM).to(tl.float32) * scale
        k = tl.load(k_ptrs + t * HEAD_DIM).to(tl.float32)
        v = tl.load(v_ptrs + t * VALUE_DIM).to(tl.float32)
        gate = tl.load(gate_ptrs + t).to(tl.float32)
        beta = tl.load(beta_ptrs + t).to(tl.float32)
        state = state * tl.exp(gate)
        error = v - tl.sum(k[:, None] * state, axis=0)
        state += k[:, None] * (beta * error)[None, :]
        out = tl.sum(q[:, None] * state, axis=0)
        tl.store(out_ptrs + t * VALUE_DIM, out.to(Out.dtype.element_ty))
    tl.store(tile_ptrs(FinalState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@triton.jit
def _chunk_wy_representation(
    K, V, Gate, Beta, W, U, A,
    seq_len,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets
    beta = tl.load(Beta + batch_head * seq_len + rows).to(tl.float32)
    gate_cumsum = tl.cumsum(tl.load(Gate + batch_head * seq_len + rows).to(tl.float32), axis=0)
    causal = chunk_offsets[:, None] > chunk_offsets[None, :]
    decay = tl.exp(tl.where(causal, gate_cumsum[:, None] - gate_cumsum[None, :], float("-inf")))

    # L = tril(diag(beta) (K K^T ⊙ decay), -1)
    lower = tl.zeros([CHUNK, CHUNK], tl.float32)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        lower += tl.dot((k * beta[:, None]).to(k.dtype), tl.trans(k))
    lower = lower * decay

    # A = (I + L)^{-1} for the strictly lower triangular L
    inverse = triangular_inverse(lower, CHUNK)
    tl.store(tile_ptrs(A, batch_head * seq_len, rows, chunk_offsets, CHUNK), inverse)

    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        w = tl.dot(inverse.to(k.dtype), (k * (beta * tl.exp(gate_cumsum))[:, None]).to(k.dtype))
        tl.store(tile_ptrs(W, batch_head * seq_len, rows, key_offsets, HEAD_DIM), w.to(W.dtype.element_ty))
    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        u = tl.dot(inverse.to(v.dtype), (v * beta[:, None]).to(v.dtype))
        tl.store(tile_ptrs(U, batch_head * seq_len, rows, value_offsets, VALUE_DIM), u.to(U.dtype.element_ty))


@triton.jit
def _chunk_state_forward(
    K, W, U, Gate, States, PseudoValues,
    seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_V: tl.constexpr,
):
    value_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    chunk_offsets = tl.arange(0, CHUNK)

    # States[b, h, i] is the state entering chunk i; States[b, h, 0] holds the initial state on entry
    state_row_base = batch_head * (num_chunks + 1) * HEAD_DIM
    state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))
    for chunk in range(num_chunks):
        rows = chunk * CHUNK + chunk_offsets
        gate = tl.load(Gate + batch_head * seq_len + rows).to(tl.float32)
        gate_cumsum = tl.cumsum(gate, axis=0)
        chunk_decay = tl.sum(gate)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        w = tl.load(tile_ptrs(W, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        u = tl.load(tile_ptrs(U, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        pseudo_values = u - tl.dot(w, state.to(w.dtype))
        tl.store(tile_ptrs(PseudoValues, batch_head * seq_len, rows, value_offsets, VALUE_DIM), pseudo_values.to(PseudoValues.dtype.element_ty))
        k_decayed = (k * tl.exp(chunk_decay - gate_cumsum)[:, None]).to(k.dtype)
        state = state * tl.exp(chunk_decay) + tl.dot(tl.trans(k_decayed), pseudo_values.to(k.dtype))
        tl.store(tile_ptrs(States, state_row_base + (chunk + 1) * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@triton.jit
def _chunk_output_forward(
    Q, K, Gate, PseudoValues, States, Out,
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
    gate_cumsum = tl.cumsum(tl.load(Gate + batch_head * seq_len + rows).to(tl.float32), axis=0)
    causal = chunk_offsets[:, None] >= chunk_offsets[None, :]
    decay = tl.exp(tl.where(causal, gate_cumsum[:, None] - gate_cumsum[None, :], float("-inf")))

    pseudo_values = tl.load(tile_ptrs(PseudoValues, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
    inter = tl.zeros([CHUNK, BLOCK_V], tl.float32)
    scores = tl.zeros([CHUNK, CHUNK], tl.float32)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))
        q_decayed = (q * tl.exp(gate_cumsum)[:, None]).to(q.dtype)
        inter += tl.dot(q_decayed, state.to(q.dtype))
        scores += tl.dot(q, tl.trans(k))

    scores = scores * decay
    out = inter + tl.dot(scores.to(pseudo_values.dtype), pseudo_values)
    tl.store(tile_ptrs(Out, batch_head * seq_len, rows, value_offsets, VALUE_DIM), out.to(Out.dtype.element_ty))


@triton.jit
def _chunk_backward_local_du(
    Q, K, Gate, dOut, dU,
    scale: tl.constexpr, seq_len,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    value_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    gate_cumsum = tl.cumsum(tl.load(Gate + batch_head * seq_len + rows).to(tl.float32), axis=0)
    causal = chunk_offsets[:, None] >= chunk_offsets[None, :]
    decay = tl.exp(tl.where(causal, gate_cumsum[:, None] - gate_cumsum[None, :], float("-inf")))

    scores = tl.zeros([CHUNK, CHUNK], tl.float32)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        scores += tl.dot(q, tl.trans(k))
    scores = scores * decay

    dout = tl.load(tile_ptrs(dOut, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
    du = tl.dot(tl.trans(scores.to(dout.dtype)), dout)
    tl.store(tile_ptrs(dU, batch_head * seq_len, rows, value_offsets, VALUE_DIM), du)


@triton.jit
def _chunk_state_backward(
    Q, K, W, Gate, dOut, dU, dStates,
    scale: tl.constexpr, seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_V: tl.constexpr,
):
    value_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    chunk_offsets = tl.arange(0, CHUNK)

    # dStates[b, h, i] is the gradient of the state entering chunk i; dStates[b, h, num_chunks] is given.
    # dU enters holding the intra-chunk part of du and leaves holding all of it.
    state_row_base = batch_head * (num_chunks + 1) * HEAD_DIM
    dstate = tl.load(tile_ptrs(dStates, state_row_base + num_chunks * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM))
    for i in range(num_chunks):
        chunk = num_chunks - 1 - i
        rows = chunk * CHUNK + chunk_offsets
        gate = tl.load(Gate + batch_head * seq_len + rows).to(tl.float32)
        gate_cumsum = tl.cumsum(gate, axis=0)
        chunk_decay = tl.sum(gate)
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        w = tl.load(tile_ptrs(W, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        dout = tl.load(tile_ptrs(dOut, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        k_decayed = (k * tl.exp(chunk_decay - gate_cumsum)[:, None]).to(k.dtype)
        q_decayed = (q * tl.exp(gate_cumsum)[:, None]).to(q.dtype)
        du = tl.load(tile_ptrs(dU, batch_head * seq_len, rows, value_offsets, VALUE_DIM)) + tl.dot(k_decayed, dstate.to(k.dtype))
        tl.store(tile_ptrs(dU, batch_head * seq_len, rows, value_offsets, VALUE_DIM), du)
        dstate = dstate * tl.exp(chunk_decay) + tl.dot(tl.trans(q_decayed), dout) - tl.dot(tl.trans(w), du.to(w.dtype))
        tl.store(tile_ptrs(dStates, state_row_base + chunk * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), dstate)


@triton.jit
def _chunk_backward_dqkw(
    Q, K, Gate, PseudoValues, dOut, dU, States, dStates, dQ, dK, dW,
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
    gate = tl.load(Gate + batch_head * seq_len + rows).to(tl.float32)
    gate_cumsum = tl.cumsum(gate, axis=0)
    chunk_decay = tl.sum(gate)
    causal = chunk_offsets[:, None] >= chunk_offsets[None, :]
    decay = tl.exp(tl.where(causal, gate_cumsum[:, None] - gate_cumsum[None, :], float("-inf")))

    q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
    k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
    dq = tl.zeros([CHUNK, BLOCK_K], tl.float32)
    dk = tl.zeros([CHUNK, BLOCK_K], tl.float32)
    dw = tl.zeros([CHUNK, BLOCK_K], tl.float32)
    dscores = tl.zeros([CHUNK, CHUNK], tl.float32)
    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        pseudo_values = tl.load(tile_ptrs(PseudoValues, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        dout = tl.load(tile_ptrs(dOut, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        du = tl.load(tile_ptrs(dU, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))
        next_dstate = tl.load(tile_ptrs(dStates, state_row_base + HEAD_DIM, key_offsets, value_offsets, VALUE_DIM))
        dq += tl.dot(dout, tl.trans(state.to(dout.dtype)))
        dk += tl.dot(pseudo_values, tl.trans(next_dstate.to(pseudo_values.dtype)))
        dw -= tl.dot(du.to(dout.dtype), tl.trans(state.to(dout.dtype)))
        dscores += tl.dot(dout, tl.trans(pseudo_values))

    dq = dq * tl.exp(gate_cumsum)[:, None]
    dk = dk * tl.exp(chunk_decay - gate_cumsum)[:, None]
    dscores = dscores * decay
    dq += tl.dot(dscores.to(k.dtype), k)
    dk += tl.dot(tl.trans(dscores.to(q.dtype)), q)
    tl.store(tile_ptrs(dQ, batch_head * seq_len, rows, key_offsets, HEAD_DIM), dq * scale)
    tl.store(tile_ptrs(dK, batch_head * seq_len, rows, key_offsets, HEAD_DIM), dk)
    tl.store(tile_ptrs(dW, batch_head * seq_len, rows, key_offsets, HEAD_DIM), dw)


@triton.jit
def _chunk_backward_wy_representation(
    K, V, Gate, Beta, A, dW, dU, dK, dV, dBeta,
    seq_len,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets
    beta = tl.load(Beta + batch_head * seq_len + rows).to(tl.float32)
    gate_cumsum = tl.cumsum(tl.load(Gate + batch_head * seq_len + rows).to(tl.float32), axis=0)
    causal = chunk_offsets[:, None] > chunk_offsets[None, :]
    decay = tl.exp(tl.where(causal, gate_cumsum[:, None] - gate_cumsum[None, :], float("-inf")))
    inverse = tl.load(tile_ptrs(A, batch_head * seq_len, rows, chunk_offsets, CHUNK))

    # W = A diag(beta exp(b)) K and U = A diag(beta) V give dA = dW (beta exp(b) K)^T + dU (beta V)^T
    dinverse = tl.zeros([CHUNK, CHUNK], tl.float32)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        dw = tl.load(tile_ptrs(dW, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        dinverse += tl.dot(dw.to(k.dtype), tl.trans((k * (beta * tl.exp(gate_cumsum))[:, None]).to(k.dtype)))
    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        du = tl.load(tile_ptrs(dU, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        dinverse += tl.dot(du.to(v.dtype), tl.trans((v * beta[:, None]).to(v.dtype)))

    # A = (I + L)^{-1} gives dL = -A^T dA A^T on the strictly lower triangle; L = tril((beta K) K^T ⊙ decay, -1)
    dlower = -tl.dot(tl.trans(inverse), tl.dot(dinverse, tl.trans(inverse), input_precision="ieee"), input_precision="ieee")
    dlower = dlower * decay

    # d(beta K) = dL K,  d(beta exp(b) K) = A^T dW,  dK += dL^T (beta K) + beta d(beta K) + beta exp(b) d(beta exp(b) K)
    dbeta = tl.zeros([CHUNK], tl.float32)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        dw = tl.load(tile_ptrs(dW, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        dk_beta = tl.dot(dlower.to(k.dtype), k)
        dk_beta_decayed = tl.dot(tl.trans(inverse).to(k.dtype), dw.to(k.dtype))
        dk = tl.load(tile_ptrs(dK, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        dk += tl.dot(tl.trans(dlower).to(k.dtype), (k * beta[:, None]).to(k.dtype))
        dk += dk_beta * beta[:, None] + dk_beta_decayed * (beta * tl.exp(gate_cumsum))[:, None]
        dbeta += tl.sum(dk_beta * k, axis=1) + tl.sum(dk_beta_decayed * k, axis=1) * tl.exp(gate_cumsum)
        tl.store(tile_ptrs(dK, batch_head * seq_len, rows, key_offsets, HEAD_DIM), dk)
    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        du = tl.load(tile_ptrs(dU, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        dv_beta = tl.dot(tl.trans(inverse).to(v.dtype), du.to(v.dtype))
        dbeta += tl.sum(dv_beta * v, axis=1)
        tl.store(tile_ptrs(dV, batch_head * seq_len, rows, value_offsets, VALUE_DIM), dv_beta * beta[:, None])
    tl.store(dBeta + batch_head * seq_len + rows, dbeta)


@triton.jit
def _chunk_gate_gradient(
    Q, dQ, V, dV, States, dStates, dGate,
    seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1)
    rows = chunk * CHUNK + tl.arange(0, CHUNK)
    next_state_row_base = (batch_head * (num_chunks + 1) + chunk + 1) * HEAD_DIM

    # dg_t = sum_{s>=t in the chunk} (q_s dq_s - v_s dv_s) + <S, dS> at the chunk boundary
    terms = tl.zeros([CHUNK], tl.float32)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)).to(tl.float32)
        dq = tl.load(tile_ptrs(dQ, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        terms += tl.sum(q * dq, axis=1)
    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM)).to(tl.float32)
        dv = tl.load(tile_ptrs(dV, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        terms -= tl.sum(v * dv, axis=1)

    boundary = tl.zeros([], tl.float32)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        for value_block in range(VALUE_DIM // BLOCK_V):
            value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
            next_state = tl.load(tile_ptrs(States, next_state_row_base, key_offsets, value_offsets, VALUE_DIM))
            next_dstate = tl.load(tile_ptrs(dStates, next_state_row_base, key_offsets, value_offsets, VALUE_DIM))
            boundary += tl.sum(next_state * next_dstate)
    dgate = tl.cumsum(terms, axis=0, reverse=True) + boundary
    tl.store(dGate + batch_head * seq_len + rows, dgate.to(dGate.dtype.element_ty))


def _wy_representation(key, value, gate, beta):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = value.shape[-1]
    w, u = torch.empty_like(key), torch.empty_like(value)
    inverse = torch.empty(batch_size, num_heads, seq_len, CHUNK_SIZE, device=key.device, dtype=torch.float32)
    grid = (seq_len // CHUNK_SIZE, batch_size * num_heads)
    _chunk_wy_representation[grid](
        key, value, gate, beta, w, u, inverse, seq_len,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )
    return w, u, inverse


def _chunk_states(key, w, u, gate, initial_state):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = u.shape[-1]
    num_chunks = seq_len // CHUNK_SIZE
    states = torch.empty(batch_size, num_heads, num_chunks + 1, head_dim, value_dim, device=key.device, dtype=torch.float32)
    states[:, :, 0] = initial_state
    pseudo_values = torch.empty_like(u)
    grid = (value_dim // BLOCK_V, batch_size * num_heads)
    _chunk_state_forward[grid](
        key, w, u, gate, states, pseudo_values, seq_len, num_chunks,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_V=BLOCK_V,
    )
    return states, pseudo_values


class ChunkGatedDeltaNet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, gate, beta, scale, initial_state):
        seq_len = query.shape[2]
        query, key, value, gate, beta = (pad_sequence(x, CHUNK_SIZE) for x in (query, key, value, gate, beta))
        batch_size, num_heads, padded_len, head_dim = query.shape
        value_dim = value.shape[-1]
        num_chunks = padded_len // CHUNK_SIZE

        w, u, _ = _wy_representation(key, value, gate, beta)
        states, pseudo_values = _chunk_states(key, w, u, gate, initial_state)
        out = torch.empty_like(value)
        grid = (num_chunks, value_dim // BLOCK_V, batch_size * num_heads)
        _chunk_output_forward[grid](
            query, key, gate, pseudo_values, states, out, scale, padded_len, num_chunks,
            HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        )

        ctx.save_for_backward(query, key, value, gate, beta, initial_state)
        ctx.scale = scale
        ctx.seq_len = seq_len
        return out[:, :, :seq_len], states[:, :, -1]

    @staticmethod
    def backward(ctx, grad_out, grad_final_state):
        query, key, value, gate, beta, initial_state = ctx.saved_tensors
        batch_size, num_heads, padded_len, head_dim = query.shape
        value_dim = value.shape[-1]
        num_chunks = padded_len // CHUNK_SIZE
        grad_out = pad_sequence(grad_out, CHUNK_SIZE)
        kernel_args = dict(HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V)

        w, u, inverse = _wy_representation(key, value, gate, beta)
        states, pseudo_values = _chunk_states(key, w, u, gate, initial_state)

        du = torch.empty(u.shape, device=u.device, dtype=torch.float32)
        grid = (num_chunks, value_dim // BLOCK_V, batch_size * num_heads)
        _chunk_backward_local_du[grid](query, key, gate, grad_out, du, ctx.scale, padded_len, **kernel_args)

        dstates = torch.empty_like(states)
        dstates[:, :, -1] = 0.0 if grad_final_state is None else grad_final_state
        grid = (value_dim // BLOCK_V, batch_size * num_heads)
        _chunk_state_backward[grid](
            query, key, w, gate, grad_out, du, dstates, ctx.scale, padded_len, num_chunks,
            HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_V=BLOCK_V, num_warps=8,
        )

        grad_query, grad_key, dw = (torch.empty(key.shape, device=key.device, dtype=torch.float32) for _ in range(3))
        grid = (num_chunks, head_dim // BLOCK_K, batch_size * num_heads)
        _chunk_backward_dqkw[grid](
            query, key, gate, pseudo_values, grad_out, du, states, dstates, grad_query, grad_key, dw,
            ctx.scale, padded_len, num_chunks, **kernel_args,
        )

        grad_value = torch.empty(value.shape, device=value.device, dtype=torch.float32)
        grad_beta = torch.empty(beta.shape, device=beta.device, dtype=torch.float32)
        grid = (num_chunks, batch_size * num_heads)
        _chunk_backward_wy_representation[grid](
            key, value, gate, beta, inverse, dw, du, grad_key, grad_value, grad_beta, padded_len, num_warps=8, **kernel_args
        )
        grad_gate = torch.empty_like(gate)
        _chunk_gate_gradient[grid](
            query, grad_query, value, grad_value, states, dstates, grad_gate, padded_len, num_chunks, **kernel_args
        )

        seq_len = ctx.seq_len
        return (
            grad_query[:, :, :seq_len].to(query.dtype),
            grad_key[:, :, :seq_len].to(key.dtype),
            grad_value[:, :, :seq_len].to(value.dtype),
            grad_gate[:, :, :seq_len],
            grad_beta[:, :, :seq_len].to(beta.dtype),
            None,
            dstates[:, :, 0],
        )


def _default_initial_state(query, value, initial_state):
    if initial_state is None:
        batch_size, num_heads, _, head_dim = query.shape
        return query.new_zeros(batch_size, num_heads, head_dim, value.shape[-1], dtype=torch.float32)
    return initial_state.float().contiguous()


@torch.compiler.disable
def chunk_gated_deltanet(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    scale = scale or query.shape[-1] ** -0.5
    query, key, value = autocast_inputs(query, key, value)
    initial_state = _default_initial_state(query, value, initial_state)
    out, final_state = ChunkGatedDeltaNet.apply(query, key, value, gate, beta, scale, initial_state)
    return (out, final_state) if output_final_state else out


def recurrent_gated_deltanet(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    scale = scale or query.shape[-1] ** -0.5
    query, key, value, gate, beta = (x.contiguous() for x in (query, key, value, gate, beta))
    initial_state = _default_initial_state(query, value, initial_state)
    batch_size, num_heads, seq_len, head_dim = query.shape
    value_dim = value.shape[-1]

    out = torch.empty_like(value)
    final_state = torch.empty_like(initial_state)
    grid = (value_dim // BLOCK_V, batch_size * num_heads)
    _recurrent_forward[grid](
        query, key, value, gate, beta, out, initial_state, final_state, scale, seq_len,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, BLOCK_V=BLOCK_V,
    )
    return (out, final_state) if output_final_state else out
