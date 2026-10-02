"""
Kimi Delta Attention (Kimi Linear, 2025): the delta rule with a per-channel forget gate on the key dimension,

    S_t = diag(exp(g_t)) S_{t-1},   u_t = beta_t (v_t - k_t S_t),   S_t = S_t + k_t^T u_t,   o_t = q_t S_t,

with log gates g_t <= 0 (one per key channel), write strengths beta_t in (0, 1) and L2-normalised keys.

Recurrent form:
    - One program per (batch, head, value block) scans the sequence with the state in registers.
    - This is the decoding form and has no backward.

Chunkwise parallel form:
    - With b_t the per-channel cumulative log gate inside a chunk, keys decayed towards the chunk start
      (k_t exp(b_t)) and towards the chunk end (k_t exp(b_end - b_t)) play the roles they play in gated linear
      attention, and the WY representation of DeltaNet is built from the gated kernel matrix
      P_ts = sum_k k_tk k_sk exp(b_tk - b_sk):
        A = (I + tril(diag(beta) P, -1))^{-1},   W = A diag(beta) (K exp(b)),   U = A diag(beta) V,   u = U - W S_i.
    - P and the scores Q K^T exp(b_t - b_s) are assembled from SUB_CHUNK x SUB_CHUNK blocks anchored at the end of
      their key sub-chunk (secondary chunking), so no exponent grows with the chunk size.
    - The gate gradient collects q dq for the query roles, +k dk for the exp(+b) key roles and -k dk for the
      exp(-b) key roles, then a chunk-local reverse cumsum plus rowsum(S dS) at the chunk boundary.
All tensors are contiguous (batch, heads, seq_len, dim); gate is (batch, heads, seq_len, head_dim), beta is
(batch, heads, seq_len).
"""

import torch
import triton
import triton.language as tl

from .utils import autocast_inputs, chunk_cumsum, gated_scores_backward_key, gated_scores_backward_left, gated_scores_diagonal, pad_sequence, tile_ptrs, triangular_inverse

CHUNK_SIZE = 64
SUB_CHUNK = 16
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
    gate_ptrs = Gate + batch_head * seq_len * HEAD_DIM + key_offsets
    v_ptrs = V + batch_head * seq_len * VALUE_DIM + value_offsets
    beta_ptrs = Beta + batch_head * seq_len
    out_ptrs = Out + batch_head * seq_len * VALUE_DIM + value_offsets

    state = tl.load(tile_ptrs(InitialState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM))
    for t in range(seq_len):
        q = tl.load(q_ptrs + t * HEAD_DIM).to(tl.float32) * scale
        k = tl.load(k_ptrs + t * HEAD_DIM).to(tl.float32)
        v = tl.load(v_ptrs + t * VALUE_DIM).to(tl.float32)
        gate = tl.load(gate_ptrs + t * HEAD_DIM).to(tl.float32)
        beta = tl.load(beta_ptrs + t).to(tl.float32)
        state = state * tl.exp(gate)[:, None]
        error = v - tl.sum(k[:, None] * state, axis=0)
        state += k[:, None] * (beta * error)[None, :]
        out = tl.sum(q[:, None] * state, axis=0)
        tl.store(out_ptrs + t * VALUE_DIM, out.to(Out.dtype.element_ty))
    tl.store(tile_ptrs(FinalState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@triton.jit
def _chunk_kernel_matrix(
    Left, K, GateCumsum, Out,
    scale: tl.constexpr, seq_len,
    HEAD_DIM: tl.constexpr, CHUNK: tl.constexpr, SUB: tl.constexpr, BLOCK_K: tl.constexpr, STRICT: tl.constexpr,
):
    # Out[t, s] = scale * sum_k Left[t, k] K[s, k] exp(b_tk - b_sk) on the (strictly) lower triangle of each chunk
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1)
    sub_offsets = tl.arange(0, SUB)

    for i in tl.static_range(CHUNK // SUB):
        left_rows = chunk * CHUNK + i * SUB + sub_offsets
        for j in tl.static_range(i + 1):
            key_rows = chunk * CHUNK + j * SUB + sub_offsets
            anchor_row = chunk * CHUNK + j * SUB + SUB - 1
            block = tl.zeros([SUB, SUB], tl.float32)
            for key_block in range(HEAD_DIM // BLOCK_K):
                key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
                left = tl.load(tile_ptrs(Left, batch_head * seq_len, left_rows, key_offsets, HEAD_DIM)) * scale
                k = tl.load(tile_ptrs(K, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM))
                left_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, left_rows, key_offsets, HEAD_DIM))
                key_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM))
                anchor = tl.load(GateCumsum + (batch_head * seq_len + anchor_row) * HEAD_DIM + key_offsets)
                if i == j:  # the diagonal block is built row by row in fp32, so no exponent is ever positive
                    block += gated_scores_diagonal(left.to(tl.float32), k.to(tl.float32), left_cumsum, key_cumsum)
                else:
                    left_decayed = (left * tl.exp(left_cumsum - anchor[None, :])).to(k.dtype)
                    k_decayed = (k * tl.exp(anchor[None, :] - key_cumsum)).to(k.dtype)
                    block += tl.dot(left_decayed, tl.trans(k_decayed))
            if i == j:
                if STRICT:
                    block = tl.where(sub_offsets[:, None] > sub_offsets[None, :], block, 0.0)
                else:
                    block = tl.where(sub_offsets[:, None] >= sub_offsets[None, :], block, 0.0)
            tl.store(tile_ptrs(Out, batch_head * seq_len, left_rows, j * SUB + sub_offsets, CHUNK), block)


@triton.jit
def _chunk_wy_representation(
    K, V, GateCumsum, Beta, KernelMatrix, W, U, A,
    seq_len,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets
    beta = tl.load(Beta + batch_head * seq_len + rows).to(tl.float32)

    # A = (I + L)^{-1} for the strictly lower triangular L = diag(beta) P
    lower = tl.load(tile_ptrs(KernelMatrix, batch_head * seq_len, rows, chunk_offsets, CHUNK)) * beta[:, None]
    inverse = triangular_inverse(lower, CHUNK)
    tl.store(tile_ptrs(A, batch_head * seq_len, rows, chunk_offsets, CHUNK), inverse)

    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        w = tl.dot(inverse.to(k.dtype), (k * (beta[:, None] * tl.exp(gate_cumsum))).to(k.dtype))
        tl.store(tile_ptrs(W, batch_head * seq_len, rows, key_offsets, HEAD_DIM), w.to(W.dtype.element_ty))
    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        u = tl.dot(inverse.to(v.dtype), (v * beta[:, None]).to(v.dtype))
        tl.store(tile_ptrs(U, batch_head * seq_len, rows, value_offsets, VALUE_DIM), u.to(U.dtype.element_ty))


@triton.jit
def _chunk_state_forward(
    K, W, U, GateCumsum, States, PseudoValues,
    seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    value_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    chunk_offsets = tl.arange(0, CHUNK)

    # States[b, h, i] is the state entering chunk i; States[b, h, 0] holds the initial state on entry.
    # The state is read back from States one key block at a time so every tile is (CHUNK x BLOCK) whatever the head dim.
    for chunk in range(num_chunks):
        rows = chunk * CHUNK + chunk_offsets
        state_row_base = (batch_head * (num_chunks + 1) + chunk) * HEAD_DIM
        pseudo_values = tl.load(tile_ptrs(U, batch_head * seq_len, rows, value_offsets, VALUE_DIM)).to(tl.float32)
        for key_block in range(HEAD_DIM // BLOCK_K):
            key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
            w = tl.load(tile_ptrs(W, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
            state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))
            pseudo_values -= tl.dot(w, state.to(w.dtype))
        tl.store(tile_ptrs(PseudoValues, batch_head * seq_len, rows, value_offsets, VALUE_DIM), pseudo_values.to(PseudoValues.dtype.element_ty))
        for key_block in range(HEAD_DIM // BLOCK_K):
            key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
            k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
            gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
            chunk_decay = tl.load(GateCumsum + (batch_head * seq_len + chunk * CHUNK + CHUNK - 1) * HEAD_DIM + key_offsets)
            state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))
            k_decayed = (k * tl.exp(chunk_decay[None, :] - gate_cumsum)).to(k.dtype)
            state = state * tl.exp(chunk_decay)[:, None] + tl.dot(tl.trans(k_decayed), pseudo_values.to(k.dtype))
            tl.store(tile_ptrs(States, state_row_base + HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@triton.jit
def _chunk_output_forward(
    Q, GateCumsum, PseudoValues, States, Scores, Out,
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

    pseudo_values = tl.load(tile_ptrs(PseudoValues, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
    scores = tl.load(tile_ptrs(Scores, batch_head * seq_len, rows, chunk_offsets, CHUNK))
    out = tl.dot(scores.to(pseudo_values.dtype), pseudo_values)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
        gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))
        q_decayed = (q * tl.exp(gate_cumsum)).to(q.dtype)
        out += tl.dot(q_decayed, state.to(q.dtype))
    tl.store(tile_ptrs(Out, batch_head * seq_len, rows, value_offsets, VALUE_DIM), out.to(Out.dtype.element_ty))


@triton.jit
def _chunk_intra_backward(
    PseudoValues, dOut, dScores,
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
        pseudo_values = tl.load(tile_ptrs(PseudoValues, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        dout = tl.load(tile_ptrs(dOut, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        dscores += tl.dot(dout, tl.trans(pseudo_values))
    dscores = tl.where(chunk_offsets[:, None] >= chunk_offsets[None, :], dscores, 0.0)
    tl.store(tile_ptrs(dScores, batch_head * seq_len, rows, chunk_offsets, CHUNK), dscores)


@triton.jit
def _chunk_state_backward(
    Q, K, W, GateCumsum, dOut, Scores, dU, dStates,
    scale: tl.constexpr, seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    value_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    chunk_offsets = tl.arange(0, CHUNK)

    # dStates[b, h, i] is the gradient of the state entering chunk i; dStates[b, h, num_chunks] is given
    for i in range(num_chunks):
        chunk = num_chunks - 1 - i
        rows = chunk * CHUNK + chunk_offsets
        next_state_row_base = (batch_head * (num_chunks + 1) + chunk + 1) * HEAD_DIM
        dout = tl.load(tile_ptrs(dOut, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        scores = tl.load(tile_ptrs(Scores, batch_head * seq_len, rows, chunk_offsets, CHUNK))
        du = tl.dot(tl.trans(scores.to(dout.dtype)), dout)
        for key_block in range(HEAD_DIM // BLOCK_K):
            key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
            k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
            gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
            chunk_decay = tl.load(GateCumsum + (batch_head * seq_len + chunk * CHUNK + CHUNK - 1) * HEAD_DIM + key_offsets)
            next_dstate = tl.load(tile_ptrs(dStates, next_state_row_base, key_offsets, value_offsets, VALUE_DIM))
            k_decayed = (k * tl.exp(chunk_decay[None, :] - gate_cumsum)).to(k.dtype)
            du += tl.dot(k_decayed, next_dstate.to(k.dtype))
        tl.store(tile_ptrs(dU, batch_head * seq_len, rows, value_offsets, VALUE_DIM), du)
        for key_block in range(HEAD_DIM // BLOCK_K):
            key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
            q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
            w = tl.load(tile_ptrs(W, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
            gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
            chunk_decay = tl.load(GateCumsum + (batch_head * seq_len + chunk * CHUNK + CHUNK - 1) * HEAD_DIM + key_offsets)
            next_dstate = tl.load(tile_ptrs(dStates, next_state_row_base, key_offsets, value_offsets, VALUE_DIM))
            q_decayed = (q * tl.exp(gate_cumsum)).to(q.dtype)
            dstate = next_dstate * tl.exp(chunk_decay)[:, None] + tl.dot(tl.trans(q_decayed), dout) - tl.dot(tl.trans(w), du.to(w.dtype))
            tl.store(tile_ptrs(dStates, next_state_row_base - HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), dstate)


@triton.jit
def _chunk_backward_dq(
    Q, K, dOut, GateCumsum, States, dScores, dQ, GateTerms,
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

        dq = dq * scale
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, query_rows, key_offsets, HEAD_DIM)).to(tl.float32)
        tl.store(tile_ptrs(dQ, batch_head * seq_len, query_rows, key_offsets, HEAD_DIM), dq)
        tl.store(tile_ptrs(GateTerms, batch_head * seq_len, query_rows, key_offsets, HEAD_DIM), q * dq)


@triton.jit
def _chunk_backward_dk(
    Q, K, PseudoValues, dU, GateCumsum, States, dStates, dScores, dK, dW, GateTerms,
    scale: tl.constexpr, seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, SUB: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    # the key roles handled here all carry exp(-b): the state write k exp(b_end - b) and the score keys
    chunk = tl.program_id(0)
    key_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    sub_offsets = tl.arange(0, SUB)
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    state_row_base = (batch_head * (num_chunks + 1) + chunk) * HEAD_DIM
    chunk_decay = tl.load(GateCumsum + (batch_head * seq_len + chunk * CHUNK + CHUNK - 1) * HEAD_DIM + key_offsets)

    for j in tl.static_range(CHUNK // SUB):
        key_rows = chunk * CHUNK + j * SUB + sub_offsets
        anchor_row = chunk * CHUNK + j * SUB + SUB - 1
        key_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM))
        anchor = tl.load(GateCumsum + (batch_head * seq_len + anchor_row) * HEAD_DIM + key_offsets)

        dk = tl.zeros([SUB, BLOCK_K], tl.float32)
        dw = tl.zeros([SUB, BLOCK_K], tl.float32)
        for value_block in range(VALUE_DIM // BLOCK_V):
            value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
            pseudo_values = tl.load(tile_ptrs(PseudoValues, batch_head * seq_len, key_rows, value_offsets, VALUE_DIM))
            du = tl.load(tile_ptrs(dU, batch_head * seq_len, key_rows, value_offsets, VALUE_DIM))
            state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))
            next_dstate = tl.load(tile_ptrs(dStates, state_row_base + HEAD_DIM, key_offsets, value_offsets, VALUE_DIM))
            dk += tl.dot(pseudo_values, tl.trans(next_dstate.to(pseudo_values.dtype)))
            dw -= tl.dot(du.to(pseudo_values.dtype), tl.trans(state.to(pseudo_values.dtype)))
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

        k = tl.load(tile_ptrs(K, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM)).to(tl.float32)
        gate_terms = tl.load(tile_ptrs(GateTerms, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM))
        tl.store(tile_ptrs(dK, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM), dk)
        tl.store(tile_ptrs(dW, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM), dw)
        tl.store(tile_ptrs(GateTerms, batch_head * seq_len, key_rows, key_offsets, HEAD_DIM), gate_terms - k * dk)


@triton.jit
def _chunk_backward_wy_representation(
    K, V, GateCumsum, Beta, A, KernelMatrix, dW, dU, dK, dV, dBeta, dKernelMatrix, GateTerms,
    seq_len,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets
    beta = tl.load(Beta + batch_head * seq_len + rows).to(tl.float32)
    inverse = tl.load(tile_ptrs(A, batch_head * seq_len, rows, chunk_offsets, CHUNK))
    kernel_matrix = tl.load(tile_ptrs(KernelMatrix, batch_head * seq_len, rows, chunk_offsets, CHUNK))

    # W = A diag(beta) (K exp(b)) and U = A diag(beta) V give dA = dW (beta K exp(b))^T + dU (beta V)^T
    dinverse = tl.zeros([CHUNK, CHUNK], tl.float32)
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        dw = tl.load(tile_ptrs(dW, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        dinverse += tl.dot(dw.to(k.dtype), tl.trans((k * (beta[:, None] * tl.exp(gate_cumsum))).to(k.dtype)))
    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        du = tl.load(tile_ptrs(dU, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        dinverse += tl.dot(du.to(v.dtype), tl.trans((v * beta[:, None]).to(v.dtype)))

    # A = (I + L)^{-1} with L = diag(beta) P gives dL = -A^T dA A^T on the strictly lower triangle, dP = beta dL
    dlower = -tl.dot(tl.trans(inverse), tl.dot(dinverse, tl.trans(inverse), input_precision="ieee"), input_precision="ieee")
    dlower = dlower * tl.where(chunk_offsets[:, None] > chunk_offsets[None, :], 1.0, 0.0)
    tl.store(tile_ptrs(dKernelMatrix, batch_head * seq_len, rows, chunk_offsets, CHUNK), dlower * beta[:, None])
    dbeta = tl.sum(dlower * kernel_matrix, axis=1)

    # the key role in W carries exp(+b): dK += beta exp(b) A^T dW and it adds k dk to the gate terms
    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        dw = tl.load(tile_ptrs(dW, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        dk_beta_decayed = tl.dot(tl.trans(inverse).to(k.dtype), dw.to(k.dtype))
        dk = dk_beta_decayed * (beta[:, None] * tl.exp(gate_cumsum))
        dbeta += tl.sum(dk_beta_decayed * k * tl.exp(gate_cumsum), axis=1)
        previous_dk = tl.load(tile_ptrs(dK, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        gate_terms = tl.load(tile_ptrs(GateTerms, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        tl.store(tile_ptrs(dK, batch_head * seq_len, rows, key_offsets, HEAD_DIM), previous_dk + dk)
        tl.store(tile_ptrs(GateTerms, batch_head * seq_len, rows, key_offsets, HEAD_DIM), gate_terms + k * dk)
    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        du = tl.load(tile_ptrs(dU, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
        dv_beta = tl.dot(tl.trans(inverse).to(v.dtype), du.to(v.dtype))
        dbeta += tl.sum(dv_beta * v, axis=1)
        tl.store(tile_ptrs(dV, batch_head * seq_len, rows, value_offsets, VALUE_DIM), dv_beta * beta[:, None])
    tl.store(dBeta + batch_head * seq_len + rows, dbeta)


@triton.jit
def _chunk_backward_kernel_matrix(
    K, GateCumsum, dKernelMatrix, dK, GateTerms,
    seq_len,
    HEAD_DIM: tl.constexpr, CHUNK: tl.constexpr, SUB: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # P_ts = sum_k k_tk exp(b_tk - anchor_k) k_sk exp(anchor_k - b_sk): the first factor carries exp(+b), the second exp(-b)
    chunk = tl.program_id(0)
    key_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    sub_offsets = tl.arange(0, SUB)
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)

    for i in tl.static_range(CHUNK // SUB):
        rows_i = chunk * CHUNK + i * SUB + sub_offsets
        cumsum_i = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows_i, key_offsets, HEAD_DIM))
        dk = tl.zeros([SUB, BLOCK_K], tl.float32)
        for j in tl.static_range(i + 1):
            rows_j = chunk * CHUNK + j * SUB + sub_offsets
            anchor = tl.load(GateCumsum + (batch_head * seq_len + chunk * CHUNK + j * SUB + SUB - 1) * HEAD_DIM + key_offsets)
            k_j = tl.load(tile_ptrs(K, batch_head * seq_len, rows_j, key_offsets, HEAD_DIM))
            cumsum_j = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows_j, key_offsets, HEAD_DIM))
            dblock = tl.load(tile_ptrs(dKernelMatrix, batch_head * seq_len, rows_i, j * SUB + sub_offsets, CHUNK))
            if i == j:
                dk += gated_scores_backward_left(dblock, k_j.to(tl.float32), cumsum_i, cumsum_j)
            else:
                k_decayed = (k_j * tl.exp(anchor[None, :] - cumsum_j)).to(k_j.dtype)
                dk += tl.dot(dblock.to(k_j.dtype), k_decayed) * tl.exp(cumsum_i - anchor[None, :])
        k_i = tl.load(tile_ptrs(K, batch_head * seq_len, rows_i, key_offsets, HEAD_DIM)).to(tl.float32)
        previous_dk = tl.load(tile_ptrs(dK, batch_head * seq_len, rows_i, key_offsets, HEAD_DIM))
        gate_terms = tl.load(tile_ptrs(GateTerms, batch_head * seq_len, rows_i, key_offsets, HEAD_DIM))
        tl.store(tile_ptrs(dK, batch_head * seq_len, rows_i, key_offsets, HEAD_DIM), previous_dk + dk)
        tl.store(tile_ptrs(GateTerms, batch_head * seq_len, rows_i, key_offsets, HEAD_DIM), gate_terms + k_i * dk)

    for j in tl.static_range(CHUNK // SUB):
        rows_j = chunk * CHUNK + j * SUB + sub_offsets
        anchor = tl.load(GateCumsum + (batch_head * seq_len + chunk * CHUNK + j * SUB + SUB - 1) * HEAD_DIM + key_offsets)
        cumsum_j = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows_j, key_offsets, HEAD_DIM))
        dk = tl.zeros([SUB, BLOCK_K], tl.float32)
        for i in tl.static_range(j, CHUNK // SUB):
            rows_i = chunk * CHUNK + i * SUB + sub_offsets
            k_i = tl.load(tile_ptrs(K, batch_head * seq_len, rows_i, key_offsets, HEAD_DIM))
            cumsum_i = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows_i, key_offsets, HEAD_DIM))
            dblock = tl.load(tile_ptrs(dKernelMatrix, batch_head * seq_len, rows_i, j * SUB + sub_offsets, CHUNK))
            if i == j:
                dk += gated_scores_backward_key(dblock, k_i.to(tl.float32), cumsum_i, cumsum_j)
            else:
                k_decayed = (k_i * tl.exp(cumsum_i - anchor[None, :])).to(k_i.dtype)
                dk += tl.dot(tl.trans(dblock.to(k_i.dtype)), k_decayed) * tl.exp(anchor[None, :] - cumsum_j)
        k_j = tl.load(tile_ptrs(K, batch_head * seq_len, rows_j, key_offsets, HEAD_DIM)).to(tl.float32)
        previous_dk = tl.load(tile_ptrs(dK, batch_head * seq_len, rows_j, key_offsets, HEAD_DIM))
        gate_terms = tl.load(tile_ptrs(GateTerms, batch_head * seq_len, rows_j, key_offsets, HEAD_DIM))
        tl.store(tile_ptrs(dK, batch_head * seq_len, rows_j, key_offsets, HEAD_DIM), previous_dk + dk)
        tl.store(tile_ptrs(GateTerms, batch_head * seq_len, rows_j, key_offsets, HEAD_DIM), gate_terms - k_j * dk)


@triton.jit
def _chunk_gate_gradient(
    GateTerms, States, dStates, dGate,
    seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    key_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    rows = chunk * CHUNK + tl.arange(0, CHUNK)
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    next_state_row_base = (batch_head * (num_chunks + 1) + chunk + 1) * HEAD_DIM

    # dg_t = sum_{s>=t in the chunk} gate_terms_s + rowsum(S ⊙ dS) at the chunk boundary
    gate_terms = tl.load(tile_ptrs(GateTerms, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
    dgate = tl.cumsum(gate_terms, axis=0, reverse=True)
    boundary = tl.zeros([BLOCK_K], tl.float32)
    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        next_state = tl.load(tile_ptrs(States, next_state_row_base, key_offsets, value_offsets, VALUE_DIM))
        next_dstate = tl.load(tile_ptrs(dStates, next_state_row_base, key_offsets, value_offsets, VALUE_DIM))
        boundary += tl.sum(next_state * next_dstate, axis=1)
    dgate = dgate + boundary[None, :]
    tl.store(tile_ptrs(dGate, batch_head * seq_len, rows, key_offsets, HEAD_DIM), dgate.to(dGate.dtype.element_ty))


def _kernel_matrix(left, key, gate_cumsum, scale, strict):
    batch_size, num_heads, seq_len, head_dim = key.shape
    out = torch.zeros(batch_size, num_heads, seq_len, CHUNK_SIZE, device=key.device, dtype=torch.float32)
    grid = (seq_len // CHUNK_SIZE, batch_size * num_heads)
    _chunk_kernel_matrix[grid](
        left, key, gate_cumsum, out, scale, seq_len,
        HEAD_DIM=head_dim, CHUNK=CHUNK_SIZE, SUB=SUB_CHUNK, BLOCK_K=BLOCK_K, STRICT=strict,
    )
    return out


def _wy_representation(key, value, gate_cumsum, beta, kernel_matrix):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = value.shape[-1]
    w, u = torch.empty_like(key), torch.empty_like(value)
    inverse = torch.empty_like(kernel_matrix)
    grid = (seq_len // CHUNK_SIZE, batch_size * num_heads)
    _chunk_wy_representation[grid](
        key, value, gate_cumsum, beta, kernel_matrix, w, u, inverse, seq_len,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )
    return w, u, inverse


def _chunk_states(key, w, u, gate_cumsum, initial_state):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = u.shape[-1]
    num_chunks = seq_len // CHUNK_SIZE
    states = torch.empty(batch_size, num_heads, num_chunks + 1, head_dim, value_dim, device=key.device, dtype=torch.float32)
    states[:, :, 0] = initial_state
    pseudo_values = torch.empty_like(u)
    grid = (value_dim // BLOCK_V, batch_size * num_heads)
    _chunk_state_forward[grid](
        key, w, u, gate_cumsum, states, pseudo_values, seq_len, num_chunks,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, num_stages=2,
    )
    return states, pseudo_values


class ChunkKimiDeltaAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, gate, beta, scale, initial_state):
        seq_len = query.shape[2]
        query, key, value, gate, beta = (pad_sequence(x, CHUNK_SIZE) for x in (query, key, value, gate, beta))
        batch_size, num_heads, padded_len, head_dim = query.shape
        value_dim = value.shape[-1]
        num_chunks = padded_len // CHUNK_SIZE

        gate_cumsum = chunk_cumsum(gate, CHUNK_SIZE)
        kernel_matrix = _kernel_matrix(key, key, gate_cumsum, 1.0, strict=True)
        w, u, _ = _wy_representation(key, value, gate_cumsum, beta, kernel_matrix)
        states, pseudo_values = _chunk_states(key, w, u, gate_cumsum, initial_state)
        scores = _kernel_matrix(query, key, gate_cumsum, scale, strict=False)
        out = torch.empty_like(value)
        grid = (num_chunks, value_dim // BLOCK_V, batch_size * num_heads)
        _chunk_output_forward[grid](
            query, gate_cumsum, pseudo_values, states, scores, out, scale, padded_len, num_chunks,
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

        gate_cumsum = chunk_cumsum(gate, CHUNK_SIZE)
        kernel_matrix = _kernel_matrix(key, key, gate_cumsum, 1.0, strict=True)
        w, u, inverse = _wy_representation(key, value, gate_cumsum, beta, kernel_matrix)
        states, pseudo_values = _chunk_states(key, w, u, gate_cumsum, initial_state)
        scores = _kernel_matrix(query, key, gate_cumsum, ctx.scale, strict=False)

        dscores = torch.empty_like(scores)
        grid = (num_chunks, batch_size * num_heads)
        _chunk_intra_backward[grid](pseudo_values, grad_out, dscores, padded_len, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_V=BLOCK_V)

        du = torch.empty(u.shape, device=u.device, dtype=torch.float32)
        dstates = torch.empty_like(states)
        dstates[:, :, -1] = 0.0 if grad_final_state is None else grad_final_state
        grid = (value_dim // BLOCK_V, batch_size * num_heads)
        _chunk_state_backward[grid](
            query, key, w, gate_cumsum, grad_out, scores, du, dstates, ctx.scale, padded_len, num_chunks,
            num_stages=2, **kernel_args,
        )

        grad_query, grad_key, dw, gate_terms = (torch.empty(key.shape, device=key.device, dtype=torch.float32) for _ in range(4))
        grid = (num_chunks, head_dim // BLOCK_K, batch_size * num_heads)
        _chunk_backward_dq[grid](
            query, key, grad_out, gate_cumsum, states, dscores, grad_query, gate_terms,
            ctx.scale, padded_len, num_chunks, SUB=SUB_CHUNK, **kernel_args,
        )
        _chunk_backward_dk[grid](
            query, key, pseudo_values, du, gate_cumsum, states, dstates, dscores, grad_key, dw, gate_terms,
            ctx.scale, padded_len, num_chunks, SUB=SUB_CHUNK, **kernel_args,
        )

        grad_value = torch.empty(value.shape, device=value.device, dtype=torch.float32)
        grad_beta = torch.empty(beta.shape, device=beta.device, dtype=torch.float32)
        dkernel_matrix = torch.empty_like(kernel_matrix)
        _chunk_backward_wy_representation[(num_chunks, batch_size * num_heads)](
            key, value, gate_cumsum, beta, inverse, kernel_matrix, dw, du, grad_key, grad_value, grad_beta, dkernel_matrix, gate_terms,
            padded_len, num_warps=8, **kernel_args,
        )
        _chunk_backward_kernel_matrix[grid](
            key, gate_cumsum, dkernel_matrix, grad_key, gate_terms, padded_len,
            HEAD_DIM=head_dim, CHUNK=CHUNK_SIZE, SUB=SUB_CHUNK, BLOCK_K=BLOCK_K,
        )
        grad_gate = torch.empty_like(gate)
        _chunk_gate_gradient[grid](gate_terms, states, dstates, grad_gate, padded_len, num_chunks, **kernel_args)

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


def _reference(query, key, value, gate, beta, scale, initial_state):
    query, key, value, gate, beta = (x.float() for x in (query, key, value, gate, beta))
    state = initial_state
    outputs = []
    for t in range(query.shape[2]):
        k, v = key[:, :, t], value[:, :, t]
        state = state * gate[:, :, t].exp()[..., None]
        error = v - torch.einsum("bhk,bhkv->bhv", k, state)
        state = state + k[..., :, None] * (beta[:, :, t, None] * error)[..., None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, :, t] * scale, state))
    return torch.stack(outputs, 2).to(value.dtype), state


@torch.compiler.disable
def chunk_kimi_delta_attention(
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
    if not query.is_cuda:
        out, final_state = _reference(query, key, value, gate, beta, scale, initial_state)
    else:
        out, final_state = ChunkKimiDeltaAttention.apply(query, key, value, gate, beta, scale, initial_state)
    return (out, final_state) if output_final_state else out


def recurrent_kimi_delta_attention(
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
    if not query.is_cuda:
        out, final_state = _reference(query, key, value, gate, beta, scale, initial_state)
        return (out, final_state) if output_final_state else out
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
