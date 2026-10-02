"""
FlashAttention-2 for causal grouped-query attention, written in Triton.

The (seq_len x seq_len) attention matrix is never materialised.
Each program owns one block of queries (forward, dQ) or one block of keys (dK, dV), streams the other side through SRAM in blocks and keeps running softmax statistics, so memory traffic stays linear in the sequence length.
All tensors are contiguous (batch, heads, seq_len, head_dim).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

FORWARD_CONFIG = dict(BLOCK_M=128, BLOCK_N=32, num_warps=4, num_stages=2)
BACKWARD_DKDV_CONFIG = dict(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=2)
BACKWARD_DQ_CONFIG = dict(BLOCK_M=128, BLOCK_N=32, num_warps=4, num_stages=2)


@triton.jit
def _block_ptrs(base, batch_head, seq_len, row_offsets, HEAD_DIM: tl.constexpr):
    return base + (batch_head * seq_len + row_offsets[:, None]) * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]


@triton.jit
def _flash_attention_forward(
    Q, K, V, Out, LSE,
    softmax_scale: tl.constexpr,
    num_heads, num_kv_heads, seq_len,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch, head = batch_head // num_heads, batch_head % num_heads
    batch_kv_head = batch * num_kv_heads + head // (num_heads // num_kv_heads)

    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    q = tl.load(_block_ptrs(Q, batch_head, seq_len, query_offsets, HEAD_DIM))

    running_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    for key_start in range(0, (query_block + 1) * BLOCK_M, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        k = tl.load(_block_ptrs(K, batch_kv_head, seq_len, key_offsets, HEAD_DIM))
        v = tl.load(_block_ptrs(V, batch_kv_head, seq_len, key_offsets, HEAD_DIM))

        scores = tl.dot(q, tl.trans(k)) * softmax_scale
        scores = tl.where(key_offsets[None, :] <= query_offsets[:, None], scores, float("-inf"))

        block_max = tl.maximum(running_max, tl.max(scores, axis=1))
        probs = tl.exp(scores - block_max[:, None])
        rescale = tl.exp(running_max - block_max)
        running_sum = running_sum * rescale + tl.sum(probs, axis=1)
        acc = acc * rescale[:, None] + tl.dot(probs.to(v.dtype), v)
        running_max = block_max

    out = acc / running_sum[:, None]
    tl.store(_block_ptrs(Out, batch_head, seq_len, query_offsets, HEAD_DIM), out.to(Out.dtype.element_ty))
    tl.store(LSE + batch_head * seq_len + query_offsets, running_max + tl.log(running_sum))


@triton.jit
def _flash_attention_backward_preprocess(
    Out, dOut, Delta,
    seq_len,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    out = tl.load(_block_ptrs(Out, batch_head, seq_len, query_offsets, HEAD_DIM)).to(tl.float32)
    dout = tl.load(_block_ptrs(dOut, batch_head, seq_len, query_offsets, HEAD_DIM)).to(tl.float32)
    tl.store(Delta + batch_head * seq_len + query_offsets, tl.sum(out * dout, axis=1))


@triton.jit
def _flash_attention_backward_dkdv(
    Q, K, V, dOut, dK, dV, LSE, Delta,
    softmax_scale: tl.constexpr,
    num_heads, num_kv_heads, seq_len,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    key_block = tl.program_id(0)
    batch_kv_head = tl.program_id(1)
    batch, kv_head = batch_kv_head // num_kv_heads, batch_kv_head % num_kv_heads
    group_size = num_heads // num_kv_heads

    key_offsets = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.load(_block_ptrs(K, batch_kv_head, seq_len, key_offsets, HEAD_DIM))
    v = tl.load(_block_ptrs(V, batch_kv_head, seq_len, key_offsets, HEAD_DIM))

    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)

    first_query_start = key_block * BLOCK_N // BLOCK_M * BLOCK_M
    for head in range(kv_head * group_size, (kv_head + 1) * group_size):
        batch_head = batch * num_heads + head
        for query_start in range(first_query_start, seq_len, BLOCK_M):
            query_offsets = query_start + tl.arange(0, BLOCK_M)
            q = tl.load(_block_ptrs(Q, batch_head, seq_len, query_offsets, HEAD_DIM))
            dout = tl.load(_block_ptrs(dOut, batch_head, seq_len, query_offsets, HEAD_DIM))
            lse = tl.load(LSE + batch_head * seq_len + query_offsets)
            delta = tl.load(Delta + batch_head * seq_len + query_offsets)

            scores = tl.dot(q, tl.trans(k)) * softmax_scale
            scores = tl.where(key_offsets[None, :] <= query_offsets[:, None], scores, float("-inf"))
            probs = tl.exp(scores - lse[:, None])

            dv += tl.dot(tl.trans(probs.to(dout.dtype)), dout)
            dprobs = tl.dot(dout, tl.trans(v))
            dscores = probs * (dprobs - delta[:, None])
            dk += tl.dot(tl.trans(dscores.to(q.dtype)), q)

    tl.store(_block_ptrs(dK, batch_kv_head, seq_len, key_offsets, HEAD_DIM), (dk * softmax_scale).to(dK.dtype.element_ty))
    tl.store(_block_ptrs(dV, batch_kv_head, seq_len, key_offsets, HEAD_DIM), dv.to(dV.dtype.element_ty))


@triton.jit
def _flash_attention_backward_dq(
    Q, K, V, dOut, dQ, LSE, Delta,
    softmax_scale: tl.constexpr,
    num_heads, num_kv_heads, seq_len,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch, head = batch_head // num_heads, batch_head % num_heads
    batch_kv_head = batch * num_kv_heads + head // (num_heads // num_kv_heads)

    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    q = tl.load(_block_ptrs(Q, batch_head, seq_len, query_offsets, HEAD_DIM))
    dout = tl.load(_block_ptrs(dOut, batch_head, seq_len, query_offsets, HEAD_DIM))
    lse = tl.load(LSE + batch_head * seq_len + query_offsets)
    delta = tl.load(Delta + batch_head * seq_len + query_offsets)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    for key_start in range(0, (query_block + 1) * BLOCK_M, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        k = tl.load(_block_ptrs(K, batch_kv_head, seq_len, key_offsets, HEAD_DIM))
        v = tl.load(_block_ptrs(V, batch_kv_head, seq_len, key_offsets, HEAD_DIM))

        scores = tl.dot(q, tl.trans(k)) * softmax_scale
        scores = tl.where(key_offsets[None, :] <= query_offsets[:, None], scores, float("-inf"))
        probs = tl.exp(scores - lse[:, None])

        dprobs = tl.dot(dout, tl.trans(v))
        dscores = probs * (dprobs - delta[:, None])
        dq += tl.dot(dscores.to(k.dtype), k)

    tl.store(_block_ptrs(dQ, batch_head, seq_len, query_offsets, HEAD_DIM), (dq * softmax_scale).to(dQ.dtype.element_ty))


def _pad_to_block(x: torch.Tensor, block: int) -> torch.Tensor:
    padding = -x.shape[2] % block
    return F.pad(x, (0, 0, 0, padding)) if padding else x.contiguous()


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, softmax_scale):
        seq_len = query.shape[2]
        query, key, value = (_pad_to_block(x, FORWARD_CONFIG["BLOCK_M"]) for x in (query, key, value))
        batch_size, num_heads, padded_len, head_dim = query.shape
        num_kv_heads = key.shape[1]

        out = torch.empty_like(query)
        lse = torch.empty(batch_size, num_heads, padded_len, device=query.device, dtype=torch.float32)
        grid = (padded_len // FORWARD_CONFIG["BLOCK_M"], batch_size * num_heads)
        _flash_attention_forward[grid](
            query, key, value, out, lse, softmax_scale, num_heads, num_kv_heads, padded_len,
            HEAD_DIM=head_dim, **FORWARD_CONFIG,
        )

        ctx.save_for_backward(query, key, value, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.seq_len = seq_len
        return out[:, :, :seq_len]

    @staticmethod
    def backward(ctx, grad_out):
        query, key, value, out, lse = ctx.saved_tensors
        batch_size, num_heads, padded_len, head_dim = query.shape
        num_kv_heads = key.shape[1]

        grad_out = _pad_to_block(grad_out, FORWARD_CONFIG["BLOCK_M"])
        grad_query, grad_key, grad_value = torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)

        delta = torch.empty_like(lse)
        grid = (padded_len // FORWARD_CONFIG["BLOCK_M"], batch_size * num_heads)
        _flash_attention_backward_preprocess[grid](
            out, grad_out, delta, padded_len, HEAD_DIM=head_dim, BLOCK_M=FORWARD_CONFIG["BLOCK_M"]
        )

        grid = (padded_len // BACKWARD_DKDV_CONFIG["BLOCK_N"], batch_size * num_kv_heads)
        _flash_attention_backward_dkdv[grid](
            query, key, value, grad_out, grad_key, grad_value, lse, delta, ctx.softmax_scale,
            num_heads, num_kv_heads, padded_len, HEAD_DIM=head_dim, **BACKWARD_DKDV_CONFIG,
        )
        grid = (padded_len // BACKWARD_DQ_CONFIG["BLOCK_M"], batch_size * num_heads)
        _flash_attention_backward_dq[grid](
            query, key, value, grad_out, grad_query, lse, delta, ctx.softmax_scale,
            num_heads, num_kv_heads, padded_len, HEAD_DIM=head_dim, **BACKWARD_DQ_CONFIG,
        )

        seq_len = ctx.seq_len
        return grad_query[:, :, :seq_len], grad_key[:, :, :seq_len], grad_value[:, :, :seq_len], None


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    softmax_scale = softmax_scale or query.shape[-1] ** -0.5
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    if not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16):
        return F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True, scale=softmax_scale)
    return FlashAttention.apply(query, key, value, softmax_scale)
