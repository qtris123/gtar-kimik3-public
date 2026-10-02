import torch
import torch.nn.functional as F
import triton
import triton.language as tl

BLOCK_D = 64
INVERSE_BLOCK = tl.constexpr(16)


@triton.jit
def tile_ptrs(base, row_base, rows, cols, row_stride):
    return base + (row_base + rows[:, None]) * row_stride + cols[None, :]


@triton.jit
def _chunk_cumsum(X, Out, seq_len, DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_D: tl.constexpr):
    chunk = tl.program_id(0)
    dim_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    rows = chunk * CHUNK + tl.arange(0, CHUNK)
    cols = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    x = tl.load(tile_ptrs(X, batch_head * seq_len, rows, cols, DIM)).to(tl.float32)
    tl.store(tile_ptrs(Out, batch_head * seq_len, rows, cols, DIM), tl.cumsum(x, axis=0))


@triton.jit
def triangular_inverse(lower, CHUNK: tl.constexpr):
    # (I + lower)^-1 for a strictly lower triangular tile: forward substitution inside INVERSE_BLOCK x INVERSE_BLOCK
    # diagonal blocks (all blocks at once), then merged pairwise with [[A, 0], [C, B]]^-1 = [[A^-1, 0], [-B^-1 C A^-1, B^-1]]
    offsets = tl.arange(0, CHUNK)
    rows, cols = offsets[:, None], offsets[None, :]
    same_block = rows // INVERSE_BLOCK == cols // INVERSE_BLOCK
    block_lower = tl.where(same_block, lower, 0.0)
    inverse = tl.where(rows == cols, 1.0, 0.0)
    for r in range(1, INVERSE_BLOCK):
        lower_row = tl.sum(tl.where(rows % INVERSE_BLOCK == r, block_lower, 0.0), axis=0)
        correction = tl.sum(lower_row[:, None] * inverse, axis=0)
        inverse = tl.where((rows % INVERSE_BLOCK == r) & same_block, inverse - correction[None, :], inverse)
    for level in tl.static_range(6):
        if INVERSE_BLOCK * 2**level < CHUNK:
            merged = rows // (INVERSE_BLOCK * 2 ** (level + 1)) == cols // (INVERSE_BLOCK * 2 ** (level + 1))
            lower_left = (rows // (INVERSE_BLOCK * 2**level) == cols // (INVERSE_BLOCK * 2**level) + 1) & merged
            product = tl.dot(tl.dot(inverse, lower), inverse)
            inverse = inverse - tl.where(lower_left, product, 0.0)
    return inverse


@triton.jit
def gated_scores_diagonal(left, key, left_cumsum, key_cumsum):
    # out[r, s] = sum_k left[r, k] key[s, k] exp(b_left[r, k] - b_key[s, k]) in fp32 with every exponent clamped at zero;
    # entries above the diagonal are wrong and must be masked by the caller
    decay = tl.exp(tl.minimum(left_cumsum[:, None, :] - key_cumsum[None, :, :], 0.0))
    return tl.sum(left[:, None, :] * key[None, :, :] * decay, axis=2)


@triton.jit
def gated_scores_backward_left(dscores, key, left_cumsum, key_cumsum):
    # d left[r, :] = sum_s dscores[r, s] key[s, :] exp(b_left[r] - b_key[s]) for the causally masked dscores
    decay = tl.exp(tl.minimum(left_cumsum[:, None, :] - key_cumsum[None, :, :], 0.0))
    return tl.sum(dscores[:, :, None] * key[None, :, :] * decay, axis=1)


@triton.jit
def gated_scores_backward_key(dscores, left, left_cumsum, key_cumsum):
    # d key[s, :] = sum_r dscores[r, s] left[r, :] exp(b_left[r] - b_key[s]) for the causally masked dscores
    decay = tl.exp(tl.minimum(left_cumsum[:, None, :] - key_cumsum[None, :, :], 0.0))
    return tl.sum(dscores[:, :, None] * left[:, None, :] * decay, axis=0)


def pad_sequence(x: torch.Tensor, block: int) -> torch.Tensor:
    padding = -x.shape[2] % block
    return F.pad(x, [0, 0] * (x.dim() - 3) + [0, padding]) if padding else x.contiguous()


def chunk_cumsum(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    batch_size, num_heads, seq_len, dim = x.shape
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    grid = (seq_len // chunk_size, dim // BLOCK_D, batch_size * num_heads)
    _chunk_cumsum[grid](x, out, seq_len, DIM=dim, CHUNK=chunk_size, BLOCK_D=BLOCK_D)
    return out



def autocast_inputs(*tensors):
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        return [x.to(dtype) for x in tensors]
    return list(tensors)
