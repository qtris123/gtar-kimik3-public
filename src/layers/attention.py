import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cache import KVCache, DraftTTTCache
from ..ops import flash_attention
from .rmsnorm import RMSNorm
from .rope import apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, num_kv_heads, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(b, num_kv_heads, n_rep, slen, head_dim)
        .reshape(b, num_kv_heads * n_rep, slen, head_dim)
    )


def build_ttt_diagonal_mask(
    query_len: int,
    k_lens: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Build attention mask for EAGLE-3 TTT diagonal-extension attention.

    At depth r (query length T_r):
    - Block 1 (step 1, length T_1): causal prefix (position t attends to t' <= t).
    - Block s (step s >= 2, length T_s): diagonal extension (position t attends only to t' == t).
    """
    q_idx = torch.arange(query_len, device=device).unsqueeze(1)  # (query_len, 1)

    # Block 1: causal prefix
    k_idx_1 = torch.arange(k_lens[0], device=device).unsqueeze(0)  # (1, T_1)
    masks = [k_idx_1 <= q_idx]

    # Blocks 2..r: diagonal extension (only same anchor position t' == t)
    for s in range(1, len(k_lens)):
        k_idx_s = torch.arange(k_lens[s], device=device).unsqueeze(0)  # (1, T_s)
        masks.append(k_idx_s == q_idx)

    # Shape: (1, 1, query_len, sum(k_lens))
    mask = torch.cat(masks, dim=1)
    return mask.unsqueeze(0).unsqueeze(0)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

    def make_cache(self, batch_size: int, max_seq_len: int, device, dtype) -> KVCache:
        return KVCache(batch_size, self.num_key_value_heads, max_seq_len, self.head_dim, device, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: KVCache | None = None,
        ttt_cache: DraftTTTCache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_embeddings)
        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states)

        if ttt_cache is not None:
            ttt_cache.append(key_states, value_states)
            if ttt_cache.step_count == 1:
                # Step 1: initial causal prefix step
                attn_output = flash_attention(query_states, key_states, value_states)
            else:
                # Step r >= 2: diagonal-extension TTT attention
                k_all = torch.cat(ttt_cache.k_list, dim=2)
                v_all = torch.cat(ttt_cache.v_list, dim=2)
                n_rep = self.num_attention_heads // self.num_key_value_heads
                k_all = repeat_kv(k_all, n_rep)
                v_all = repeat_kv(v_all, n_rep)

                if ttt_cache.mode == "single_anchor":
                    attn_output = F.scaled_dot_product_attention(
                        query_states, k_all, v_all, attn_mask=None
                    )
                else:
                    k_lens = [k.shape[2] for k in ttt_cache.k_list]
                    attn_mask = build_ttt_diagonal_mask(seq_len, k_lens, hidden_states.device)
                    attn_output = F.scaled_dot_product_attention(
                        query_states, k_all, v_all, attn_mask=attn_mask
                    )
        elif key_states.shape[2] == seq_len:
            attn_output = flash_attention(query_states, key_states, value_states)
        else:
            past_len = key_states.shape[2] - seq_len
            causal = torch.ones(seq_len, key_states.shape[2], dtype=torch.bool, device=hidden_states.device).tril(past_len)
            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=causal, enable_gqa=True)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
