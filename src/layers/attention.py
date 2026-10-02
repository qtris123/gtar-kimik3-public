import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cache import KVCache
from ..ops import flash_attention
from .rmsnorm import RMSNorm
from .rope import apply_rotary_pos_emb


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

        if key_states.shape[2] == seq_len:
            attn_output = flash_attention(query_states, key_states, value_states)
        else:
            past_len = key_states.shape[2] - seq_len
            causal = torch.ones(seq_len, key_states.shape[2], dtype=torch.bool, device=hidden_states.device).tril(past_len)
            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=causal, enable_gqa=True)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
