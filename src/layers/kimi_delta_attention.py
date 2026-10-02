import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cache import RecurrentCache
from ..ops import chunk_kimi_delta_attention, recurrent_kimi_delta_attention
from .rmsnorm import RMSNorm
from .short_convolution import ShortConvolution


class KimiDeltaAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, conv_size: int = 4, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_size = conv_size
        inner_size = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.q_conv = ShortConvolution(inner_size, conv_size)
        self.k_conv = ShortConvolution(inner_size, conv_size)
        self.v_conv = ShortConvolution(inner_size, conv_size)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.decay_proj = nn.Sequential(nn.Linear(hidden_size, head_dim, bias=False), nn.Linear(head_dim, inner_size, bias=True))
        self.output_gate_proj = nn.Sequential(nn.Linear(hidden_size, head_dim, bias=False), nn.Linear(head_dim, inner_size, bias=False))
        self.norm = RMSNorm(head_dim, eps)
        self.o_proj = nn.Linear(inner_size, hidden_size, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads).uniform_(1, 16)))
        dt = torch.exp(torch.empty(inner_size).uniform_(math.log(1e-3), math.log(1e-1)))
        self.decay_proj[1].bias.data = dt + torch.log(-torch.expm1(-dt))

    def make_cache(self, batch_size: int, max_seq_len: int, device, dtype) -> RecurrentCache:
        inner_size = self.num_heads * self.head_dim
        return RecurrentCache(batch_size, self.num_heads, self.head_dim, self.head_dim, inner_size, self.conv_size, device, dtype)

    def forward(self, hidden_states: torch.Tensor, cache: RecurrentCache | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        head_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        conv_states = (None, None, None) if cache is None else cache.conv_states

        query = F.normalize(self.q_conv(self.q_proj(hidden_states), conv_states[0]).view(head_shape), dim=-1)
        key = F.normalize(self.k_conv(self.k_proj(hidden_states), conv_states[1]).view(head_shape), dim=-1)
        value = self.v_conv(self.v_proj(hidden_states), conv_states[2]).view(head_shape)
        beta = self.beta_proj(hidden_states).float().sigmoid()
        gate = -self.A_log.exp()[:, None] * F.softplus(self.decay_proj(hidden_states).float().view(head_shape))

        attention = chunk_kimi_delta_attention if seq_len > 1 else recurrent_kimi_delta_attention
        out, final_state = attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), gate.transpose(1, 2), beta.transpose(1, 2),
            initial_state=None if cache is None else cache.state, output_final_state=True,
        )
        if cache is not None:
            cache.update(final_state, seq_len)
        out = self.norm(out.transpose(1, 2)) * F.silu(self.output_gate_proj(hidden_states).view(head_shape))
        return self.o_proj(out.reshape(batch_size, seq_len, -1))
