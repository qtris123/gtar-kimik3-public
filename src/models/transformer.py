import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cache import Cache, KVCache
from ..layers import Attention, RMSNorm, RotaryEmbedding, SwiGLU


@dataclass
class TransformerConfig:
    vocab_size: int = 32768
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
        )
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), position_embeddings, cache)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, input_ids: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        past_len = 0 if cache is None else cache.seq_len
        position_ids = torch.arange(past_len, past_len + input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(position_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_embeddings, None if cache is None else cache[layer_idx])
        return self.norm(hidden_states)


class TransformerForCausalLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.model = Transformer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.init_weights()

    def init_weights(self) -> None:
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=std)
        for layer in self.model.layers:
            for residual_proj in (layer.self_attn.o_proj, layer.mlp.down_proj):
                nn.init.normal_(residual_proj.weight, std=std / math.sqrt(2 * self.config.num_hidden_layers))

    def estimate_flops_per_token(self, seq_len: int) -> int:
        config = self.config
        matmul_params = sum(p.numel() for p in self.model.layers.parameters()) + self.lm_head.weight.numel()
        attention_flops = 12 * config.num_hidden_layers * config.num_attention_heads * config.head_dim * seq_len
        return 6 * matmul_params + attention_flops

    def make_cache(self, batch_size: int, max_seq_len: int, dtype) -> Cache:
        device = self.lm_head.weight.device
        return Cache([layer.self_attn.make_cache(batch_size, max_seq_len, device, dtype) for layer in self.model.layers])

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None, cache: Cache | None = None) -> torch.Tensor:
        hidden_states = self.model(input_ids, cache)
        logits = self.lm_head(hidden_states)
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)
