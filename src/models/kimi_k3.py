import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cache import Cache
from ..layers import Attention, KimiDeltaAttention, RMSNorm, SwiGLU


@dataclass
class KimiK3Config:
    vocab_size: int = 32768
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    head_dim: int = 64
    kda_num_heads: int = 6
    kda_head_dim: int = 128
    conv_size: int = 4
    attention_interval: int = 4
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02


class KimiK3Block(nn.Module):
    def __init__(self, config: KimiK3Config, layer_idx: int):
        super().__init__()
        self.is_attention = (layer_idx + 1) % config.attention_interval == 0
        if self.is_attention:
            self.mixer = Attention(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
            )
        else:
            self.mixer = KimiDeltaAttention(
                config.hidden_size,
                config.kda_num_heads,
                config.kda_head_dim,
                config.conv_size,
                config.rms_norm_eps,
            )
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, cache=None) -> torch.Tensor:
        hidden_states = hidden_states + self.mixer(self.input_layernorm(hidden_states), cache=cache)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class KimiK3(nn.Module):
    def __init__(self, config: KimiK3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(KimiK3Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, None if cache is None else cache[layer_idx])
        return self.norm(hidden_states)


class KimiK3ForCausalLM(nn.Module):
    def __init__(self, config: KimiK3Config):
        super().__init__()
        self.config = config
        self.model = KimiK3(config)
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
            for residual_proj in (layer.mixer.o_proj, layer.mlp.down_proj):
                nn.init.normal_(residual_proj.weight, std=std / math.sqrt(2 * self.config.num_hidden_layers))

    def estimate_flops_per_token(self, seq_len: int) -> int:
        config = self.config
        matmul_params = sum(p.numel() for p in self.model.layers.parameters()) + self.lm_head.weight.numel()
        num_attention_layers = sum(layer.is_attention for layer in self.model.layers)
        attention_flops = 12 * num_attention_layers * config.num_attention_heads * config.head_dim * seq_len
        return 6 * matmul_params + attention_flops

    def make_cache(self, batch_size: int, max_seq_len: int, dtype) -> Cache:
        device = self.lm_head.weight.device
        return Cache([layer.mixer.make_cache(batch_size, max_seq_len, device, dtype) for layer in self.model.layers])

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None, cache: Cache | None = None) -> torch.Tensor:
        hidden_states = self.model(input_ids, cache)
        logits = self.lm_head(hidden_states)
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)
