import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cache import Cache
from ..layers import Attention, KimiDeltaAttention, RMSNorm, SwiGLU
from .mtp import MTPBlock


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

    # --- MTP configuration (Stage 1) ---
    mtp_enabled: bool = False
    mtp_loss_weight: float = 0.3
    # Indices of layers whose hidden states are exposed via extract_features.
    # Used by Stage 2 for low/mid/high feature fusion.
    # Empty list means no intermediate features are collected.
    feature_layer_indices: list[int] = field(default_factory=list)


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
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(KimiK3Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Cache | None = None,
        extract_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) token IDs.
            cache: optional inference cache.
            extract_features: if True, return a dict of intermediate features
                alongside the final normalized hidden states.

        Returns:
            If extract_features is False: (B, T, D) normalized hidden states.
            If extract_features is True: tuple of (normalized_hidden, features_dict)
                where features_dict contains:
                  'final_hidden': hidden states BEFORE final norm (for MTP Stage 1)
                  'feature_<idx>': hidden states after layer <idx> for each index
                                   in config.feature_layer_indices
        """
        hidden_states = self.embed_tokens(input_ids)
        features = {} if extract_features else None

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, None if cache is None else cache[layer_idx])
            if extract_features and layer_idx in self.config.feature_layer_indices:
                features[f"feature_{layer_idx}"] = hidden_states

        if extract_features:
            # Final hidden states BEFORE the final norm (used by MTP in Stage 1).
            features["final_hidden"] = hidden_states
            return self.norm(hidden_states), features

        return self.norm(hidden_states)


class KimiK3ForCausalLM(nn.Module):
    def __init__(self, config: KimiK3Config):
        super().__init__()
        self.config = config
        self.model = KimiK3(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # --- Optional MTP block (Stage 1) ---
        self.mtp_block = None
        if config.mtp_enabled:
            self.mtp_block = MTPBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                rms_norm_eps=config.rms_norm_eps,
            )

        self.init_weights()

    def init_weights(self) -> None:
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=std)
        for layer in self.model.layers:
            for residual_proj in (layer.mixer.o_proj, layer.mlp.down_proj):
                nn.init.normal_(residual_proj.weight, std=std / math.sqrt(2 * self.config.num_hidden_layers))
        # MTP block residual projections get the same scaled init
        if self.mtp_block is not None:
            for residual_proj in (self.mtp_block.mixer.o_proj, self.mtp_block.mlp.down_proj):
                nn.init.normal_(residual_proj.weight, std=std / math.sqrt(2))

    def make_cache(self, batch_size: int, max_seq_len: int, dtype) -> Cache:
        device = self.lm_head.weight.device
        return Cache([layer.mixer.make_cache(batch_size, max_seq_len, device, dtype) for layer in self.model.layers])

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: Cache | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass with optional MTP joint training.

        When targets is None (inference): returns logits (B, T, V).
        When targets is provided and MTP is disabled: returns scalar main CE loss.
        When targets is provided and MTP is enabled: returns dict with
            'loss' (combined), 'main_loss', 'mtp_loss' scalars.
        When return_features is True: returns dict with 'logits', 'features'.
        """
        need_features = (
            return_features
            or (targets is not None and self.mtp_block is not None)
        )

        if need_features:
            hidden_states, features = self.model(input_ids, cache, extract_features=True)
        else:
            hidden_states = self.model(input_ids, cache)
            features = None

        logits = self.lm_head(hidden_states)

        # --- Inference path ---
        if targets is None:
            if return_features:
                return {"logits": logits, "features": features}
            return logits

        # --- Training: compute main CE loss ---
        main_loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)

        # Main top-1 agreement
        with torch.no_grad():
            valid_targets = targets != -100
            if valid_targets.any():
                main_agreement = (logits.argmax(-1)[valid_targets] == targets[valid_targets]).float().mean()
            else:
                main_agreement = torch.zeros((), device=main_loss.device)

        # --- No MTP: return scalar loss (backward compatible) ---
        if self.mtp_block is None:
            if return_features:
                return {
                    "loss": main_loss,
                    "main_loss": main_loss,
                    "main_agreement": main_agreement,
                    "features": features,
                }
            return main_loss

        # --- MTP: compute shifted auxiliary CE ---
        # Alignment (see todolist):
        #   final_hidden[:, :-1]  = hidden states for positions 0..T-2
        #   embed(targets[:, :-1]) = embeddings of tokens at positions 1..T-1
        #   mtp_labels = targets[:, 1:] = tokens at positions 2..T
        final_hidden = features["final_hidden"]

        mtp_hidden_input = final_hidden[:, :-1]     # (B, T-1, D)
        # Mask ignored labels before embedding
        mtp_target_tokens = targets[:, :-1].clone()  # (B, T-1)
        valid_mask = mtp_target_tokens != -100
        mtp_target_tokens[~valid_mask] = 0  # safe index for embedding
        mtp_embeddings = self.model.embed_tokens(mtp_target_tokens)  # (B, T-1, D)
        # Zero out embeddings for ignored positions
        mtp_embeddings = mtp_embeddings * valid_mask.unsqueeze(-1).float()

        mtp_labels = targets[:, 1:]  # (B, T-1)

        # Handle sequences too short for MTP
        if mtp_hidden_input.shape[1] == 0:
            mtp_loss = torch.zeros((), device=main_loss.device, dtype=main_loss.dtype)
            mtp_agreement = torch.zeros((), device=main_loss.device)
        else:
            mtp_hidden_out = self.mtp_block(mtp_hidden_input, mtp_embeddings)
            mtp_logits = self.lm_head(self.mtp_block.get_output_hidden(mtp_hidden_out))
            mtp_loss = F.cross_entropy(
                mtp_logits.float().flatten(0, 1), mtp_labels.flatten(), ignore_index=-100
            )
            with torch.no_grad():
                valid_mtp = mtp_labels != -100
                if valid_mtp.any():
                    mtp_agreement = (mtp_logits.argmax(-1)[valid_mtp] == mtp_labels[valid_mtp]).float().mean()
                else:
                    mtp_agreement = torch.zeros((), device=main_loss.device)

        total_loss = main_loss + self.config.mtp_loss_weight * mtp_loss

        ret = {
            "loss": total_loss,
            "main_loss": main_loss,
            "mtp_loss": mtp_loss,
            "mtp_agreement": mtp_agreement,
            "main_agreement": main_agreement,
        }
        if return_features:
            ret["features"] = features
        return ret
