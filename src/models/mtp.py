"""
One reusable Multi-Token Prediction (MTP) block.

Architecture (following DeepSeek-v3-Lite / R1):
  1. Separately RMSNorm the prior hidden state and next-token embedding.
  2. Concatenate them along the feature dimension and project 2d -> d.
  3. Run a shallow causal self-attention block (one layer).
  4. Run a SwiGLU feed-forward block.

The block does NOT own the embedding table or vocabulary head -- those are
shared with the target model.  It DOES own its own output RMSNorm so that
the saved draft checkpoint includes all normalization state it needs.

Hidden-state contract for recursive use:
  - Input:  (batch, seq, hidden_size)  prior hidden states
  - Input:  (batch, seq, hidden_size)  token embeddings for the *next* token
  - Output: (batch, seq, hidden_size)  refined hidden states (ready to be
            fed back as the next step's "prior hidden state")

Reference: atandra2000/DeepSeek-v3-Lite models/mtp.py (MTPBlock, MTPModule)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cache import KVCache, DraftTTTCache
from ..layers import Attention, RMSNorm, SwiGLU


class MTPBlock(nn.Module):
    """Single MTP / draft block that can be called recursively.

    Parameters
    ----------
    hidden_size : int
        Model dimension *d*.
    intermediate_size : int
        SwiGLU intermediate dimension.
    num_attention_heads : int
        Number of attention heads in the shallow causal layer.
    num_key_value_heads : int
        Number of key/value heads (GQA).
    head_dim : int
        Per-head dimension.
    rms_norm_eps : float
        Epsilon for all RMSNorm layers.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        # --- Fusion: normalize separately, concat, project 2d -> d ---
        self.norm_hidden = RMSNorm(hidden_size, rms_norm_eps)
        self.norm_embed = RMSNorm(hidden_size, rms_norm_eps)
        self.fuse_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # --- Shallow causal self-attention ---
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.mixer = Attention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
        )

        # --- SwiGLU FFN ---
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

        # --- Draft output normalization (owned by the block, saved with it) ---
        self.output_norm = RMSNorm(hidden_size, rms_norm_eps)

    def make_cache(self, batch_size: int, max_seq_len: int, device, dtype) -> KVCache:
        """Create a KV cache for the draft block's internal Attention layer.

        Used during speculative decoding to maintain attention history
        across recursive draft steps.
        """
        return self.mixer.make_cache(batch_size, max_seq_len, device, dtype)

    def fuse(self, prev_hidden: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        """Fuse prior hidden state and next-token embedding."""
        return self.fuse_proj(
            torch.cat([self.norm_hidden(prev_hidden), self.norm_embed(token_embedding)], dim=-1)
        )

    def forward(
        self,
        prev_hidden: torch.Tensor,
        token_embedding: torch.Tensor,
        cache: KVCache | None = None,
        ttt_cache: DraftTTTCache | None = None,
    ) -> torch.Tensor:
        """
        Args:
            prev_hidden: (B, T, D) hidden states from the prior step.
            token_embedding: (B, T, D) embeddings of the next-position tokens.
            cache: optional KV cache for standard incremental attention.
            ttt_cache: optional DraftTTTCache for EAGLE-3 TTT recursive attention.

        Returns:
            hidden_states: (B, T, D) refined hidden states.
        """
        # Fuse: normalize each input independently, concatenate, project
        fused = self.fuse(prev_hidden, token_embedding)

        # Residual causal self-attention (optionally cached or TTT-managed)
        hidden_states = fused + self.mixer(self.input_layernorm(fused), cache=cache, ttt_cache=ttt_cache)

        # Residual SwiGLU FFN
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states

    def get_output_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply output normalization to get hidden states ready for the LM head."""
        return self.output_norm(hidden_states)
