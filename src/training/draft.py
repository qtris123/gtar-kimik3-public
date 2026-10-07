from __future__ import annotations

"""
Stage 2: Draft fine-tuning with recursive unroll and LK acceptance-overlap loss.

Trains a pretrained MTP block (from Stage 1) plus a feature projection layer
against a frozen target model. The draft block is called recursively: start
from fused target features, then feed each draft hidden output back into the
same block at the next step.

Key design choices:
- Teacher-forced: use shifted ground-truth token embeddings (not sampled tokens).
- LK (acceptance-overlap) loss at temperature 1 (NVIDIA "alpha" objective).
- Causal draft attention history maintained across recursive steps via DraftTTTCache.
- Feature projection: Linear(3d, d), initialized as [0, 0, I] so initial
  projected feature equals the high feature used in Stage 1.
- Step loss decay: 0.8 depth decay weighting.

Reference: NVIDIA Automodel eagle/core.py (Eagle3TrainerModule, _lk_step_loss)
Reference: NVIDIA Automodel eagle/draft_kimi_k3.py (Eagle3KimiK3Model.__init__)
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..cache import DraftTTTCache
from ..models.mtp import MTPBlock


@dataclass
class DraftConfig:
    """Configuration for Stage 2 draft training."""
    draft_steps: int = 4  # Number of recursive unroll steps
    # Feature layer indices for low/mid/high fusion (must match Stage 1)
    feature_layer_indices: list[int] = field(default_factory=list)
    # Step loss weighting
    step_loss_decay: float = 0.8  # Per-step decay factor (NVIDIA EAGLE-3 default)
    step_loss_weighting: str = "decay"  # 'decay' or 'uniform'


class FeatureProjection(nn.Module):
    """Bias-free Linear(3d, d) for low/mid/high target feature fusion.

    Initialized with zero blocks for low and mid features, and an identity
    block for the high feature, so that the initial projection output equals
    exactly the high feature used in Stage 1.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 3, hidden_size, bias=False)

    def init_identity_high(self) -> None:
        """Initialize so proj([low, mid, high]) = high exactly.

        Weight shape is (d, 3d). Split into three (d, d) blocks:
          [W_low | W_mid | W_high]
        Set W_low = 0, W_mid = 0, W_high = I.
        """
        d = self.proj.weight.shape[0]
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:, 2 * d:].copy_(torch.eye(d, device=self.proj.weight.device, dtype=self.proj.weight.dtype))

    def forward(self, low: torch.Tensor, mid: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """
        Args:
            low, mid, high: (B, T, D) target features from early/middle/final layers.

        Returns:
            (B, T, D) fused feature.
        """
        return self.proj(torch.cat([low, mid, high], dim=-1))


def lk_loss(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pure LK (acceptance-overlap) loss at temperature 1.

    LK loss = -log(overlap) where overlap = sum_v min(p_v, q_v).
    This is the "alpha" objective from NVIDIA's Eagle3 training.

    Args:
        target_logits: (B, T, V) logits from the frozen target.
        draft_logits: (B, T, V) logits from the draft model.
        mask: (B, T) boolean mask of valid positions.
        eps: clamp minimum for numerical stability.

    Returns:
        Scalar loss: masked mean of -log(overlap).
    """
    p = torch.softmax(target_logits.float(), dim=-1).detach()  # target probs, no grad
    q = torch.softmax(draft_logits.float(), dim=-1)
    overlap = torch.minimum(p, q).sum(dim=-1)  # (B, T)

    # NVIDIA-style zero-overlap handling: preserve gradient for positive overlap, avoid inf/NaN for exact zero
    safe_log = torch.log(overlap.clamp_min(torch.finfo(overlap.dtype).tiny))
    log_overlap = torch.where(overlap > 0, safe_log, torch.zeros_like(overlap))
    per_position_loss = -log_overlap  # (B, T)

    # Masked mean
    mask_float = mask.float()
    denom = mask_float.sum().clamp_min(1.0)
    return (per_position_loss * mask_float).sum() / denom


class DraftTrainer(nn.Module):
    """Manages the draft model for Stage 2 training.

    Contains:
    - A pretrained MTP block (initialized from Stage 1 checkpoint).
    - A FeatureProjection layer for low/mid/high fusion.

    The target model is external and frozen; it is passed to the forward method.
    """

    def __init__(self, mtp_block: MTPBlock, config: DraftConfig):
        super().__init__()
        self.config = config
        self.mtp_block = mtp_block
        self.feature_proj = FeatureProjection(mtp_block.hidden_size)
        self.feature_proj.init_identity_high()

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the parameters that should be optimized in Stage 2."""
        return list(self.mtp_block.parameters()) + list(self.feature_proj.parameters())

    def forward(
        self,
        target_features: dict[str, torch.Tensor],
        target_logits: torch.Tensor,
        targets: torch.Tensor,
        embed_fn: callable,
        lm_head_fn: callable,
        output_norm_fn: callable | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run recursive draft unroll and compute LK loss.

        Args:
            target_features: dict from target model's extract_features, containing
                'final_hidden' and 'feature_<idx>' entries.
            target_logits: (B, T, V) logits from the frozen target.
            targets: (B, T) ground truth token IDs.
            embed_fn: embedding function (token_ids -> embeddings).
            lm_head_fn: vocabulary head function (hidden -> logits).
            output_norm_fn: optional normalization before lm_head (if not using
                            mtp_block.get_output_hidden).

        Returns:
            Dict with 'loss' and 'per_step_loss'.
        """
        cfg = self.config
        B, T, V = target_logits.shape
        device = target_logits.device

        # Gather low/mid/high features
        feature_indices = cfg.feature_layer_indices
        assert len(feature_indices) == 3, f"Need exactly 3 feature indices (low/mid/high), got {len(feature_indices)}"

        low_feat = target_features[f"feature_{feature_indices[0]}"]
        mid_feat = target_features[f"feature_{feature_indices[1]}"]
        high_feat = target_features[f"feature_{feature_indices[2]}"]

        # Fuse features: shape (B, T, D)
        fused_features = self.feature_proj(low_feat, mid_feat, high_feat)

        # Recursive unroll with teacher-forced token embeddings
        per_step_losses = []
        draft_hidden = fused_features
        ttt_cache = DraftTTTCache(mode="vectorized")

        for step_r in range(1, cfg.draft_steps + 1):
            if T - step_r < 1:
                break

            # Slice hidden states and teacher-forced token embeddings
            hidden_input = draft_hidden[:, : T - step_r]  # (B, T-step_r, D)

            token_ids = targets[:, step_r - 1 : T - 1].clone()  # (B, T-step_r)
            valid_tokens = token_ids != -100
            token_ids[~valid_tokens] = 0
            token_emb = embed_fn(token_ids)  # (B, T-step_r, D)
            token_emb = token_emb * valid_tokens.unsqueeze(-1).float()

            # Run MTP block with EAGLE-3 TTT attention
            draft_hidden_out = self.mtp_block(hidden_input, token_emb, ttt_cache=ttt_cache)

            # Get draft logits
            draft_norm = self.mtp_block.get_output_hidden(draft_hidden_out)
            draft_logits = lm_head_fn(draft_norm)  # (B, T-step_r, V)

            # Supervision: target logits shifted by step_r
            sup_target_logits = target_logits[:, step_r:T]  # (B, T-step_r, V)
            sup_targets = targets[:, step_r:T]

            step_mask = (sup_targets != -100) & valid_tokens  # (B, T-step_r)
            step_loss = lk_loss(sup_target_logits, draft_logits, step_mask)
            per_step_losses.append(step_loss)

            # Feed back draft hidden states for the next recursive step
            new_hidden = torch.zeros_like(fused_features)
            new_hidden[:, : T - step_r] = draft_hidden_out
            draft_hidden = new_hidden

        # Combine step losses
        if not per_step_losses:
            total_loss = torch.zeros((), device=device, requires_grad=True)
        else:
            num_steps = len(per_step_losses)
            if cfg.step_loss_weighting == "decay":
                weights = torch.tensor(
                    [cfg.step_loss_decay ** i for i in range(num_steps)],
                    device=device,
                )
            else:
                weights = torch.ones(num_steps, device=device)

            weights = weights / weights.sum()
            total_loss = sum(w * l for w, l in zip(weights, per_step_losses))

        return {
            "loss": total_loss,
            "per_step_loss": per_step_losses,
        }
