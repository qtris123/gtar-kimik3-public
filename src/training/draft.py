from __future__ import annotations

"""
Stage 2: Draft fine-tuning with recursive unroll and LK acceptance-overlap loss.

Trains a pretrained MTP block (from Stage 1) plus a feature projection layer
against a frozen target model.  The draft block is called recursively: start
from fused target features, then feed each draft hidden output back into the
same block at the next step.

Key design choices (from the implementation brief):
- Teacher-forced: use shifted ground-truth token embeddings (not sampled tokens).
- LK (acceptance-overlap) loss at temperature 1 (NVIDIA "alpha" objective).
- Causal draft attention history maintained across recursive steps.
- Feature projection: Linear(3d, d), initialized as [0, 0, I] so initial
  projected feature equals the high feature used in Stage 1.

Reference: NVIDIA Automodel eagle/core.py (Eagle3TrainerModule, _lk_step_loss)
Reference: NVIDIA Automodel eagle/draft_kimi_k3.py (Eagle3KimiK3Model.__init__)
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cache import DraftTTTCache
from ..models.mtp import MTPBlock
from ..layers import RMSNorm


@dataclass
class DraftConfig:
    """Configuration for Stage 2 draft training."""
    draft_steps: int = 4  # Number of recursive unroll steps (supports up to 7)
    # Feature layer indices for low/mid/high fusion.
    # Must match the feature_layer_indices used during Stage 1.
    feature_layer_indices: list[int] = field(default_factory=list)
    # Loss settings
    lk_temperature: float = 1.0
    lk_eps: float = 1e-8
    # Step loss weighting: 'uniform' or 'decay'
    step_loss_weighting: str = "uniform"
    step_loss_decay: float = 0.8  # Per-step decay factor when weighting='decay' (NVIDIA EAGLE-3 default)


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

        Weight shape is (d, 3d).  Split into three (d, d) blocks:
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
    # Full vocabulary, float32 for probability/loss calculations
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


def compute_overlap_and_agreement(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean overlap and top-1 agreement for logging.

    Args:
        target_logits: (B, T, V) target logits.
        draft_logits: (B, T, V) draft logits.
        mask: (B, T) boolean valid positions.

    Returns:
        (overlap, top1_agreement) as tensors.
    """
    with torch.no_grad():
        p = torch.softmax(target_logits.float(), dim=-1)
        q = torch.softmax(draft_logits.float(), dim=-1)
        overlap = torch.minimum(p, q).sum(dim=-1)  # (B, T)
        top1_agree = (target_logits.argmax(-1) == draft_logits.argmax(-1)).float()

        mask_float = mask.float()
        denom = mask_float.sum().clamp_min(1.0)
        mean_overlap = (overlap * mask_float).sum() / denom
        mean_agree = (top1_agree * mask_float).sum() / denom
    return mean_overlap, mean_agree


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
        # Initialize projection so output = high feature (identity on high block)
        self.feature_proj.init_identity_high()

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the parameters that should be optimized in Stage 2."""
        params = list(self.mtp_block.parameters()) + list(self.feature_proj.parameters())
        return params

    def forward(
        self,
        target_features: dict[str, torch.Tensor],
        target_logits: torch.Tensor,
        targets: torch.Tensor,
        embed_fn: callable,
        lm_head_fn: callable,
        output_norm_fn: callable | None = None,
        return_trajectories: bool = False,
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
            Dict with 'loss', 'per_step_loss', 'per_step_overlap', 'per_step_agreement'.
        """
        cfg = self.config
        B, T, V = target_logits.shape
        device = target_logits.device

        # --- Gather low/mid/high features ---
        feature_indices = cfg.feature_layer_indices
        assert len(feature_indices) == 3, f"Need exactly 3 feature indices (low/mid/high), got {len(feature_indices)}"

        low_feat = target_features[f"feature_{feature_indices[0]}"]
        mid_feat = target_features[f"feature_{feature_indices[1]}"]
        high_feat = target_features[f"feature_{feature_indices[2]}"]

        # --- Fuse features ---
        fused_features = self.feature_proj(low_feat, mid_feat, high_feat)  # (B, T, D)

        # --- Recursive unroll with teacher-forced token embeddings ---
        # For an anchor at position t, step r (1-indexed):
        #   - consumes token embedding x[t+r]
        #   - supervises the distribution for x[t+r+1]
        #
        # We implement this by shifting targets for each step.
        # Step 0 hidden = fused_features
        # Step r: draft_hidden = mtp_block(prev_hidden[:, :-1], embed(targets[:, r-1:-1]))
        #         draft supervised by target_logits shifted by r positions

        per_step_losses = []
        per_step_overlaps = []
        per_step_agreements = []
        draft_hiddens = []
        draft_logits_list = []

        # Current hidden state for the draft (starts from fused target features)
        draft_hidden = fused_features
        # EAGLE-3 TTT attention cache preserving per-anchor K/V history across depths
        ttt_cache = DraftTTTCache(mode="vectorized")

        for step_r in range(1, cfg.draft_steps + 1):
            # Valid region: we need targets[t + step_r] as input and
            # targets[t + step_r + 1] (or equivalently target_logits[t + step_r])
            # as supervision.
            # But since target_logits[t] predicts targets[t] (= next token of input[t]),
            # the supervision logits for step r at anchor t are target_logits[t + step_r - 1]
            # Wait -- let me be precise about the alignment:
            #
            # targets[:, t] is the token following inputs[:, t].
            # target_logits[:, t] produces distribution over targets[:, t].
            #
            # For draft step r=1 at anchor t:
            #   - Input embedding: embed(targets[:, t])  (= token at position t+1)
            #   - Supervision: target distribution for token at position t+2 = target_logits[:, t+1]
            #   - Draft hidden input: fused_features[:, t] (or prev_hidden[:, t])
            #
            # So for step r:
            #   - Input token: targets[:, (r-1) : T-(1)]  shifted r-1 from start
            #   - Supervision logits: target_logits[:, r : T]
            #   - Supervision targets: targets[:, r : T]  (for mask building)
            #   - Draft hidden: prev_hidden[:, : T-r]

            if T - step_r < 1:
                # Sequence too short for this step
                break

            # Slice hidden states and embeddings
            hidden_input = draft_hidden[:, :T - step_r]  # (B, T-step_r, D)

            # Teacher-forced token embeddings
            token_ids = targets[:, step_r - 1: T - 1].clone()  # (B, T-step_r)
            valid_tokens = token_ids != -100
            token_ids[~valid_tokens] = 0
            token_emb = embed_fn(token_ids)  # (B, T-step_r, D)
            token_emb = token_emb * valid_tokens.unsqueeze(-1).float()

            # Run MTP block with EAGLE-3 TTT attention
            draft_hidden_out = self.mtp_block(hidden_input, token_emb, ttt_cache=ttt_cache)  # (B, T-step_r, D)

            # Get draft logits
            draft_norm = self.mtp_block.get_output_hidden(draft_hidden_out)
            draft_logits = lm_head_fn(draft_norm)  # (B, T-step_r, V)

            if return_trajectories:
                draft_hiddens.append(draft_hidden_out)
                draft_logits_list.append(draft_logits)

            # Supervision: target logits shifted by step_r
            sup_target_logits = target_logits[:, step_r:T]  # (B, T-step_r, V)
            sup_targets = targets[:, step_r:T]  # for building mask

            # Valid mask: both the input token and supervision must be valid
            step_mask = (sup_targets != -100) & valid_tokens  # (B, T-step_r)

            # Compute LK loss for this step
            step_loss = lk_loss(sup_target_logits, draft_logits, step_mask, eps=cfg.lk_eps)
            overlap, agreement = compute_overlap_and_agreement(
                sup_target_logits, draft_logits, step_mask
            )

            per_step_losses.append(step_loss)
            per_step_overlaps.append(overlap)
            per_step_agreements.append(agreement)

            # Feed back draft hidden states for the next recursive step.
            # Pad back to length T for uniform slicing in the next step.
            # We create a new tensor to keep gradient flow clean.
            new_hidden = torch.zeros_like(fused_features)
            new_hidden[:, :T - step_r] = draft_hidden_out
            draft_hidden = new_hidden

        # --- Combine step losses ---
        if not per_step_losses:
            total_loss = torch.zeros((), device=device, requires_grad=True)
        else:
            num_steps = len(per_step_losses)
            if cfg.step_loss_weighting == "decay":
                weights = torch.tensor(
                    [cfg.step_loss_decay ** i for i in range(num_steps)],
                    device=device,
                )
            else:  # uniform
                weights = torch.ones(num_steps, device=device)

            # Normalize weights
            weights = weights / weights.sum()
            total_loss = sum(w * l for w, l in zip(weights, per_step_losses))

        res = {
            "loss": total_loss,
            "per_step_loss": per_step_losses,  # Keep as tensors
            "per_step_overlap": per_step_overlaps,
            "per_step_agreement": per_step_agreements,
        }
        if return_trajectories:
            res["draft_hidden"] = draft_hiddens
            res["draft_logits"] = draft_logits_list
        return res


def single_anchor_sequential_reference(
    draft_trainer: DraftTrainer,
    target_features: dict[str, torch.Tensor],
    targets: torch.Tensor,
    embed_fn: callable,
    anchor_idx: int,
    draft_steps: int = 4,
    lm_head_fn: callable | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    """Slow unvectorized single-anchor sequential reference for EAGLE-3 TTT attention.

    Evaluates a single anchor position `anchor_idx` step-by-step:
    - Step 1: evaluates prefix 0..anchor_idx through mtp_block, recording
      the step-1 causal prefix K/V history.
    - Step r (r >= 2): evaluates ONLY anchor_idx from step r-1 and token
      embedding x[anchor_idx + r].
      In attention, query at anchor_idx attends to:
        K^{(1)}_{0:anchor_idx+1} (step-1 causal prefix)
        + K^{(2)}_{anchor_idx}
        + ...
        + K^{(r)}_{anchor_idx} (same anchor's draft steps)

    Returns:
        (hidden_states_per_step, logits_per_step) where each list has
        elements corresponding to depths 1..draft_steps at anchor_idx,
        each of shape (B, 1, D) and (B, 1, V).
    """
    cfg = draft_trainer.config
    feature_indices = cfg.feature_layer_indices
    low_feat = target_features[f"feature_{feature_indices[0]}"]
    mid_feat = target_features[f"feature_{feature_indices[1]}"]
    high_feat = target_features[f"feature_{feature_indices[2]}"]

    fused_features = draft_trainer.feature_proj(low_feat, mid_feat, high_feat)
    B, T, D = fused_features.shape
    t = anchor_idx

    assert 0 <= t < T - 1, f"Anchor {t} out of range for sequence length {T}"

    hidden_outputs = []
    logits_outputs = [] if lm_head_fn is not None else None

    # Step 1: run full causal prefix 0..t
    prefix_hidden = fused_features[:, : t + 1]  # (B, t+1, D)
    token_ids_1 = targets[:, : t + 1].clone()
    valid_1 = token_ids_1 != -100
    token_ids_1[~valid_1] = 0
    token_emb_1 = embed_fn(token_ids_1) * valid_1.unsqueeze(-1).float()

    ref_cache = DraftTTTCache(mode="single_anchor")
    step1_out = draft_trainer.mtp_block(prefix_hidden, token_emb_1, ttt_cache=ref_cache)
    h_t_1 = step1_out[:, t : t + 1, :]
    hidden_outputs.append(h_t_1)

    if lm_head_fn is not None:
        norm_1 = draft_trainer.mtp_block.get_output_hidden(h_t_1)
        logits_outputs.append(lm_head_fn(norm_1))

    # Step 2..draft_steps: single anchor continuation
    curr_hidden = h_t_1
    for step_r in range(2, draft_steps + 1):
        if t + step_r > T - 1:
            break
        token_id_r = targets[:, t + step_r - 1 : t + step_r].clone()
        valid_r = token_id_r != -100
        token_id_r[~valid_r] = 0
        token_emb_r = embed_fn(token_id_r) * valid_r.unsqueeze(-1).float()

        h_t_r = draft_trainer.mtp_block(curr_hidden, token_emb_r, ttt_cache=ref_cache)
        hidden_outputs.append(h_t_r)

        if lm_head_fn is not None:
            norm_r = draft_trainer.mtp_block.get_output_hidden(h_t_r)
            logits_outputs.append(lm_head_fn(norm_r))

        curr_hidden = h_t_r

    return hidden_outputs, logits_outputs
