from __future__ import annotations

import torch
import torch.nn as nn

from src.cache import DraftTTTCache
from src.training.draft import DraftTrainer


def compute_overlap_and_agreement(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean overlap and top-1 agreement for testing and validation.

    Args:
        target_logits: (B, T, V) target logits.
        draft_logits: (B, T, V) draft logits.
        mask: (B, T) boolean valid positions.

    Returns:
        (overlap, top1_agreement) as scalar tensors.
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


def draft_forward_with_trajectories(
    draft_trainer: DraftTrainer,
    target_features: dict[str, torch.Tensor],
    target_logits: torch.Tensor,
    targets: torch.Tensor,
    embed_fn: callable,
    lm_head_fn: callable,
) -> dict[str, list[torch.Tensor]]:
    """Test helper: runs the vectorized DraftTrainer forward pass and captures
    intermediate draft hidden states and logits at each recursive depth.
    """
    cfg = draft_trainer.config
    B, T, V = target_logits.shape

    feature_indices = cfg.feature_layer_indices
    low_feat = target_features[f"feature_{feature_indices[0]}"]
    mid_feat = target_features[f"feature_{feature_indices[1]}"]
    high_feat = target_features[f"feature_{feature_indices[2]}"]

    fused_features = draft_trainer.feature_proj(low_feat, mid_feat, high_feat)

    draft_hiddens = []
    draft_logits_list = []

    draft_hidden = fused_features
    ttt_cache = DraftTTTCache(mode="vectorized")

    for step_r in range(1, cfg.draft_steps + 1):
        if T - step_r < 1:
            break

        hidden_input = draft_hidden[:, : T - step_r]

        token_ids = targets[:, step_r - 1 : T - 1].clone()
        valid_tokens = token_ids != -100
        token_ids[~valid_tokens] = 0
        token_emb = embed_fn(token_ids)
        token_emb = token_emb * valid_tokens.unsqueeze(-1).float()

        draft_hidden_out = draft_trainer.mtp_block(hidden_input, token_emb, ttt_cache=ttt_cache)
        draft_norm = draft_trainer.mtp_block.get_output_hidden(draft_hidden_out)
        draft_logits = lm_head_fn(draft_norm)

        draft_hiddens.append(draft_hidden_out)
        draft_logits_list.append(draft_logits)

        new_hidden = torch.zeros_like(fused_features)
        new_hidden[:, : T - step_r] = draft_hidden_out
        draft_hidden = new_hidden

    return {
        "draft_hidden": draft_hiddens,
        "draft_logits": draft_logits_list,
    }
