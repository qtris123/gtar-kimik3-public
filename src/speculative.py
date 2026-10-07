"""
Greedy speculative decoding engine for K3 + MTP draft models.

Implements the verify-then-accept loop at inference batch size 1:
  1. Fork the target cache.
  2. Use the MTP block to draft K candidate tokens greedily.
  3. Run the target model on [pending_token, D_1, ..., D_K] in one batched forward.
  4. Compare target greedy predictions vs draft tokens left-to-right.
  5. If all K match: commit all (1 + K) tokens into the target cache,
     take the bonus token as the next pending token.
  6. On rejection (first mismatch at position n_accepted):
     - Discard shadow state (target_cache.rollback()).
     - Replay [pending_token, D_1, ..., D_{n_accepted}] through the target model
       to properly commit the accepted prefix and restore clean recurrent state.
     - Take the target model's correction prediction as the next pending token.
  7. Reset/rebuild draft state for the next round.

Correctness invariant:
  The output sequence is **token-for-token identical** to target-only greedy
  generation. This is verified in tests/test_speculative.py.

Cache and pending-token invariants:
  At the beginning of each speculation round:
  - `target_cache` contains all past tokens up to the token immediately
    preceding `pending_token`.
  - Exactly one `pending_token` is held (the next token to be verified /
    committed to the cache).
  - `prev_hidden` is the target model's final_hidden representation for the
    token immediately preceding `pending_token`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .cache import Cache, DraftTTTCache
from .models.mtp import MTPBlock


@dataclass
class SpeculativeStats:
    """Encapsulates speculation rounds, proposal counts, and acceptance statistics."""
    draft_steps: int
    num_rounds: int = 0
    total_drafted: int = 0
    total_accepted: int = 0
    per_step_proposed: list[int] = field(default_factory=list)
    per_step_accepted: list[int] = field(default_factory=list)
    cumulative_prefix_accepted: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.per_step_proposed:
            self.per_step_proposed = [0] * self.draft_steps
        if not self.per_step_accepted:
            self.per_step_accepted = [0] * self.draft_steps
        if not self.cumulative_prefix_accepted:
            self.cumulative_prefix_accepted = [0] * self.draft_steps

    def record_round(self, K: int, n_accepted: int) -> None:
        self.num_rounds += 1
        self.total_drafted += K
        self.total_accepted += n_accepted
        for i in range(K):
            self.per_step_proposed[i] += 1
        for i in range(n_accepted):
            self.per_step_accepted[i] += 1
        for i in range(min(K, self.draft_steps)):
            if n_accepted > i:
                self.cumulative_prefix_accepted[i] += 1

    def to_dict(self, total_tokens_generated: int) -> dict:
        rounds = max(1, self.num_rounds)
        drafted = max(1, self.total_drafted)
        return {
            "num_rounds": self.num_rounds,
            "draft_proposed": self.total_drafted,
            "draft_accepted": self.total_accepted,
            "acceptance_rate": self.total_accepted / drafted,
            "mean_accepted_per_round": self.total_accepted / rounds,
            "per_step_acceptance": [
                self.per_step_accepted[i] / max(1, self.per_step_proposed[i])
                for i in range(self.draft_steps)
            ],
            "per_step_proposed": self.per_step_proposed,
            "per_step_accepted": self.per_step_accepted,
            "cumulative_prefix_acceptance": [
                self.cumulative_prefix_accepted[i] / rounds
                for i in range(self.draft_steps)
            ],
            "expected_accepted_per_verification": sum(
                self.cumulative_prefix_accepted[i] / rounds
                for i in range(self.draft_steps)
            ),
            "total_tokens_generated": total_tokens_generated,
        }


class SpeculativeEngine:
    """Greedy speculative decoding engine.

    Args:
        target_model: The KimiK3ForCausalLM model (with mtp_block).
        draft_block: The MTP block used for drafting. Can be the same
            as target_model.mtp_block (Stage 1) or a separately trained
            draft block (Stage 2). Must share the same embedding/lm_head.
        feature_proj: Optional FeatureProjection layer for fusing early/mid/high
            target features (Stage 2). If None, uses final_hidden directly.
        draft_steps: Number of tokens to draft per speculation round.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_block: MTPBlock | nn.Module | None = None,
        feature_proj: nn.Module | None = None,
        feature_layer_indices: list[int] | None = None,
        draft_steps: int = 4,
    ):
        self.target_model = target_model.eval()

        # If a DraftTrainer was passed, unwrap its mtp_block, feature_proj, and feature_layer_indices
        if draft_block is not None and hasattr(draft_block, "mtp_block"):
            if feature_proj is None and hasattr(draft_block, "feature_proj"):
                feature_proj = draft_block.feature_proj
            if feature_layer_indices is None and hasattr(draft_block, "config") and getattr(draft_block.config, "feature_layer_indices", None):
                feature_layer_indices = list(draft_block.config.feature_layer_indices)
            draft_block = draft_block.mtp_block

        self.draft_block = (draft_block or target_model.mtp_block)
        if self.draft_block is not None:
            self.draft_block = self.draft_block.eval()

        self.feature_proj = feature_proj
        if self.feature_proj is not None:
            self.feature_proj = self.feature_proj.eval()

        # Determine feature layer indices for fusion and synchronize with target model
        self.feature_layer_indices = feature_layer_indices or getattr(target_model.config, "feature_layer_indices", [0, 4, 27])
        if hasattr(self.target_model, "config") and self.feature_proj is not None:
            self.target_model.config.feature_layer_indices = self.feature_layer_indices

        self.draft_steps = draft_steps

        assert self.draft_block is not None, (
            "No draft block available. Either pass draft_block or use a "
            "target model with mtp_enabled=True."
        )

        self.device = next(target_model.parameters()).device
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

    def _get_hidden_and_logits(
        self,
        input_ids: torch.Tensor,
        cache: Cache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run target model and return (seed_hidden, logits)."""
        result = self.target_model(input_ids, cache=cache, return_features=True)
        logits = result["logits"]
        features = result["features"]

        if self.feature_proj is not None:
            feature_indices = self.feature_layer_indices
            low = features[f"feature_{feature_indices[0]}"]
            mid = features[f"feature_{feature_indices[1]}"]
            high = features[f"feature_{feature_indices[2]}"]
            seed_hidden = self.feature_proj(low, mid, high)
        else:
            seed_hidden = features["final_hidden"]

        return seed_hidden, logits

    def build_draft_prefix(
        self,
        fused_hidden: torch.Tensor,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build Depth-1 drafter prefix K/V cache for tokens up to anchor."""
        if fused_hidden.shape[1] == 0:
            return None, None
        token_emb = self.target_model.model.embed_tokens(tokens)
        cache = DraftTTTCache(mode="single_anchor")
        self.draft_block(fused_hidden, token_emb, ttt_cache=cache)
        return cache.k_list[0], cache.v_list[0]

    def _extend_draft_prefix(
        self,
        prefix_k: torch.Tensor | None,
        prefix_v: torch.Tensor | None,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Extend Depth-1 drafter prefix cache with newly accepted tokens."""
        n_tokens = hidden.shape[1]
        if n_tokens == 0:
            return prefix_k, prefix_v
        curr_k, curr_v = prefix_k, prefix_v
        for j in range(n_tokens):
            step_cache = DraftTTTCache(prefix_k=curr_k, prefix_v=curr_v, mode="single_anchor")
            tok_emb = self.target_model.model.embed_tokens(tokens[:, j : j + 1])
            self.draft_block(hidden[:, j : j + 1, :], tok_emb, ttt_cache=step_cache)
            curr_k, curr_v = step_cache.k_list[0], step_cache.v_list[0]
        return curr_k, curr_v

    @torch.no_grad()
    def draft_trajectory(
        self,
        prompt_tokens: torch.Tensor,
        teacher_forced_tokens: torch.Tensor,
        draft_steps: int = 4,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Diagnostic / test oracle helper for trajectory alignment verification."""
        from tests.helpers import spec_engine_draft_trajectory
        return spec_engine_draft_trajectory(self, prompt_tokens, teacher_forced_tokens, draft_steps)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Generate tokens using greedy speculative decoding.

        Args:
            input_ids: (1, prompt_len) token IDs. Batch size must be 1.
            max_new_tokens: maximum number of new tokens to generate.
            eos_token_id: optional EOS token to stop generation.
            return_stats: if True, returns (tokens, stats_dict).

        Returns:
            (1, num_generated) tensor of generated token IDs, or (tokens, stats) if return_stats=True.
        """
        input_ids = input_ids.to(self.device)
        batch_size, prompt_len = input_ids.shape
        assert batch_size == 1, f"Speculative decoding requires batch_size=1, got {batch_size}"

        if max_new_tokens <= 0:
            empty = torch.empty((1, 0), dtype=input_ids.dtype, device=self.device)
            return (empty, {}) if return_stats else empty

        max_total_len = prompt_len + max_new_tokens + self.draft_steps + 4
        target_cache = self.target_model.make_cache(1, max_total_len, self.dtype)
        generated: list[torch.Tensor] = []
        stats = SpeculativeStats(self.draft_steps)

        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            # --- Prefill: run the full prompt through the target ---
            all_fused, target_logits = self._get_hidden_and_logits(input_ids, target_cache)
            prev_hidden = all_fused[:, -1:, :]

            # First generated token (greedy from last prefill position)
            pending_token = target_logits[:, -1].float().argmax(-1)
            generated.append(pending_token)

            if eos_token_id is not None and pending_token.item() == eos_token_id:
                res = torch.stack(generated, dim=1)
                return (res, stats.to_dict(res.shape[1])) if return_stats else res

            # Build persistent Depth-1 drafter prefix K/V cache for prompt tokens
            if prompt_len > 1:
                draft_prefix_k, draft_prefix_v = self.build_draft_prefix(
                    all_fused[:, :-1, :], input_ids[:, 1:]
                )
            else:
                draft_prefix_k, draft_prefix_v = None, None

            while len(generated) < max_new_tokens:
                steps_to_draft = min(self.draft_steps, max_new_tokens - len(generated))
                if steps_to_draft <= 0:
                    break

                # === DRAFT PHASE ===
                draft_cache = DraftTTTCache(
                    prefix_k=draft_prefix_k,
                    prefix_v=draft_prefix_v,
                    mode="single_anchor",
                )

                draft_tokens: list[torch.Tensor] = []
                draft_hidden = prev_hidden
                curr_draft_in = pending_token

                for _ in range(steps_to_draft):
                    tok = curr_draft_in.view(1, 1)
                    token_emb = self.target_model.model.embed_tokens(tok)

                    draft_out = self.draft_block(
                        draft_hidden, token_emb, ttt_cache=draft_cache
                    )

                    draft_logits = self.target_model.lm_head(
                        self.draft_block.get_output_hidden(draft_out)
                    )
                    draft_tok = draft_logits[:, -1].float().argmax(-1)

                    draft_tokens.append(draft_tok)
                    draft_hidden = draft_out
                    curr_draft_in = draft_tok

                    if eos_token_id is not None and draft_tok.item() == eos_token_id:
                        break

                K = len(draft_tokens)
                if K == 0:
                    target_hidden, target_logits = self._get_hidden_and_logits(
                        pending_token.view(1, 1), target_cache
                    )
                    draft_prefix_k, draft_prefix_v = self._extend_draft_prefix(
                        draft_prefix_k, draft_prefix_v, prev_hidden, pending_token.view(1, 1)
                    )
                    prev_hidden = target_hidden[:, -1:, :]
                    pending_token = target_logits[:, -1].float().argmax(-1)
                    generated.append(pending_token)
                    if eos_token_id is not None and pending_token.item() == eos_token_id:
                        break
                    continue

                # === VERIFY PHASE ===
                verify_tokens = torch.cat(
                    [pending_token.view(1, 1)] + [dt.view(1, 1) for dt in draft_tokens],
                    dim=1,
                )

                target_cache.fork()
                verify_hidden, verify_logits = self._get_hidden_and_logits(
                    verify_tokens, target_cache
                )
                target_preds = verify_logits.float().argmax(-1)

                # Determine length of matching prefix
                n_accepted = 0
                eos_in_accepted = False
                for i in range(K):
                    if target_preds[0, i].item() == draft_tokens[i].item():
                        n_accepted += 1
                        if eos_token_id is not None and draft_tokens[i].item() == eos_token_id:
                            eos_in_accepted = True
                            break
                    else:
                        break

                stats.record_round(K, n_accepted)

                if eos_in_accepted:
                    for i in range(n_accepted):
                        generated.append(draft_tokens[i])
                    target_cache.rollback()
                    break

                if n_accepted == K:
                    # All K drafts accepted
                    for i in range(K):
                        generated.append(draft_tokens[i])

                    target_cache.commit(1 + K)

                    if len(generated) >= max_new_tokens:
                        break

                    bonus = target_preds[:, K]
                    generated.append(bonus)

                    if eos_token_id is not None and bonus.item() == eos_token_id:
                        break

                    prev_hidden = verify_hidden[:, -1:, :]
                    pending_token = bonus

                    draft_prefix_k, draft_prefix_v = self._extend_draft_prefix(
                        draft_cache.k_list[0],
                        draft_cache.v_list[0],
                        verify_hidden[:, :K, :],
                        verify_tokens[:, 1 : 1 + K],
                    )
                else:
                    # Mismatch at position n_accepted
                    for i in range(n_accepted):
                        generated.append(draft_tokens[i])

                    target_cache.rollback()

                    if len(generated) >= max_new_tokens:
                        break

                    correction = target_preds[:, n_accepted]
                    generated.append(correction)

                    if eos_token_id is not None and correction.item() == eos_token_id:
                        break

                    replay_list = [pending_token.view(1, 1)] + [
                        draft_tokens[i].view(1, 1) for i in range(n_accepted)
                    ]
                    replay_tokens = torch.cat(replay_list, dim=1)

                    replay_hidden, _ = self._get_hidden_and_logits(
                        replay_tokens, target_cache
                    )

                    prev_hidden = replay_hidden[:, -1:, :]
                    pending_token = correction

                    if n_accepted == 0:
                        draft_prefix_k = draft_cache.k_list[0]
                        draft_prefix_v = draft_cache.v_list[0]
                    else:
                        draft_prefix_k, draft_prefix_v = self._extend_draft_prefix(
                            draft_cache.k_list[0],
                            draft_cache.v_list[0],
                            replay_hidden[:, :n_accepted, :],
                            replay_tokens[:, 1 : 1 + n_accepted],
                        )

        result = torch.stack(generated, dim=1)[:, :max_new_tokens]
        if return_stats:
            return result, stats.to_dict(result.shape[1])
        return result
