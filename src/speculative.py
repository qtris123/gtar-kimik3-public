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

import torch
import torch.nn as nn

from .cache import Cache, DraftTTTCache
from .models.mtp import MTPBlock


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
        draft_steps: int = 4,
    ):
        self.target_model = target_model.eval()

        # If a DraftTrainer was passed, unwrap its mtp_block and feature_proj
        if draft_block is not None and hasattr(draft_block, "mtp_block"):
            if feature_proj is None and hasattr(draft_block, "feature_proj"):
                feature_proj = draft_block.feature_proj
            draft_block = draft_block.mtp_block

        self.draft_block = (draft_block or target_model.mtp_block)
        if self.draft_block is not None:
            self.draft_block = self.draft_block.eval()

        self.feature_proj = feature_proj
        if self.feature_proj is not None:
            self.feature_proj = self.feature_proj.eval()

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
        """Run target model and return (seed_hidden, logits).

        seed_hidden is the representation used to seed the MTP draft block.
        If feature_proj is configured, features from feature_layer_indices
        are projected; otherwise, final_hidden (before the final norm) is used.
        """
        result = self.target_model(input_ids, cache=cache, return_features=True)
        logits = result["logits"]
        features = result["features"]

        if self.feature_proj is not None:
            feature_indices = self.target_model.config.feature_layer_indices
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
        """Build Depth-1 drafter prefix K/V cache for tokens up to anchor.

        Args:
            fused_hidden: (1, L, D) target fused hidden states for positions 0..L-1.
            tokens: (1, L) next token IDs 1..L.

        Returns:
            (prefix_k, prefix_v) or (None, None) if L == 0.
        """
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
        """Run the actual inference draft path for an anchor with persistent prefix state.

        Args:
            prompt_tokens: (1, prompt_len) tokens up to anchor t (prompt_len = t + 1).
            teacher_forced_tokens: (1, draft_steps) token IDs consumed at each draft depth
                (step 1 consumes x[t+1], step 2 consumes x[t+2], ..., step r consumes x[t+r]).
            draft_steps: number of draft steps to unroll.

        Returns:
            (hidden_list, logits_list) where each list has draft_steps tensors of shape
            (1, 1, D) and (1, 1, V).
        """
        prompt_tokens = prompt_tokens.to(self.device)
        teacher_forced_tokens = teacher_forced_tokens.to(self.device)
        prompt_len = prompt_tokens.shape[1]
        dummy_cache = self.target_model.make_cache(1, prompt_len + 10, self.dtype)
        all_fused, _ = self._get_hidden_and_logits(prompt_tokens, dummy_cache)
        anchor_hidden = all_fused[:, -1:, :]

        if prompt_len > 1:
            draft_prefix_k, draft_prefix_v = self.build_draft_prefix(
                all_fused[:, :-1, :], prompt_tokens[:, 1:]
            )
        else:
            draft_prefix_k, draft_prefix_v = None, None

        draft_cache = DraftTTTCache(
            prefix_k=draft_prefix_k,
            prefix_v=draft_prefix_v,
            mode="single_anchor",
        )

        hidden_list = []
        logits_list = []
        curr_hidden = anchor_hidden

        for step in range(draft_steps):
            tok = teacher_forced_tokens[:, step : step + 1]
            token_emb = self.target_model.model.embed_tokens(tok)
            draft_out = self.draft_block(curr_hidden, token_emb, ttt_cache=draft_cache)
            draft_logits = self.target_model.lm_head(
                self.draft_block.get_output_hidden(draft_out)
            )
            hidden_list.append(draft_out)
            logits_list.append(draft_logits)
            curr_hidden = draft_out

        return hidden_list, logits_list

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

        # Speculative decoding stats
        num_rounds = 0
        total_drafted = 0
        total_accepted = 0
        per_step_proposed = [0] * self.draft_steps
        per_step_accepted = [0] * self.draft_steps
        # Cumulative prefix acceptance: P(A1), P(A1∩A2), P(A1∩A2∩A3), etc.
        cumulative_prefix_accepted = [0] * self.draft_steps

        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            # --- Prefill: run the full prompt through the target ---
            all_fused, target_logits = self._get_hidden_and_logits(input_ids, target_cache)
            # prev_hidden: (1, 1, D) — representation of the last prompt token
            prev_hidden = all_fused[:, -1:, :]

            # First generated token (greedy from last prefill position)
            pending_token = target_logits[:, -1].float().argmax(-1)  # (1,)
            generated.append(pending_token)

            if eos_token_id is not None and pending_token.item() == eos_token_id:
                res = torch.stack(generated, dim=1)
                if return_stats:
                    return res, {
                        "num_rounds": 0,
                        "draft_proposed": 0,
                        "draft_accepted": 0,
                        "acceptance_rate": 0.0,
                        "mean_accepted_per_round": 0.0,
                        "per_step_acceptance": [0.0] * self.draft_steps,
                        "per_step_proposed": per_step_proposed,
                        "per_step_accepted": per_step_accepted,
                        "total_tokens_generated": res.shape[1],
                    }
                return res

            # Build persistent Depth-1 drafter prefix K/V cache for prompt tokens 0..prompt_len-2
            if prompt_len > 1:
                draft_prefix_k, draft_prefix_v = self.build_draft_prefix(
                    all_fused[:, :-1, :], input_ids[:, 1:]
                )
            else:
                draft_prefix_k, draft_prefix_v = None, None

            # Invariant maintained across iterations:
            # 1. target_cache contains all tokens up to the token immediately
            #    preceding `pending_token`.
            # 2. `pending_token` is the latest accepted token, pending entry into target_cache.
            # 3. `prev_hidden` is the representation of the token immediately
            #    preceding `pending_token` (the last token in target_cache).
            # 4. (draft_prefix_k, draft_prefix_v) contains Depth-1 drafter K/V history
            #    for all accepted tokens preceding `pending_token`.
            while len(generated) < max_new_tokens:
                steps_to_draft = min(self.draft_steps, max_new_tokens - len(generated))
                if steps_to_draft <= 0:
                    break

                # === DRAFT PHASE ===
                # EAGLE-3 TTT draft cache seeded with persistent Depth-1 prefix state
                draft_cache = DraftTTTCache(
                    prefix_k=draft_prefix_k,
                    prefix_v=draft_prefix_v,
                    mode="single_anchor",
                )

                draft_tokens: list[torch.Tensor] = []
                draft_hidden = prev_hidden  # (1, 1, D)
                curr_draft_in = pending_token  # pending token seeds the first draft step

                for _ in range(steps_to_draft):
                    tok = curr_draft_in.view(1, 1)
                    token_emb = self.target_model.model.embed_tokens(tok)  # (1, 1, D)

                    draft_out = self.draft_block(
                        draft_hidden, token_emb, ttt_cache=draft_cache
                    )  # (1, 1, D)

                    draft_logits = self.target_model.lm_head(
                        self.draft_block.get_output_hidden(draft_out)
                    )  # (1, 1, V)
                    draft_tok = draft_logits[:, -1].float().argmax(-1)  # (1,)

                    draft_tokens.append(draft_tok)
                    draft_hidden = draft_out
                    curr_draft_in = draft_tok

                    if eos_token_id is not None and draft_tok.item() == eos_token_id:
                        break

                K = len(draft_tokens)
                if K == 0:
                    # Fallback if no draft steps: process pending_token directly
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
                # Batched verification of pending_token + draft tokens
                verify_tokens = torch.cat(
                    [pending_token.view(1, 1)] + [dt.view(1, 1) for dt in draft_tokens],
                    dim=1,
                )  # (1, K+1)

                target_cache.fork()

                verify_hidden, verify_logits = self._get_hidden_and_logits(
                    verify_tokens, target_cache
                )
                target_preds = verify_logits.float().argmax(-1)  # (1, K+1)

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

                # Track speculative decoding stats
                num_rounds += 1
                total_drafted += K
                total_accepted += n_accepted
                for i in range(K):
                    per_step_proposed[i] += 1
                for i in range(n_accepted):
                    per_step_accepted[i] += 1
                
                # Track cumulative prefix acceptance
                for i in range(min(K, self.draft_steps)):
                    if n_accepted > i:  # At least i+1 tokens accepted (prefix of length i+1)
                        cumulative_prefix_accepted[i] += 1

                if eos_in_accepted:
                    # One of the accepted draft tokens is EOS
                    for i in range(n_accepted):
                        generated.append(draft_tokens[i])
                    target_cache.rollback()
                    break

                if n_accepted == K:
                    # All K drafts accepted!
                    for i in range(K):
                        generated.append(draft_tokens[i])

                    # Commit all 1 + K tokens into the target cache
                    target_cache.commit(1 + K)

                    if len(generated) >= max_new_tokens:
                        break

                    # Bonus token predicted after draft_tokens[-1]
                    bonus = target_preds[:, K]  # (1,)
                    generated.append(bonus)

                    if eos_token_id is not None and bonus.item() == eos_token_id:
                        break

                    # Target cache now contains tokens up to draft_tokens[K-1].
                    # bonus is the new pending_token.
                    # prev_hidden is the hidden state of draft_tokens[K-1].
                    prev_hidden = verify_hidden[:, -1:, :]
                    pending_token = bonus

                    # Extend persistent drafter prefix state with verified tokens
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

                    # Discard shadow state from speculative forward
                    target_cache.rollback()

                    if len(generated) >= max_new_tokens:
                        break

                    # Target model prediction at mismatch position
                    correction = target_preds[:, n_accepted]  # (1,)
                    generated.append(correction)

                    if eos_token_id is not None and correction.item() == eos_token_id:
                        break

                    # Replay pending_token plus accepted prefix to commit them
                    # to target_cache and get prev_hidden for the correction token
                    replay_list = [pending_token.view(1, 1)] + [
                        draft_tokens[i].view(1, 1) for i in range(n_accepted)
                    ]
                    replay_tokens = torch.cat(replay_list, dim=1)  # (1, 1 + n_accepted)

                    replay_hidden, _ = self._get_hidden_and_logits(
                        replay_tokens, target_cache
                    )

                    # Target cache now contains replay_tokens.
                    # correction is the new pending_token.
                    # prev_hidden is the hidden state of the last replayed token.
                    prev_hidden = replay_hidden[:, -1:, :]
                    pending_token = correction

                    # Extend persistent drafter prefix state with verified tokens
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
            stats = {
                "num_rounds": num_rounds,
                "draft_proposed": total_drafted,
                "draft_accepted": total_accepted,
                "acceptance_rate": total_accepted / max(1, total_drafted),
                "mean_accepted_per_round": total_accepted / max(1, num_rounds),
                "per_step_acceptance": [
                    per_step_accepted[i] / max(1, per_step_proposed[i])
                    for i in range(self.draft_steps)
                ],
                "per_step_proposed": per_step_proposed,
                "per_step_accepted": per_step_accepted,
                # Cumulative prefix acceptance probabilities: P(A1), P(A1∩A2), etc.
                "cumulative_prefix_acceptance": [
                    cumulative_prefix_accepted[i] / max(1, num_rounds)
                    for i in range(self.draft_steps)
                ],
                "expected_accepted_per_verification": sum(
                    cumulative_prefix_accepted[i] / max(1, num_rounds)
                    for i in range(self.draft_steps)
                ),
                "total_tokens_generated": result.shape[1],
            }
            return result, stats
        return result
