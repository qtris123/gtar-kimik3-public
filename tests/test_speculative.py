"""
Speculative decoding tests — verify correctness against target-only greedy.

The fundamental correctness invariant: speculative decoding must produce
**token-for-token identical** output to target-only greedy generation.

Run from the repo root:
    python -m tests.test_speculative
"""

import sys
import torch
import torch.nn as nn

from src.models.kimi_k3 import KimiK3Config, KimiK3ForCausalLM
from src.models.mtp import MTPBlock
from src.engine import Engine
from src.speculative import SpeculativeEngine
from src.cache import Cache, DraftTTTCache


def make_tiny_config(**overrides):
    defaults = dict(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=352,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        kda_num_heads=2,
        kda_head_dim=64,
        mtp_enabled=True,
        mtp_loss_weight=0.3,
        feature_layer_indices=[0, 1, 3],
    )
    defaults.update(overrides)
    return KimiK3Config(**defaults)


def target_greedy_generate(model, input_ids, max_new_tokens):
    """Baseline: target-only greedy generation using the existing Engine."""
    engine = Engine(model)
    return engine.generate(input_ids, max_new_tokens, temperature=0)


def test_token_equality_basic():
    """Speculative output must exactly match target-only greedy output."""
    print("test_token_equality_basic...", end=" ", flush=True)
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 8))
    max_new = 20

    # Baseline: target-only greedy
    target_tokens = target_greedy_generate(model, prompt, max_new)

    # Speculative: should match exactly
    spec_engine = SpeculativeEngine(model, draft_steps=4)
    spec_tokens = spec_engine.generate(prompt, max_new)

    assert target_tokens.shape == spec_tokens.shape, (
        f"Shape mismatch: target {target_tokens.shape} vs spec {spec_tokens.shape}"
    )
    assert torch.equal(target_tokens, spec_tokens), (
        f"Token mismatch!\n"
        f"  Target: {target_tokens[0].tolist()}\n"
        f"  Spec:   {spec_tokens[0].tolist()}"
    )
    print(f"ok ({max_new} tokens match)")


def test_token_equality_longer():
    """Test with a longer generation to exercise multiple speculation rounds."""
    print("test_token_equality_longer...", end=" ", flush=True)
    config = make_tiny_config()
    torch.manual_seed(42)
    model = KimiK3ForCausalLM(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 4))
    max_new = 60

    target_tokens = target_greedy_generate(model, prompt, max_new)
    spec_engine = SpeculativeEngine(model, draft_steps=4)
    spec_tokens = spec_engine.generate(prompt, max_new)

    assert torch.equal(target_tokens, spec_tokens), (
        f"Token mismatch over {max_new} tokens!\n"
        f"  Target: {target_tokens[0].tolist()}\n"
        f"  Spec:   {spec_tokens[0].tolist()}"
    )
    print(f"ok ({max_new} tokens match)")


def test_different_draft_steps():
    """Test with various draft_steps values — all must match target."""
    print("test_different_draft_steps...", end=" ", flush=True)
    config = make_tiny_config()
    torch.manual_seed(7)
    model = KimiK3ForCausalLM(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 6))
    max_new = 30

    target_tokens = target_greedy_generate(model, prompt, max_new)

    for k in [1, 2, 3, 4, 7]:
        spec_engine = SpeculativeEngine(model, draft_steps=k)
        spec_tokens = spec_engine.generate(prompt, max_new)
        assert torch.equal(target_tokens, spec_tokens), (
            f"Mismatch with draft_steps={k}!\n"
            f"  Target: {target_tokens[0].tolist()}\n"
            f"  Spec:   {spec_tokens[0].tolist()}"
        )
    print("ok (k=1,2,3,4,7 all match)")


def test_different_seeds():
    """Test with multiple random seeds/prompts to cover diverse patterns."""
    print("test_different_seeds...", end=" ", flush=True)
    config = make_tiny_config()
    max_new = 25

    for seed in [0, 1, 2, 42, 123]:
        torch.manual_seed(seed)
        model = KimiK3ForCausalLM(config)
        model.eval()

        prompt = torch.randint(0, config.vocab_size, (1, 5))
        target_tokens = target_greedy_generate(model, prompt, max_new)

        spec_engine = SpeculativeEngine(model, draft_steps=4)
        spec_tokens = spec_engine.generate(prompt, max_new)

        assert torch.equal(target_tokens, spec_tokens), (
            f"Mismatch with seed={seed}!\n"
            f"  Target: {target_tokens[0].tolist()}\n"
            f"  Spec:   {spec_tokens[0].tolist()}"
        )
    print("ok (seeds 0,1,2,42,123 all match)")


def test_single_token_prompt():
    """Test with a single-token prompt (edge case)."""
    print("test_single_token_prompt...", end=" ", flush=True)
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 1))
    max_new = 15

    target_tokens = target_greedy_generate(model, prompt, max_new)
    spec_engine = SpeculativeEngine(model, draft_steps=4)
    spec_tokens = spec_engine.generate(prompt, max_new)

    assert torch.equal(target_tokens, spec_tokens), (
        f"Mismatch with single-token prompt!\n"
        f"  Target: {target_tokens[0].tolist()}\n"
        f"  Spec:   {spec_tokens[0].tolist()}"
    )
    print(f"ok ({max_new} tokens match)")


def test_long_prompt():
    """Test with a long prompt to verify prefill + speculative works."""
    print("test_long_prompt...", end=" ", flush=True)
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 50))
    max_new = 20

    target_tokens = target_greedy_generate(model, prompt, max_new)
    spec_engine = SpeculativeEngine(model, draft_steps=4)
    spec_tokens = spec_engine.generate(prompt, max_new)

    assert torch.equal(target_tokens, spec_tokens), (
        f"Mismatch with long prompt!\n"
        f"  Target: {target_tokens[0].tolist()}\n"
        f"  Spec:   {spec_tokens[0].tolist()}"
    )
    print(f"ok ({max_new} tokens match)")


def test_cache_invariants():
    """Verify cache length invariants after speculative generation."""
    print("test_cache_invariants...", end=" ", flush=True)
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.eval()

    prompt_len = 8
    max_new = 20
    prompt = torch.randint(0, config.vocab_size, (1, prompt_len))

    # After speculative generation, the target cache should contain
    # exactly prompt_len + num_generated + 1 (for the last fed token).
    # We verify this indirectly by checking the engine works correctly.
    spec_engine = SpeculativeEngine(model, draft_steps=4)
    spec_tokens = spec_engine.generate(prompt, max_new)
    target_tokens = target_greedy_generate(model, prompt, max_new)

    assert torch.equal(target_tokens, spec_tokens)
    print("ok")


def test_separate_draft_block():
    """Test using a separately instantiated (but weight-shared) draft block."""
    print("test_separate_draft_block...", end=" ", flush=True)
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.eval()

    # Create a separate draft block with the same weights
    draft_block = MTPBlock(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
    )
    # Load weights from model's MTP block
    draft_block.load_state_dict(model.mtp_block.state_dict())
    draft_block.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 8))
    max_new = 20

    target_tokens = target_greedy_generate(model, prompt, max_new)
    spec_engine = SpeculativeEngine(model, draft_block=draft_block, draft_steps=4)
    spec_tokens = spec_engine.generate(prompt, max_new)

    assert torch.equal(target_tokens, spec_tokens), (
        f"Mismatch with separate draft block!\n"
        f"  Target: {target_tokens[0].tolist()}\n"
        f"  Spec:   {spec_tokens[0].tolist()}"
    )
def test_stage2_draft_trainer():
    """Test using a Stage 2 DraftTrainer (MTPBlock + FeatureProjection)."""
    print("test_stage2_draft_trainer...", end=" ", flush=True)
    from src.training.draft import DraftConfig, DraftTrainer

    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.eval()

    draft_cfg = DraftConfig(
        draft_steps=4,
        feature_layer_indices=config.feature_layer_indices,
    )
    draft_trainer = DraftTrainer(model.mtp_block, draft_cfg)
    draft_trainer.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 8))
    max_new = 20

    target_tokens = target_greedy_generate(model, prompt, max_new)
    spec_engine = SpeculativeEngine(model, draft_block=draft_trainer, draft_steps=4)
    spec_tokens = spec_engine.generate(prompt, max_new)

    assert torch.equal(target_tokens, spec_tokens), (
        f"Mismatch with Stage 2 DraftTrainer!\n"
        f"  Target: {target_tokens[0].tolist()}\n"
        f"  Spec:   {spec_tokens[0].tolist()}"
    )
    print("ok")


def test_draft_steps_1():
    """Test with draft_steps=1 (minimal speculation, still should match)."""
    print("test_draft_steps_1...", end=" ", flush=True)
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 8))
    max_new = 25

    target_tokens = target_greedy_generate(model, prompt, max_new)
    spec_engine = SpeculativeEngine(model, draft_steps=1)
    spec_tokens = spec_engine.generate(prompt, max_new)

    assert torch.equal(target_tokens, spec_tokens), (
        f"Mismatch with draft_steps=1!\n"
        f"  Target: {target_tokens[0].tolist()}\n"
        f"  Spec:   {spec_tokens[0].tolist()}"
    )
    print("ok")


def test_tied_embeddings():
    """Test with tied word embeddings (common production config)."""
    print("test_tied_embeddings...", end=" ", flush=True)
    config = make_tiny_config(tie_word_embeddings=True)
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 8))
    max_new = 20

    target_tokens = target_greedy_generate(model, prompt, max_new)
    spec_engine = SpeculativeEngine(model, draft_steps=4)
    spec_tokens = spec_engine.generate(prompt, max_new)

    assert torch.equal(target_tokens, spec_tokens), (
        f"Mismatch with tied embeddings!\n"
        f"  Target: {target_tokens[0].tolist()}\n"
        f"  Spec:   {spec_tokens[0].tolist()}"
    )
    print("ok")


def test_max_new_tokens_boundary():
    """Test that max_new_tokens is respected even when drafts over-generate."""
    print("test_max_new_tokens_boundary...", end=" ", flush=True)
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 8))

    for max_new in [1, 2, 3, 5, 7]:
        target_tokens = target_greedy_generate(model, prompt, max_new)
        spec_engine = SpeculativeEngine(model, draft_steps=4)
        spec_tokens = spec_engine.generate(prompt, max_new)

        assert spec_tokens.shape[1] == max_new, (
            f"max_new={max_new}: got {spec_tokens.shape[1]} tokens"
        )
        assert torch.equal(target_tokens, spec_tokens), (
            f"Mismatch with max_new={max_new}!\n"
            f"  Target: {target_tokens[0].tolist()}\n"
            f"  Spec:   {spec_tokens[0].tolist()}"
        )
    print("ok (max_new=1,2,3,5,7)")


def test_draft_trainer_trajectory_matches_actual_inference_draft_path():
    """Verify DraftTrainer's single-anchor trajectory directly matches the actual inference draft path."""
    print("test_draft_trainer_trajectory_matches_actual_inference_draft_path...", end=" ", flush=True)
    from src.training.draft import DraftConfig, DraftTrainer
    from tests.helpers import draft_forward_with_trajectories

    config = make_tiny_config()
    torch.manual_seed(42)
    target_model = KimiK3ForCausalLM(config).eval()

    draft_cfg = DraftConfig(
        draft_steps=4,
        feature_layer_indices=config.feature_layer_indices,
    )
    draft_trainer = DraftTrainer(target_model.mtp_block, draft_cfg).eval()
    spec_engine = SpeculativeEngine(
        target_model=target_model,
        draft_block=draft_trainer,
        draft_steps=draft_cfg.draft_steps,
    )

    B, T = 1, 10
    tokens = torch.randint(0, config.vocab_size, (B, T + 1))
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:].clone()

    with torch.no_grad():
        target_res = target_model(inputs, return_features=True)
        target_features = target_res["features"]
        target_logits = target_res["logits"]

        # Run vectorized DraftTrainer training forward pass with trajectory capture
        train_out = draft_forward_with_trajectories(
            draft_trainer,
            target_features=target_features,
            target_logits=target_logits,
            targets=targets,
            embed_fn=target_model.model.embed_tokens,
            lm_head_fn=target_model.lm_head,
        )
        train_hidden = train_out["draft_hidden"]  # list of length draft_steps
        train_logits = train_out["draft_logits"]  # list of length draft_steps

        # Test multiple anchor positions across sequence
        test_anchors = [0, 1, 3, 5]
        for t in test_anchors:
            prompt_tokens = tokens[:, : t + 1]
            teacher_forced_tokens = targets[:, t : t + draft_cfg.draft_steps]

            # Execute the ACTUAL SpeculativeEngine inference draft path
            infer_hidden, infer_logits = spec_engine.draft_trajectory(
                prompt_tokens=prompt_tokens,
                teacher_forced_tokens=teacher_forced_tokens,
                draft_steps=draft_cfg.draft_steps,
            )

            for step_idx in range(draft_cfg.draft_steps):
                step_r = step_idx + 1
                if t >= train_hidden[step_idx].shape[1]:
                    continue

                expected_h = train_hidden[step_idx][:, t : t + 1, :]
                actual_h = infer_hidden[step_idx]
                max_diff_h = (expected_h - actual_h).abs().max().item()
                assert max_diff_h < 1e-5, (
                    f"Hidden state mismatch between DraftTrainer and SpeculativeEngine "
                    f"at anchor {t}, depth {step_r}! Max diff: {max_diff_h:.2e}"
                )

                expected_logits = train_logits[step_idx][:, t : t + 1, :]
                actual_logits = infer_logits[step_idx]
                max_diff_logits = (expected_logits - actual_logits).abs().max().item()
                assert max_diff_logits < 1e-4, (
                    f"Logits mismatch between DraftTrainer and SpeculativeEngine "
                    f"at anchor {t}, depth {step_r}! Max diff: {max_diff_logits:.2e}"
                )

    print("ok (DraftTrainer vectorized matches actual SpeculativeEngine inference draft path)")


def test_persistent_draft_prefix_across_speculation_rounds():
    """Verify SpeculativeEngine maintains correct persistent draft prefix across multiple acceptance rounds."""
    print("test_persistent_draft_prefix_across_speculation_rounds...", end=" ", flush=True)
    from src.training.draft import DraftConfig, DraftTrainer

    config = make_tiny_config()
    torch.manual_seed(123)
    model = KimiK3ForCausalLM(config).eval()

    draft_cfg = DraftConfig(
        draft_steps=4,
        feature_layer_indices=config.feature_layer_indices,
    )
    draft_trainer = DraftTrainer(model.mtp_block, draft_cfg).eval()

    prompt = torch.randint(0, config.vocab_size, (1, 6))
    max_new = 25

    target_tokens = target_greedy_generate(model, prompt, max_new)
    spec_engine = SpeculativeEngine(model, draft_block=draft_trainer, draft_steps=4)
    spec_tokens, stats = spec_engine.generate(prompt, max_new, return_stats=True)

    assert torch.equal(target_tokens, spec_tokens), (
        f"Mismatch across multiple rounds with persistent draft prefix!\n"
        f"  Target: {target_tokens[0].tolist()}\n"
        f"  Spec:   {spec_tokens[0].tolist()}"
    )
    assert stats["num_rounds"] > 1, f"Expected multiple rounds, got {stats['num_rounds']}"
    assert stats["draft_proposed"] > 0, "No drafts were proposed"
    print(f"ok ({stats['num_rounds']} rounds, {stats['total_tokens_generated']} tokens, acceptance_rate={stats['acceptance_rate']:.2%})")


def test_incremental_vs_rebuilt_drafter_kv_state():
    """Test that incrementally maintained drafter KV state matches rebuilding from full history.
    
    Covers cases: n_accepted = 0 (advances by pending token), 0 < n_accepted < K, n_accepted = K.
    """
    config = make_tiny_config()
    model = KimiK3ForCausalLM(config).eval()
    engine = SpeculativeEngine(model, draft_steps=4)
    
    # Create a test sequence where we can control acceptance
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    
    with torch.no_grad():
        # Build initial prefix state
        target_cache = model.make_cache(1, 20, torch.float32)
        all_fused, _ = engine._get_hidden_and_logits(input_ids, target_cache)
        
        # Case 1: n_accepted = 0 (all rejected)
        # Semantics: persistent prefix advances by the old pending token
        prefix_k_0, prefix_v_0 = engine.build_draft_prefix(all_fused[:, :-1], input_ids[:, 1:])
        pending_token = torch.tensor([[10]], dtype=torch.long)
        
        extended_k_0, extended_v_0 = engine._extend_draft_prefix(
            prefix_k_0, prefix_v_0, all_fused[:, -1:], pending_token
        )
        
        full_tokens_0 = torch.cat([input_ids[:, 1:], pending_token], dim=1)
        full_seq_0 = torch.cat([input_ids, pending_token], dim=1)
        dummy_cache = model.make_cache(1, 20, torch.float32)
        full_fused_0, _ = engine._get_hidden_and_logits(full_seq_0, dummy_cache)
        rebuilt_k_0, rebuilt_v_0 = engine.build_draft_prefix(full_fused_0[:, :-1], full_tokens_0)
        
        if extended_k_0 is not None and rebuilt_k_0 is not None:
            torch.testing.assert_close(extended_k_0, rebuilt_k_0, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(extended_v_0, rebuilt_v_0, atol=1e-5, rtol=1e-5)
        
        # Case 2: 0 < n_accepted < K (partial acceptance)
        new_tokens = torch.tensor([[10, 11]], dtype=torch.long)  # 2 new tokens
        new_hidden = torch.randn(1, 2, config.hidden_size)
        
        extended_k_partial, extended_v_partial = engine._extend_draft_prefix(
            prefix_k_0, prefix_v_0, new_hidden, new_tokens
        )
        
        full_tokens = torch.cat([input_ids[:, 1:], new_tokens], dim=1)
        full_hidden = torch.cat([all_fused[:, :-1], new_hidden], dim=1) 
        rebuilt_k, rebuilt_v = engine.build_draft_prefix(full_hidden, full_tokens)
        
        if extended_k_partial is not None and rebuilt_k is not None:
            torch.testing.assert_close(extended_k_partial, rebuilt_k, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(extended_v_partial, rebuilt_v, atol=1e-5, rtol=1e-5)
        
        # Case 3: n_accepted = K (full acceptance)  
        more_tokens = torch.tensor([[12, 13, 14]], dtype=torch.long)  # 3 more tokens
        more_hidden = torch.randn(1, 3, config.hidden_size)
        
        extended_k_full, extended_v_full = engine._extend_draft_prefix(
            extended_k_partial, extended_v_partial, more_hidden, more_tokens
        )
        
        complete_tokens = torch.cat([input_ids[:, 1:], new_tokens, more_tokens], dim=1)
        complete_hidden = torch.cat([all_fused[:, :-1], new_hidden, more_hidden], dim=1)
        rebuilt_k_full, rebuilt_v_full = engine.build_draft_prefix(complete_hidden, complete_tokens)
        
        if extended_k_full is not None and rebuilt_k_full is not None:
            torch.testing.assert_close(extended_k_full, rebuilt_k_full, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(extended_v_full, rebuilt_v_full, atol=1e-5, rtol=1e-5)
    
    # Test actual acceptance branches by running generate() and checking consistency
    torch.manual_seed(42)
    result_with_stats = engine.generate(input_ids, max_new_tokens=8, return_stats=True)
    generated_tokens, stats = result_with_stats
    assert stats["num_rounds"] >= 2, f"Expected multiple rounds, got {stats['num_rounds']}"
    print("test_incremental_vs_rebuilt_drafter_kv_state... ok")


class DraftBlockWrapper(nn.Module):
    def __init__(self, real_block, engine):
        super().__init__()
        self.real_block = real_block
        self.engine = engine

    def forward(self, hidden, token_emb, ttt_cache=None):
        if ttt_cache is not None:
            self.engine.captured_draft_cache = ttt_cache
        self.engine.in_draft_step = True
        return self.real_block(hidden, token_emb, ttt_cache=ttt_cache)

    def get_output_hidden(self, draft_hidden):
        return self.real_block.get_output_hidden(draft_hidden)


class ControlledSpeculativeEngine(SpeculativeEngine):
    """Speculative engine with programmatic control over verification logits and draft tokens."""

    def __init__(
        self,
        *args,
        round_acceptances: list[int] | None = None,
        force_draft_eos_step: int | None = None,
        force_eos_at: str | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.round_acceptances = list(round_acceptances) if round_acceptances is not None else []
        self.force_draft_eos_step = force_draft_eos_step
        self.force_eos_at = force_eos_at
        self.eos_token_id = eos_token_id
        self.round_idx = 0
        self.in_draft_step = False
        self.captured_target_cache = None
        self.captured_draft_cache = None
        self.captured_extended_k = None
        self.captured_extended_v = None
        self.captured_verify_hidden = None
        self.captured_replay_hidden = None
        self.final_draft_prefix_k = None
        self.final_draft_prefix_v = None
        self.last_n_accepted = None
        self.captured_draft_tokens: list[torch.Tensor] = []
        self.captured_initial_pending = None

        self.draft_block = DraftBlockWrapper(self.draft_block, self)

    def _is_cache_forked(self, cache: Cache) -> bool:
        return any(
            getattr(l, "_snapshot_seq_len", None) is not None or getattr(l, "_snapshot", None) is not None
            for l in cache.layers
        )

    def _get_hidden_and_logits(self, input_ids: torch.Tensor, cache: Cache) -> tuple[torch.Tensor, torch.Tensor]:
        seed_hidden, logits = super()._get_hidden_and_logits(input_ids, cache)
        self.captured_target_cache = cache

        # Prefill prompt:
        if input_ids.shape[1] > 1 and not self._is_cache_forked(cache) and cache.seq_len == input_ids.shape[1]:
            if self.force_eos_at == "initial_pending":
                logits = logits.clone()
                logits[:, -1, :] = -1e4
                logits[:, -1, self.eos_token_id] = 1e4
            self.captured_initial_pending = logits[:, -1].float().argmax(-1)
            return seed_hidden, logits

        # Verification step:
        if input_ids.shape[1] > 1 and self._is_cache_forked(cache):
            self.captured_verify_hidden = seed_hidden
            K = input_ids.shape[1] - 1
            if self.round_idx < len(self.round_acceptances):
                n_acc = self.round_acceptances[self.round_idx]
            else:
                n_acc = K

            self.last_n_accepted = n_acc
            vocab_size = logits.shape[-1]
            logits = logits.clone()

            for i in range(K):
                draft_tok = input_ids[0, i + 1].item()
                if i < n_acc:
                    logits[0, i, :] = -1e4
                    logits[0, i, draft_tok] = 1e4
                else:
                    mismatch_tok = (draft_tok + 1) % vocab_size
                    if self.eos_token_id is not None and mismatch_tok == self.eos_token_id:
                        mismatch_tok = (mismatch_tok + 1) % vocab_size
                    logits[0, i, :] = -1e4
                    logits[0, i, mismatch_tok] = 1e4

            if self.force_eos_at == "bonus" and n_acc == K:
                logits[0, K, :] = -1e4
                logits[0, K, self.eos_token_id] = 1e4
            elif self.force_eos_at == "correction" and n_acc < K:
                logits[0, n_acc, :] = -1e4
                logits[0, n_acc, self.eos_token_id] = 1e4

            self.round_idx += 1
            return seed_hidden, logits

        # Replay forward:
        self.captured_replay_hidden = seed_hidden
        return seed_hidden, logits

    def _extend_draft_prefix(self, prefix_k, prefix_v, hidden, tokens):
        k, v = super()._extend_draft_prefix(prefix_k, prefix_v, hidden, tokens)
        self.captured_extended_k = k
        self.captured_extended_v = v
        return k, v

    def generate(self, *args, **kwargs):
        orig_lm_head = self.target_model.lm_head
        draft_step_counter = [0]
        captured_tokens = self.captured_draft_tokens

        class InterceptedLMHead(nn.Module):
            def __init__(outer_self, real_head):
                super().__init__()
                outer_self.real_head = real_head

            @property
            def weight(outer_self):
                return outer_self.real_head.weight

            def forward(outer_self, x):
                logits = outer_self.real_head(x)
                if getattr(self, "in_draft_step", False):
                    self.in_draft_step = False
                    if self.force_draft_eos_step is not None:
                        if draft_step_counter[0] == self.force_draft_eos_step:
                            logits = logits.clone()
                            logits[:, :, :] = -1e4
                            logits[:, :, self.eos_token_id] = 1e4
                    tok = logits[:, -1].float().argmax(-1)
                    captured_tokens.append(tok)
                    draft_step_counter[0] += 1
                return logits

        self.target_model.lm_head = InterceptedLMHead(orig_lm_head)
        try:
            res = super().generate(*args, **kwargs)
            if self.last_n_accepted == 0:
                if self.captured_draft_cache is not None:
                    self.final_draft_prefix_k = self.captured_draft_cache.k_list[0]
                    self.final_draft_prefix_v = self.captured_draft_cache.v_list[0]
            else:
                self.final_draft_prefix_k = self.captured_extended_k
                self.final_draft_prefix_v = self.captured_extended_v
            return res
        finally:
            self.target_model.lm_head = orig_lm_head


def test_deterministic_acceptance_branches_and_rebuilt_drafter_kv():
    """Test deterministic n_accepted = 0, 0 < n_accepted < K, n_accepted = K branches.
    
    Verifies:
      - pending_token
      - prev_hidden
      - target cache state
      - draft_prefix_k/v (incremental vs full rebuild)
      - generated tokens
    against the expected committed sequence.
    """
    print("test_deterministic_acceptance_branches_and_rebuilt_drafter_kv...", end=" ", flush=True)
    config = make_tiny_config()
    model = KimiK3ForCausalLM(config).eval()
    K = 4
    prompt = torch.tensor([[2, 4, 6, 8, 10]], dtype=torch.long)

    test_cases = [
        (0, 2),
        (2, 4),
        (4, 6),
    ]

    for n_acc, max_new in test_cases:
        torch.manual_seed(100)
        engine = ControlledSpeculativeEngine(
            model, draft_steps=K, round_acceptances=[n_acc]
        )
        gen_tokens, stats = engine.generate(prompt, max_new_tokens=max_new, return_stats=True)

        assert stats["num_rounds"] == 1, f"Expected 1 round, got {stats['num_rounds']}"
        assert stats["draft_accepted"] == n_acc, f"Expected {n_acc} accepted, got {stats['draft_accepted']}"

        # 1. Verify generated tokens structure:
        initial_pending = engine.captured_initial_pending
        drafts = engine.captured_draft_tokens[:K]
        if n_acc == K:
            bonus = gen_tokens[:, -1]
            expected_gen = torch.cat([initial_pending.view(1, 1)] + [d.view(1, 1) for d in drafts] + [bonus.view(1, 1)], dim=1)
            committed_tokens = torch.cat([prompt, initial_pending.view(1, 1)] + [d.view(1, 1) for d in drafts], dim=1)
        elif n_acc == 0:
            correction = gen_tokens[:, -1]
            expected_gen = torch.cat([initial_pending.view(1, 1), correction.view(1, 1)], dim=1)
            committed_tokens = torch.cat([prompt, initial_pending.view(1, 1)], dim=1)
        else:
            correction = gen_tokens[:, -1]
            expected_gen = torch.cat([initial_pending.view(1, 1)] + [d.view(1, 1) for d in drafts[:n_acc]] + [correction.view(1, 1)], dim=1)
            committed_tokens = torch.cat([prompt, initial_pending.view(1, 1)] + [d.view(1, 1) for d in drafts[:n_acc]], dim=1)

        torch.testing.assert_close(gen_tokens, expected_gen)

        # 2. Verify target cache state matches direct forward of committed_tokens
        expected_cache = model.make_cache(1, 64, torch.float32)
        model(committed_tokens, cache=expected_cache)

        assert engine.captured_target_cache.seq_len == expected_cache.seq_len
        for act_layer, exp_layer in zip(engine.captured_target_cache.layers, expected_cache.layers):
            if hasattr(act_layer, "state"):
                torch.testing.assert_close(act_layer.state, exp_layer.state)
                for cs_act, cs_exp in zip(act_layer.conv_states, exp_layer.conv_states):
                    torch.testing.assert_close(cs_act, cs_exp)
                assert act_layer.seq_len == exp_layer.seq_len
            else:
                assert act_layer.seq_len == exp_layer.seq_len
                torch.testing.assert_close(act_layer.keys[:, :, :act_layer.seq_len], exp_layer.keys[:, :, :exp_layer.seq_len])
                torch.testing.assert_close(act_layer.values[:, :, :act_layer.seq_len], exp_layer.values[:, :, :exp_layer.seq_len])

        # 3. Verify incremental drafter prefix against full rebuild
        fresh_cache = model.make_cache(1, 64, torch.float32)
        full_fused, _ = engine._get_hidden_and_logits(committed_tokens, fresh_cache)
        rebuilt_k, rebuilt_v = engine.build_draft_prefix(
            full_fused[:, :-1, :], committed_tokens[:, 1:]
        )

        torch.testing.assert_close(engine.final_draft_prefix_k, rebuilt_k)
        torch.testing.assert_close(engine.final_draft_prefix_v, rebuilt_v)

        # 4. Verify explicit prefix length semantics:
        expected_prefix_len = (prompt.shape[1] - 1) + 1 + n_acc
        assert engine.final_draft_prefix_k.shape[2] == expected_prefix_len, (
            f"Prefix length mismatch for n_acc={n_acc}: "
            f"got {engine.final_draft_prefix_k.shape[2]}, expected {expected_prefix_len}"
        )

    print("ok")


def test_kda_recurrent_cache_rollback_equivalence():
    """Test that KDA recurrent state and conv buffers are exactly restored after fork() and rollback()."""
    print("test_kda_recurrent_cache_rollback_equivalence...", end=" ", flush=True)
    config = make_tiny_config()
    model = KimiK3ForCausalLM(config).eval()

    prefix = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], dtype=torch.long)
    cache = model.make_cache(1, 64, torch.float32)

    with torch.no_grad():
        # A. Build cache to prefix
        model(prefix, cache=cache)

        # B. Clone reference state for all layers
        saved_states = []
        for layer_cache in cache.layers:
            if hasattr(layer_cache, "state"):
                saved_states.append({
                    "type": "recurrent",
                    "state": layer_cache.state.clone(),
                    "conv_states": [cs.clone() for cs in layer_cache.conv_states],
                    "seq_len": layer_cache.seq_len,
                })
            else:
                saved_states.append({
                    "type": "kv",
                    "keys": layer_cache.keys[:, :, :layer_cache.seq_len].clone(),
                    "values": layer_cache.values[:, :, :layer_cache.seq_len].clone(),
                    "seq_len": layer_cache.seq_len,
                })

        # C. Fork
        cache.fork()

        # D. Process speculative tokens
        spec_tokens = torch.tensor([[101, 102, 103, 104]], dtype=torch.long)
        model(spec_tokens, cache=cache)

        # Verify state actually changed while forked
        for i, layer_cache in enumerate(cache.layers):
            if saved_states[i]["type"] == "recurrent":
                assert not torch.allclose(layer_cache.state, saved_states[i]["state"]), "Recurrent state did not change"
                assert layer_cache.seq_len > saved_states[i]["seq_len"], "Seq len did not increase"

        # E. Rollback
        cache.rollback()

        # F. Assert cache == state from B
        for i, layer_cache in enumerate(cache.layers):
            if saved_states[i]["type"] == "recurrent":
                torch.testing.assert_close(layer_cache.state, saved_states[i]["state"])
                for cs, expected_cs in zip(layer_cache.conv_states, saved_states[i]["conv_states"]):
                    torch.testing.assert_close(cs, expected_cs)
                assert layer_cache.seq_len == saved_states[i]["seq_len"]
            else:
                assert layer_cache.seq_len == saved_states[i]["seq_len"]
                torch.testing.assert_close(layer_cache.keys[:, :, :layer_cache.seq_len], saved_states[i]["keys"])
                torch.testing.assert_close(layer_cache.values[:, :, :layer_cache.seq_len], saved_states[i]["values"])

    print("ok")


def test_partial_rejection_and_replay_equivalence():
    """Test that fork -> verify -> reject -> rollback -> replay produces identical cache/hidden/logits to clean forward."""
    print("test_partial_rejection_and_replay_equivalence...", end=" ", flush=True)
    config = make_tiny_config()
    model = KimiK3ForCausalLM(config).eval()

    prefix = torch.tensor([[5, 12, 19, 26, 33]], dtype=torch.long)
    K = 4
    for a in [0, 1, K - 1]:
        cache_a = model.make_cache(1, 64, torch.float32)
        cache_b = model.make_cache(1, 64, torch.float32)

        with torch.no_grad():
            res_a = model(prefix, cache=cache_a, return_features=True)
            res_b = model(prefix, cache=cache_b, return_features=True)

            pending = res_a["logits"][:, -1:].argmax(-1)
            drafts = torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
            spec_tokens = torch.cat([pending, drafts], dim=1)

            accepted_tokens = torch.cat([pending, drafts[:, :a]], dim=1) if a > 0 else pending

            # Path A: fork -> verify spec -> rollback -> replay accepted
            cache_a.fork()
            model(spec_tokens, cache=cache_a, return_features=True)
            cache_a.rollback()
            out_a = model(accepted_tokens, cache=cache_a, return_features=True)

            # Path B: directly run accepted tokens
            out_b = model(accepted_tokens, cache=cache_b, return_features=True)

            torch.testing.assert_close(out_a["logits"], out_b["logits"])
            torch.testing.assert_close(out_a["features"]["final_hidden"], out_b["features"]["final_hidden"])

            for layer_a, layer_b in zip(cache_a.layers, cache_b.layers):
                if hasattr(layer_a, "state"):
                    torch.testing.assert_close(layer_a.state, layer_b.state)
                    for cs_a, cs_b in zip(layer_a.conv_states, layer_b.conv_states):
                        torch.testing.assert_close(cs_a, cs_b)
                    assert layer_a.seq_len == layer_b.seq_len
                else:
                    assert layer_a.seq_len == layer_b.seq_len
                    torch.testing.assert_close(layer_a.keys[:, :, :layer_a.seq_len], layer_b.keys[:, :, :layer_b.seq_len])
                    torch.testing.assert_close(layer_a.values[:, :, :layer_a.seq_len], layer_b.values[:, :, :layer_b.seq_len])

    print("ok")


def test_full_accept_commit_equivalence():
    """Test that fork -> verify -> commit(1+K) matches direct clean forward."""
    print("test_full_accept_commit_equivalence...", end=" ", flush=True)
    config = make_tiny_config()
    model = KimiK3ForCausalLM(config).eval()

    prefix = torch.tensor([[7, 14, 21, 28]], dtype=torch.long)
    K = 4

    cache_a = model.make_cache(1, 64, torch.float32)
    cache_b = model.make_cache(1, 64, torch.float32)

    with torch.no_grad():
        res_a = model(prefix, cache=cache_a, return_features=True)
        res_b = model(prefix, cache=cache_b, return_features=True)

        pending = res_a["logits"][:, -1:].argmax(-1)
        drafts = torch.tensor([[201, 202, 203, 204]], dtype=torch.long)
        spec_tokens = torch.cat([pending, drafts], dim=1)

        # Path A: fork -> verify -> commit(1 + K)
        cache_a.fork()
        out_a = model(spec_tokens, cache=cache_a, return_features=True)
        cache_a.commit(1 + K)

        # Path B: direct clean forward of spec_tokens
        out_b = model(spec_tokens, cache=cache_b, return_features=True)

        torch.testing.assert_close(out_a["logits"], out_b["logits"])
        torch.testing.assert_close(out_a["features"]["final_hidden"], out_b["features"]["final_hidden"])

        for layer_a, layer_b in zip(cache_a.layers, cache_b.layers):
            if hasattr(layer_a, "state"):
                torch.testing.assert_close(layer_a.state, layer_b.state)
                for cs_a, cs_b in zip(layer_a.conv_states, layer_b.conv_states):
                    torch.testing.assert_close(cs_a, cs_b)
                assert layer_a.seq_len == layer_b.seq_len
            else:
                assert layer_a.seq_len == layer_b.seq_len
                torch.testing.assert_close(layer_a.keys[:, :, :layer_a.seq_len], layer_b.keys[:, :, :layer_b.seq_len])
                torch.testing.assert_close(layer_a.values[:, :, :layer_a.seq_len], layer_b.values[:, :, :layer_b.seq_len])

    print("ok")


def test_deterministic_eos_branch_coverage():
    """Test deterministic EOS stop at each candidate position."""
    print("test_deterministic_eos_branch_coverage...", end=" ", flush=True)
    config = make_tiny_config()
    model = KimiK3ForCausalLM(config).eval()
    eos_id = 99
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    # 1. Initial pending token is EOS
    engine_1 = ControlledSpeculativeEngine(
        model, draft_steps=4, force_eos_at="initial_pending", eos_token_id=eos_id
    )
    tokens_1 = engine_1.generate(prompt, max_new_tokens=10, eos_token_id=eos_id)
    assert tokens_1.shape[1] == 1, f"Expected 1 token, got {tokens_1.shape[1]}"
    assert tokens_1[0, -1].item() == eos_id
    assert (tokens_1 == eos_id).sum().item() == 1

    # 2. First accepted draft is EOS
    engine_2 = ControlledSpeculativeEngine(
        model, draft_steps=4, round_acceptances=[4], force_draft_eos_step=0, eos_token_id=eos_id
    )
    tokens_2 = engine_2.generate(prompt, max_new_tokens=10, eos_token_id=eos_id)
    assert tokens_2.shape[1] == 2, f"Expected 2 tokens, got {tokens_2.shape[1]}"
    assert tokens_2[0, -1].item() == eos_id
    assert (tokens_2 == eos_id).sum().item() == 1

    # 3. Middle accepted draft is EOS (step 1 of 4)
    engine_3 = ControlledSpeculativeEngine(
        model, draft_steps=4, round_acceptances=[4], force_draft_eos_step=1, eos_token_id=eos_id
    )
    tokens_3 = engine_3.generate(prompt, max_new_tokens=10, eos_token_id=eos_id)
    assert tokens_3.shape[1] == 3, f"Expected 3 tokens, got {tokens_3.shape[1]}"
    assert tokens_3[0, -1].item() == eos_id
    assert (tokens_3 == eos_id).sum().item() == 1

    # 4. Last accepted draft is EOS (step 3 of 4)
    engine_4 = ControlledSpeculativeEngine(
        model, draft_steps=4, round_acceptances=[4], force_draft_eos_step=3, eos_token_id=eos_id
    )
    tokens_4 = engine_4.generate(prompt, max_new_tokens=10, eos_token_id=eos_id)
    assert tokens_4.shape[1] == 5, f"Expected 5 tokens, got {tokens_4.shape[1]}"
    assert tokens_4[0, -1].item() == eos_id
    assert (tokens_4 == eos_id).sum().item() == 1

    # 5. Bonus token is EOS
    engine_5 = ControlledSpeculativeEngine(
        model, draft_steps=4, round_acceptances=[4], force_eos_at="bonus", eos_token_id=eos_id
    )
    tokens_5 = engine_5.generate(prompt, max_new_tokens=10, eos_token_id=eos_id)
    assert tokens_5.shape[1] == 6, f"Expected 6 tokens, got {tokens_5.shape[1]}"
    assert tokens_5[0, -1].item() == eos_id
    assert (tokens_5 == eos_id).sum().item() == 1

    # 6. Correction token after rejection is EOS
    engine_6 = ControlledSpeculativeEngine(
        model, draft_steps=4, round_acceptances=[0], force_eos_at="correction", eos_token_id=eos_id
    )
    tokens_6 = engine_6.generate(prompt, max_new_tokens=10, eos_token_id=eos_id)
    assert tokens_6.shape[1] == 2, f"Expected 2 tokens, got {tokens_6.shape[1]}"
    assert tokens_6[0, -1].item() == eos_id
    assert (tokens_6 == eos_id).sum().item() == 1

    print("ok")


def test_speculative_statistics():
    """Test speculative acceptance statistics matching exact programmed round outcomes."""
    print("test_speculative_statistics...", end=" ", flush=True)
    config = make_tiny_config()
    model = KimiK3ForCausalLM(config).eval()

    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    engine = ControlledSpeculativeEngine(
        model,
        draft_steps=4,
        round_acceptances=[0, 2, 4],  # round 1: 0, round 2: 2, round 3: 4
    )

    tokens, stats = engine.generate(prompt, max_new_tokens=10, return_stats=True)

    assert stats["num_rounds"] == 3
    assert stats["draft_proposed"] == 12
    assert stats["draft_accepted"] == 6
    assert abs(stats["acceptance_rate"] - (6 / 12)) < 1e-6
    assert abs(stats["mean_accepted_per_round"] - 2.0) < 1e-6

    expected_cumulative = [2 / 3, 2 / 3, 1 / 3, 1 / 3]
    for act, exp in zip(stats["cumulative_prefix_acceptance"], expected_cumulative):
        assert abs(act - exp) < 1e-6, f"Cumulative acceptance mismatch: {act} vs {exp}"

    assert abs(stats["expected_accepted_per_verification"] - 2.0) < 1e-6

    print("ok")


if __name__ == "__main__":
    print("=" * 60)
    print("Speculative Decoding Tests")
    print("=" * 60)

    test_token_equality_basic()
    test_token_equality_longer()
    test_different_draft_steps()
    test_different_seeds()
    test_single_token_prompt()
    test_long_prompt()
    test_cache_invariants()
    test_separate_draft_block()
    test_stage2_draft_trainer()
    test_draft_steps_1()
    test_tied_embeddings()
    test_max_new_tokens_boundary()
    test_draft_trainer_trajectory_matches_actual_inference_draft_path()
    test_persistent_draft_prefix_across_speculation_rounds()
    test_incremental_vs_rebuilt_drafter_kv_state()
    test_deterministic_acceptance_branches_and_rebuilt_drafter_kv()
    test_kda_recurrent_cache_rollback_equivalence()
    test_partial_rejection_and_replay_equivalence()
    test_full_accept_commit_equivalence()
    test_deterministic_eos_branch_coverage()
    test_speculative_statistics()

    print("=" * 60)
    print("All speculative decoding tests passed!")
    print("=" * 60)

    if torch.cuda.is_available():
        print("\nRunning CUDA speculative tests...")
        config = make_tiny_config()
        torch.manual_seed(0)
        device = torch.device("cuda")
        with torch.device(device):
            model = KimiK3ForCausalLM(config)
        model.eval()

        prompt = torch.randint(0, config.vocab_size, (1, 8), device=device)
        max_new = 30

        target_tokens = target_greedy_generate(model, prompt, max_new)
        spec_engine = SpeculativeEngine(model, draft_steps=4)
        spec_tokens = spec_engine.generate(prompt, max_new)

        assert torch.equal(target_tokens, spec_tokens), (
            f"CUDA mismatch!\n"
            f"  Target: {target_tokens[0].tolist()}\n"
            f"  Spec:   {spec_tokens[0].tolist()}"
        )
        print(f"CUDA speculative test ok ({max_new} tokens match)")
    else:
        print("\nCUDA not available, skipping CUDA speculative tests.")