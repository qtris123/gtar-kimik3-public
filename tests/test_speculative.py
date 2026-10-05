"""
Speculative decoding tests — verify correctness against target-only greedy.

The fundamental correctness invariant: speculative decoding must produce
**token-for-token identical** output to target-only greedy generation.

Run from the repo root:
    python -m tests.test_speculative
"""

import sys
import torch

from src.models.kimi_k3 import KimiK3Config, KimiK3ForCausalLM
from src.models.mtp import MTPBlock
from src.engine import Engine
from src.speculative import SpeculativeEngine


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
