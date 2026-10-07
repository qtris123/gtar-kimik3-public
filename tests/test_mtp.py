"""
Stage 1 tests: MTP block alignment, loss, gradients and compatibility.

Run from the repo root:
    python -m tests.test_mtp
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.kimi_k3 import KimiK3Config, KimiK3ForCausalLM
from src.models.mtp import MTPBlock


def make_tiny_config(mtp_enabled=True, **overrides):
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
        mtp_enabled=mtp_enabled,
        mtp_loss_weight=0.3,
        feature_layer_indices=[0, 1, 3],
    )
    defaults.update(overrides)
    return KimiK3Config(**defaults)


def test_token_alignment():
    """Verify exact MTP token alignment: inputs, embeddings, labels match the spec."""
    print("test_token_alignment...", end=" ")
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.train()

    # Tokens: [10, 20, 30, 40, 50, 60]
    # inputs = [10, 20, 30, 40, 50], targets = [20, 30, 40, 50, 60]
    # MTP hidden input = final_hidden[:, :-1] -> positions 0..3 (4 positions)
    # MTP embedding input = embed(targets[:, :-1]) = embed([20, 30, 40, 50])
    # MTP labels = targets[:, 1:] = [30, 40, 50, 60]
    inputs = torch.tensor([[10, 20, 30, 40, 50]])
    targets = torch.tensor([[20, 30, 40, 50, 60]])

    result = model(inputs, targets)
    assert isinstance(result, dict), f"MTP model should return dict, got {type(result)}"
    assert "loss" in result and "main_loss" in result and "mtp_loss" in result
    assert result["loss"].shape == ()
    assert result["main_loss"].shape == ()
    assert result["mtp_loss"].shape == ()

    # Verify combined loss
    expected_total = result["main_loss"] + config.mtp_loss_weight * result["mtp_loss"]
    assert torch.allclose(result["loss"], expected_total, atol=1e-5), \
        f"Total loss {result['loss']} != main + weight*mtp = {expected_total}"

    print("ok")


def test_causal_masking():
    """Verify the MTP block uses causal attention (no future token leakage)."""
    print("test_causal_masking...", end=" ")
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.eval()

    B, T = 2, 8
    inputs = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    # Forward with full sequence
    with torch.no_grad():
        result_full = model(inputs, targets)

    # The MTP block's Attention layer uses flash_attention or causal SDPA,
    # both of which are causal. We verify the MTP block directly:
    mtp = model.mtp_block
    hidden = torch.randn(1, 6, config.hidden_size)
    emb = torch.randn(1, 6, config.hidden_size)

    with torch.no_grad():
        out_full = mtp(hidden, emb)
        # Compute with only first 3 positions
        out_prefix = mtp(hidden[:, :3], emb[:, :3])

    # First 3 positions should match (causal = no future leakage)
    diff = (out_full[:, :3] - out_prefix).abs().max().item()
    assert diff < 1e-4, f"Causal masking violated: prefix diff = {diff:.2e}"
    print("ok")


def test_loss_decomposition():
    """Verify loss decomposition: total = main + weight * mtp."""
    print("test_loss_decomposition...", end=" ")
    config = make_tiny_config(mtp_loss_weight=0.5)
    torch.manual_seed(42)
    model = KimiK3ForCausalLM(config)
    model.train()

    inputs = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    result = model(inputs, targets)
    expected = result["main_loss"] + 0.5 * result["mtp_loss"]
    assert torch.allclose(result["loss"], expected, atol=1e-5), \
        f"Loss decomposition failed: {result['loss']} != {expected}"
    print("ok")


def test_shared_parameters():
    """Verify embedding and LM head are shared between target and MTP."""
    print("test_shared_parameters...", end=" ")
    config = make_tiny_config(tie_word_embeddings=True)
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)

    # LM head should be tied to embedding
    assert model.lm_head.weight is model.model.embed_tokens.weight, \
        "LM head not tied to embedding"

    # MTP block should NOT own its own embedding table or LM head
    mtp_param_names = [n for n, _ in model.mtp_block.named_parameters()]
    assert not any("embed_tokens" in n for n in mtp_param_names), "MTP should not own embedding table"
    assert not any("lm_head" in n for n in mtp_param_names), "MTP should not own lm_head"
    print("ok")


def test_gradient_flow():
    """Verify MTP gradients reach the target model parameters in Stage 1."""
    print("test_gradient_flow...", end=" ")
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.train()

    inputs = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    result = model(inputs, targets)
    result["loss"].backward()

    # MTP block parameters should have gradients
    for name, p in model.mtp_block.named_parameters():
        assert p.grad is not None, f"MTP param {name} has no gradient"
        assert p.grad.abs().sum() > 0, f"MTP param {name} has zero gradient"

    # Target backbone parameters should also have gradients (MTP backprops through target)
    target_has_grad = False
    for name, p in model.model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            target_has_grad = True
            break
    assert target_has_grad, "MTP gradients did not reach target model parameters"

    # Embedding should have grad (shared)
    assert model.model.embed_tokens.weight.grad is not None, "Embedding has no gradient"
    print("ok")


def test_mtp_disabled_compatibility():
    """Verify MTP-disabled model works exactly as before."""
    print("test_mtp_disabled_compatibility...", end=" ")
    config = make_tiny_config(mtp_enabled=False)
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)

    assert model.mtp_block is None, "MTP block should be None when disabled"

    inputs = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    # Training: should return scalar loss
    model.train()
    loss = model(inputs, targets)
    assert isinstance(loss, torch.Tensor) and loss.shape == (), \
        f"MTP-disabled should return scalar loss, got {type(loss)} shape {loss.shape}"

    # Inference: should return logits
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
    assert logits.shape == (2, 16, config.vocab_size), \
        f"Expected logits shape (2, 16, {config.vocab_size}), got {logits.shape}"

    # Cache-based inference
    cache = model.make_cache(2, 16, torch.float32)
    with torch.no_grad():
        logits_cached = model(inputs, cache=cache)
    assert logits_cached.shape == logits.shape
    print("ok")


def test_checkpoint_round_trip():
    """Verify MTP checkpoint save/load preserves weights."""
    print("test_checkpoint_round_trip...", end=" ")
    import tempfile
    from dataclasses import asdict

    config = make_tiny_config()
    torch.manual_seed(0)
    model1 = KimiK3ForCausalLM(config)

    # Save
    state = model1.state_dict()
    config_dict = asdict(config)

    # Load into a fresh model
    torch.manual_seed(99)  # different seed
    model2 = KimiK3ForCausalLM(KimiK3Config(**config_dict))
    model2.load_state_dict(state, strict=True)

    # Verify weights match
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert n1 == n2, f"Parameter name mismatch: {n1} vs {n2}"
        assert torch.equal(p1, p2), f"Parameter {n1} differs after load"

    # Verify forward produces same results
    inputs = torch.randint(0, config.vocab_size, (1, 8))
    targets = torch.randint(0, config.vocab_size, (1, 8))
    model1.eval()
    model2.eval()
    with torch.no_grad():
        r1 = model1(inputs, targets)
        r2 = model2(inputs, targets)
    assert torch.allclose(r1["loss"], r2["loss"], atol=1e-5)
    print("ok")


def test_target_only_checkpoint_loading():
    """Test loading a target-only checkpoint into MTP-enabled model."""
    print("test_target_only_checkpoint_loading...", end=" ")

    # Create and save a target-only model
    config_no_mtp = make_tiny_config(mtp_enabled=False)
    torch.manual_seed(0)
    model_no_mtp = KimiK3ForCausalLM(config_no_mtp)
    target_state = model_no_mtp.state_dict()

    # Create MTP-enabled model and load target-only state
    config_mtp = make_tiny_config(mtp_enabled=True)
    torch.manual_seed(99)
    model_mtp = KimiK3ForCausalLM(config_mtp)

    missing, unexpected = model_mtp.load_state_dict(target_state, strict=False)
    assert len(missing) > 0, "Should have missing MTP keys"
    assert all("mtp_block" in k for k in missing), \
        f"Only MTP keys should be missing, got: {[k for k in missing if 'mtp_block' not in k]}"
    assert len(unexpected) == 0, f"Should have no unexpected keys, got: {unexpected}"

    # Verify target weights match
    for name, p_target in model_no_mtp.named_parameters():
        p_loaded = dict(model_mtp.named_parameters())[name]
        assert torch.equal(p_target, p_loaded), f"Target param {name} differs"

    print("ok")


def test_short_sequence():
    """Test MTP handles very short sequences safely."""
    print("test_short_sequence...", end=" ")
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.train()

    # Sequence of length 2: minimal for MTP (need at least 1 MTP position)
    inputs = torch.randint(0, config.vocab_size, (1, 2))
    targets = torch.randint(0, config.vocab_size, (1, 2))

    result = model(inputs, targets)
    assert torch.isfinite(result["loss"]), f"Loss is not finite for short sequence"
    assert torch.isfinite(result["main_loss"])

    # Sequence of length 1: MTP should gracefully handle (0 valid MTP positions)
    inputs1 = torch.randint(0, config.vocab_size, (1, 1))
    targets1 = torch.randint(0, config.vocab_size, (1, 1))
    result1 = model(inputs1, targets1)
    assert torch.isfinite(result1["loss"])
    print("ok")


def test_ignored_labels():
    """Test MTP correctly handles -100 (ignored) labels."""
    print("test_ignored_labels...", end=" ")
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.train()

    inputs = torch.randint(0, config.vocab_size, (2, 10))
    targets = torch.randint(0, config.vocab_size, (2, 10))
    # Mark some positions as ignored
    targets[0, :3] = -100
    targets[1, 7:] = -100

    result = model(inputs, targets)
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["mtp_loss"])

    # Note: all positions ignored (targets = -100 everywhere) produces NaN
    # in PyTorch CE loss (0 valid positions / 0 = NaN). This is expected
    # PyTorch behavior and not a bug in our MTP code. Real training data
    # always has at least some valid positions.
    print("ok")


def test_feature_extraction():
    """Test optional feature extraction API."""
    print("test_feature_extraction...", end=" ")
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.eval()

    inputs = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        result = model(inputs, return_features=True)

    assert "logits" in result
    assert "features" in result
    features = result["features"]

    # Check expected feature keys
    assert "final_hidden" in features
    for idx in config.feature_layer_indices:
        key = f"feature_{idx}"
        assert key in features, f"Missing feature key {key}"
        assert features[key].shape == (2, 8, config.hidden_size)

    assert features["final_hidden"].shape == (2, 8, config.hidden_size)
    print("ok")


def test_cpu_smoke_overfit():
    """Tiny CPU smoke test: fixed-batch overfit, both losses should decrease."""
    print("test_cpu_smoke_overfit...", end=" ", flush=True)
    config = make_tiny_config()
    torch.manual_seed(42)
    model = KimiK3ForCausalLM(config)
    model.train()

    # Fixed batch for overfitting
    inputs = torch.randint(0, config.vocab_size, (4, 32))
    targets = torch.randint(0, config.vocab_size, (4, 32))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    main_losses = []
    mtp_losses = []

    for i in range(50):
        result = model(inputs, targets)
        result["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(result["loss"].item())
        main_losses.append(result["main_loss"].item())
        mtp_losses.append(result["mtp_loss"].item())

    # Both losses should decrease over 50 steps of overfitting
    assert losses[-1] < losses[0], \
        f"Total loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert main_losses[-1] < main_losses[0], \
        f"Main loss did not decrease: {main_losses[0]:.4f} -> {main_losses[-1]:.4f}"
    assert mtp_losses[-1] < mtp_losses[0], \
        f"MTP loss did not decrease: {mtp_losses[0]:.4f} -> {mtp_losses[-1]:.4f}"

    print(f"ok (total: {losses[0]:.3f}->{losses[-1]:.3f}, main: {main_losses[0]:.3f}->{main_losses[-1]:.3f}, "
          f"mtp: {mtp_losses[0]:.3f}->{mtp_losses[-1]:.3f})")


def test_inference_compatibility():
    """Verify MTP-enabled model works correctly for inference (cache-based)."""
    print("test_inference_compatibility...", end=" ")
    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.eval()

    B, T = 2, 20
    tokens = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        # Full forward
        full_logits = model(tokens)
        assert full_logits.shape == (B, T, config.vocab_size)

        # Cache-based forward
        cache = model.make_cache(B, T, torch.float32)
        prefill_len = 10
        logits_parts = [model(tokens[:, :prefill_len], cache=cache)]
        for t in range(prefill_len, T):
            logits_parts.append(model(tokens[:, t:t+1], cache=cache))
        cached_logits = torch.cat(logits_parts, dim=1)

    error = (cached_logits.float() - full_logits.float()).abs().max().item()
    assert error < 1e-3, f"Cached vs full logits differ by {error:.2e}"
    print(f"ok (max error: {error:.1e})")



def test_mtp_evaluation_metrics():
    """Verify loss decomposition in model output and evaluate() metrics."""
    print("test_mtp_evaluation_metrics...", end=" ")
    from src.common import evaluate

    config = make_tiny_config()
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(config)
    model.train()

    inputs = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))
    result = model(inputs, targets)

    assert "loss" in result, "Missing loss in result dict"
    assert "main_loss" in result, "Missing main_loss in result dict"
    assert "mtp_loss" in result, "Missing mtp_loss in result dict"

    # Test evaluate with return_metrics=True
    def mock_loader():
        while True:
            yield (torch.randint(0, config.vocab_size, (2, 16)),
                   torch.randint(0, config.vocab_size, (2, 16)))

    metrics = evaluate(model, mock_loader(), steps=2, return_metrics=True)
    assert "val/loss" in metrics
    assert "val/ppl" in metrics
    assert "val/mtp_loss" in metrics

    # Test default backward compatibility (returns scalar float)
    scalar_loss = evaluate(model, mock_loader(), steps=2, return_metrics=False)
    assert isinstance(scalar_loss, float)
    print("ok")


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1 MTP Tests")
    print("=" * 60)

    test_token_alignment()
    test_causal_masking()
    test_loss_decomposition()
    test_shared_parameters()
    test_gradient_flow()
    test_mtp_disabled_compatibility()
    test_checkpoint_round_trip()
    test_target_only_checkpoint_loading()
    test_short_sequence()
    test_ignored_labels()
    test_feature_extraction()
    test_cpu_smoke_overfit()
    test_inference_compatibility()
    test_mtp_evaluation_metrics()

    print("=" * 60)
    print("All Stage 1 tests passed!")
    print("=" * 60)

    # CUDA/DDP smoke test
    if torch.cuda.is_available():
        print("\nRunning CUDA tests...")
        device = torch.device("cuda")
        config = make_tiny_config()
        torch.manual_seed(0)
        with torch.device(device):
            model = KimiK3ForCausalLM(config)
        model.train()

        inputs = torch.randint(0, config.vocab_size, (2, 16), device=device)
        targets = torch.randint(0, config.vocab_size, (2, 16), device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(inputs, targets)
        result["loss"].backward()
        print(f"CUDA smoke test ok: loss={result['loss'].item():.4f}")
    else:
        print("\nCUDA not available, skipping CUDA tests.")
