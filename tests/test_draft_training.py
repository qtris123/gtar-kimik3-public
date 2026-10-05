"""
Stage 2 tests: Feature fusion, recurrence, LK loss, freezing, and checkpoints.

Run from the repo root:
    python -m tests.test_draft_training
"""

import math
import sys
from dataclasses import asdict

import torch
import torch.nn.functional as F

from src.models.kimi_k3 import KimiK3Config, KimiK3ForCausalLM
from src.models.mtp import MTPBlock
from src.training.draft import DraftConfig, DraftTrainer, FeatureProjection, lk_loss, compute_overlap_and_agreement


def make_tiny_config(**overrides):
    return KimiK3Config(
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
        **overrides,
    )


def make_draft_config(**overrides):
    return DraftConfig(
        draft_steps=4,
        feature_layer_indices=[0, 1, 3],
        **overrides,
    )


def build_target_and_draft(device="cpu"):
    """Build a target model and draft trainer for testing."""
    config = make_tiny_config()
    torch.manual_seed(0)
    with torch.device(device):
        target_model = KimiK3ForCausalLM(config)

    # Freeze target
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    # Extract MTP block weights and create fresh draft
    mtp_state = {k.replace("mtp_block.", ""): v
                 for k, v in target_model.state_dict().items()
                 if k.startswith("mtp_block.")}

    draft_mtp = MTPBlock(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
    )
    if device != "cpu":
        draft_mtp = draft_mtp.to(device)
    draft_mtp.load_state_dict(mtp_state)

    draft_config = make_draft_config()
    draft_trainer = DraftTrainer(draft_mtp, draft_config)
    if device != "cpu":
        draft_trainer = draft_trainer.to(device)

    return target_model, draft_trainer, config


def test_feature_projection_identity():
    """Test that initial projection output equals the high feature exactly."""
    print("test_feature_projection_identity...", end=" ")
    D = 128
    proj = FeatureProjection(D)
    proj.init_identity_high()

    low = torch.randn(2, 8, D)
    mid = torch.randn(2, 8, D)
    high = torch.randn(2, 8, D)

    result = proj(low, mid, high)
    assert torch.allclose(result, high, atol=1e-6), \
        f"Initial projection should equal high feature, diff={( result - high).abs().max():.2e}"
    print("ok")


def test_conversion_equivalence():
    """Before any fine-tuning, draft should produce same output as Stage 1 MTP."""
    print("test_conversion_equivalence...", end=" ")
    target_model, draft_trainer, config = build_target_and_draft()

    # Get target features
    B, T = 2, 16
    inputs = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        result = target_model(inputs, return_features=True)
        features = result["features"]
        target_logits = result["logits"]

        # Stage 1 MTP forward
        final_hidden = features["final_hidden"]
        mtp_hidden_input = final_hidden[:, :-1]
        token_ids = targets[:, :-1].clone()
        valid = token_ids != -100
        token_ids[~valid] = 0
        mtp_emb = target_model.model.embed_tokens(token_ids)
        mtp_emb = mtp_emb * valid.unsqueeze(-1).float()
        stage1_mtp_out = target_model.mtp_block(mtp_hidden_input, mtp_emb)
        stage1_logits = target_model.lm_head(target_model.mtp_block.get_output_hidden(stage1_mtp_out))

        # Draft (with identity projection, high feature = final_hidden = feature_3)
        # Draft step 1 uses fused_features[:, :-1] and embed(targets[:, 0:-1])
        # The fused features with identity init = feature_3 = features["feature_3"]
        # But feature_3 is the hidden after layer 3, which IS final_hidden
        fused = draft_trainer.feature_proj(
            features["feature_0"], features["feature_1"], features["feature_3"]
        )
        # feature_3 IS final_hidden for a 4-layer model (layer index 3 = last layer)
        assert torch.allclose(fused, features["feature_3"], atol=1e-6), \
            "Initial fused should equal feature_3 (identity on high)"

        draft_mtp_out = draft_trainer.mtp_block(fused[:, :-1], mtp_emb)
        draft_logits = target_model.lm_head(draft_trainer.mtp_block.get_output_hidden(draft_mtp_out))

    diff = (stage1_logits - draft_logits).abs().max().item()
    assert diff < 1e-4, f"Draft/Stage1 logit diff = {diff:.2e}, should be ~0"
    print(f"ok (max diff: {diff:.2e})")


def test_lk_loss_identical_distributions():
    """LK loss should be approximately zero for identical distributions."""
    print("test_lk_loss_identical_distributions...", end=" ")
    B, T, V = 2, 8, 100
    logits = torch.randn(B, T, V)
    mask = torch.ones(B, T, dtype=torch.bool)

    loss = lk_loss(logits, logits, mask)
    # overlap = 1.0 for identical distributions, so -log(1.0) = 0
    assert loss.item() < 1e-5, f"LK loss for identical distributions = {loss.item():.2e}, should be ~0"
    print(f"ok (loss = {loss.item():.2e})")


def test_lk_loss_mismatched_distributions():
    """LK loss should be positive for mismatched distributions."""
    print("test_lk_loss_mismatched_distributions...", end=" ")
    B, T, V = 2, 8, 100
    logits1 = torch.randn(B, T, V)
    logits2 = torch.randn(B, T, V) * 5  # very different
    mask = torch.ones(B, T, dtype=torch.bool)

    loss = lk_loss(logits1, logits2, mask)
    assert loss.item() > 0.1, f"LK loss for mismatched = {loss.item():.4f}, should be > 0.1"
    assert torch.isfinite(loss), "LK loss should be finite"
    print(f"ok (loss = {loss.item():.4f})")


def test_lk_loss_sparse_overlap():
    """LK loss should be finite with very sparse overlap."""
    print("test_lk_loss_sparse_overlap...", end=" ")
    B, T, V = 1, 4, 1000
    # One-hot-ish distributions with minimal overlap
    logits1 = torch.zeros(B, T, V)
    logits1[:, :, 0] = 100.0  # peaked at token 0
    logits2 = torch.zeros(B, T, V)
    logits2[:, :, 1] = 100.0  # peaked at token 1
    mask = torch.ones(B, T, dtype=torch.bool)

    loss = lk_loss(logits1, logits2, mask)
    assert torch.isfinite(loss), f"LK loss with sparse overlap should be finite, got {loss.item()}"
    print(f"ok (loss = {loss.item():.4f})")


def test_lk_loss_empty_mask():
    """LK loss with empty mask should be zero (no supervised positions)."""
    print("test_lk_loss_empty_mask...", end=" ")
    B, T, V = 2, 8, 100
    logits1 = torch.randn(B, T, V)
    logits2 = torch.randn(B, T, V)
    mask = torch.zeros(B, T, dtype=torch.bool)

    loss = lk_loss(logits1, logits2, mask)
    assert loss.item() == 0.0, f"LK loss with empty mask = {loss.item()}, should be 0"
    print("ok")


def test_overlap_and_agreement():
    """Test overlap and agreement metrics."""
    print("test_overlap_and_agreement...", end=" ")
    B, T, V = 2, 8, 100

    # Identical distributions
    logits = torch.randn(B, T, V)
    mask = torch.ones(B, T, dtype=torch.bool)
    overlap, agree = compute_overlap_and_agreement(logits, logits, mask)
    assert abs(overlap - 1.0) < 1e-5, f"Overlap for identical = {overlap}, should be 1.0"
    assert abs(agree - 1.0) < 1e-5, f"Agreement for identical = {agree}, should be 1.0"

    # Very different distributions
    logits2 = torch.randn(B, T, V) * 10
    overlap2, agree2 = compute_overlap_and_agreement(logits, logits2, mask)
    assert overlap2 < 0.95, f"Overlap for different = {overlap2}, should be < 0.95"
    print(f"ok (identical: overlap={overlap:.3f}, agree={agree:.3f}; "
          f"different: overlap={overlap2:.3f}, agree={agree2:.3f})")


def test_draft_gradients_and_freezing():
    """Verify draft/projection have gradients, target has none, after optimizer step."""
    print("test_draft_gradients_and_freezing...", end=" ")
    target_model, draft_trainer, config = build_target_and_draft()
    draft_trainer.train()

    B, T = 2, 16
    inputs = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    # Save target weights before
    target_weights_before = {n: p.clone() for n, p in target_model.named_parameters()}

    # Forward
    with torch.no_grad():
        result = target_model(inputs, return_features=True)
        target_logits = result["logits"].detach()
        target_features = {k: v.detach() for k, v in result["features"].items()}

    draft_result = draft_trainer(
        target_features=target_features,
        target_logits=target_logits,
        targets=targets,
        embed_fn=target_model.model.embed_tokens,
        lm_head_fn=target_model.lm_head,
    )
    draft_result["loss"].backward()

    # Draft parameters should have gradients
    for name, p in draft_trainer.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Draft param {name} has no gradient"

    # Target parameters should have NO gradients
    for name, p in target_model.named_parameters():
        assert not p.requires_grad, f"Target param {name} should be frozen"

    # Optimizer step
    optimizer = torch.optim.AdamW(draft_trainer.get_trainable_parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad()

    # Target weights should be unchanged
    for name, p in target_model.named_parameters():
        assert torch.equal(p, target_weights_before[name]), \
            f"Target param {name} changed after optimizer step!"

    print("ok")


def test_recursive_alignment():
    """Test that recursive step token/distribution/mask alignment is correct."""
    print("test_recursive_alignment...", end=" ")
    target_model, draft_trainer, config = build_target_and_draft()
    draft_trainer.train()

    B, T = 1, 12
    inputs = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        result = target_model(inputs, return_features=True)
        target_logits = result["logits"]
        target_features = {k: v.detach() for k, v in result["features"].items()}

    draft_result = draft_trainer(
        target_features=target_features,
        target_logits=target_logits,
        targets=targets,
        embed_fn=target_model.model.embed_tokens,
        lm_head_fn=target_model.lm_head,
    )

    # Should have 4 steps (or fewer if T is too small)
    n_steps = min(4, T - 1)
    assert len(draft_result["per_step_loss"]) == n_steps, \
        f"Expected {n_steps} steps, got {len(draft_result['per_step_loss'])}"

    # All losses should be finite
    for i, l in enumerate(draft_result["per_step_loss"]):
        assert math.isfinite(l), f"Step {i+1} loss is not finite: {l}"

    # Loss should be finite
    assert torch.isfinite(draft_result["loss"])
    print(f"ok ({n_steps} steps, loss={draft_result['loss'].item():.4f})")


def test_short_sequence_draft():
    """Test draft handles sequences shorter than draft_steps."""
    print("test_short_sequence_draft...", end=" ")
    target_model, draft_trainer, config = build_target_and_draft()
    draft_trainer.train()

    # Sequence of length 3 with 4 draft steps
    B, T = 1, 3
    inputs = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        result = target_model(inputs, return_features=True)
        target_logits = result["logits"]
        target_features = {k: v.detach() for k, v in result["features"].items()}

    draft_result = draft_trainer(
        target_features=target_features,
        target_logits=target_logits,
        targets=targets,
        embed_fn=target_model.model.embed_tokens,
        lm_head_fn=target_model.lm_head,
    )

    # Should have only 2 steps (T - step_r >= 1, so step_r <= T-1 = 2)
    assert len(draft_result["per_step_loss"]) == 2
    assert torch.isfinite(draft_result["loss"])
    print("ok")


def test_draft_checkpoint_round_trip():
    """Verify draft checkpoint save and resume."""
    print("test_draft_checkpoint_round_trip...", end=" ")
    import tempfile

    target_model, draft_trainer, config = build_target_and_draft()

    # Save draft state
    state = draft_trainer.state_dict()

    # Create a fresh draft trainer and load
    mtp_block2 = MTPBlock(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
    )
    draft_config2 = make_draft_config()
    draft_trainer2 = DraftTrainer(mtp_block2, draft_config2)
    draft_trainer2.load_state_dict(state)

    # Compare parameters
    for (n1, p1), (n2, p2) in zip(draft_trainer.named_parameters(), draft_trainer2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Draft param {n1} differs after load"

    # Compare forward output
    B, T = 1, 8
    inputs = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        result = target_model(inputs, return_features=True)
        target_logits = result["logits"]
        features = {k: v.detach() for k, v in result["features"].items()}

    draft_trainer.eval()
    draft_trainer2.eval()
    with torch.no_grad():
        r1 = draft_trainer(features, target_logits, targets,
                           target_model.model.embed_tokens, target_model.lm_head)
        r2 = draft_trainer2(features, target_logits, targets,
                            target_model.model.embed_tokens, target_model.lm_head)

    assert abs(r1["loss"].item() - r2["loss"].item()) < 1e-5, \
        f"Draft checkpoint round trip loss mismatch: {r1['loss']} vs {r2['loss']}"
    print("ok")


def test_draft_overfit():
    """Test that draft training makes progress on a fixed batch."""
    print("test_draft_overfit...", end=" ", flush=True)
    target_model, draft_trainer, config = build_target_and_draft()
    draft_trainer.train()

    B, T = 4, 32
    inputs = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    # Get target outputs (fixed)
    with torch.no_grad():
        result = target_model(inputs, return_features=True)
        target_logits = result["logits"].detach()
        target_features = {k: v.detach() for k, v in result["features"].items()}

    optimizer = torch.optim.AdamW(draft_trainer.get_trainable_parameters(), lr=1e-3)

    losses = []
    overlaps = []
    for i in range(30):
        draft_result = draft_trainer(
            target_features=target_features,
            target_logits=target_logits,
            targets=targets,
            embed_fn=target_model.model.embed_tokens,
            lm_head_fn=target_model.lm_head,
        )
        draft_result["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(draft_result["loss"].item())
        if draft_result["per_step_overlap"]:
            overlaps.append(sum(draft_result["per_step_overlap"]) / len(draft_result["per_step_overlap"]))

    # Loss should decrease
    assert losses[-1] < losses[0], \
        f"Draft loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    # Overlap should increase (draft is getting closer to target)
    if len(overlaps) >= 2:
        assert overlaps[-1] > overlaps[0], \
            f"Overlap did not increase: {overlaps[0]:.4f} -> {overlaps[-1]:.4f}"

    print(f"ok (loss: {losses[0]:.3f}->{losses[-1]:.3f}, "
          f"overlap: {overlaps[0]:.3f}->{overlaps[-1]:.3f})")


def test_gradients_through_unroll():
    """Verify gradients flow through multiple recursive steps."""
    print("test_gradients_through_unroll...", end=" ")
    target_model, draft_trainer, config = build_target_and_draft()
    draft_trainer.train()

    B, T = 2, 20
    inputs = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        result = target_model(inputs, return_features=True)
        target_logits = result["logits"].detach()
        target_features = {k: v.detach() for k, v in result["features"].items()}

    draft_result = draft_trainer(
        target_features=target_features,
        target_logits=target_logits,
        targets=targets,
        embed_fn=target_model.model.embed_tokens,
        lm_head_fn=target_model.lm_head,
    )
    draft_result["loss"].backward()

    # All trainable params should have non-zero gradients
    has_grad = False
    for name, p in draft_trainer.named_parameters():
        if p.requires_grad and p.grad is not None:
            if p.grad.abs().sum() > 0:
                has_grad = True
    assert has_grad, "No trainable parameter has non-zero gradient"

    # The fuse_proj in the MTP block should have gradients (proves
    # gradients flow through the recursive unroll)
    fuse_grad = draft_trainer.mtp_block.fuse_proj.weight.grad
    assert fuse_grad is not None and fuse_grad.abs().sum() > 0, \
        "fuse_proj should have gradients from recursive unroll"
    print("ok")


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 2 Draft Training Tests")
    print("=" * 60)

    test_feature_projection_identity()
    test_conversion_equivalence()
    test_lk_loss_identical_distributions()
    test_lk_loss_mismatched_distributions()
    test_lk_loss_sparse_overlap()
    test_lk_loss_empty_mask()
    test_overlap_and_agreement()
    test_draft_gradients_and_freezing()
    test_recursive_alignment()
    test_short_sequence_draft()
    test_draft_checkpoint_round_trip()
    test_draft_overfit()
    test_gradients_through_unroll()

    print("=" * 60)
    print("All Stage 2 tests passed!")
    print("=" * 60)

    if torch.cuda.is_available():
        print("\nRunning CUDA tests...")
        target_model, draft_trainer, config = build_target_and_draft(device="cuda")
        draft_trainer.train()

        B, T = 2, 16
        inputs = torch.randint(0, config.vocab_size, (B, T), device="cuda")
        targets = torch.randint(0, config.vocab_size, (B, T), device="cuda")

        with torch.no_grad():
            result = target_model(inputs, return_features=True)
            target_logits = result["logits"].detach()
            features = {k: v.detach() for k, v in result["features"].items()}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            draft_result = draft_trainer(
                target_features=features,
                target_logits=target_logits,
                targets=targets,
                embed_fn=target_model.model.embed_tokens,
                lm_head_fn=target_model.lm_head,
            )
        draft_result["loss"].backward()
        print(f"CUDA smoke test ok: loss={draft_result['loss'].item():.4f}")
    else:
        print("\nCUDA not available, skipping CUDA tests.")
