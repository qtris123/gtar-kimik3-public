# Kimi K3 Two-Stage MTP Implementation Report

## Summary

This report documents the completed implementation and verification of the two-stage Multi-Token Prediction (MTP) and speculative drafting pipeline for Kimi K3 in `gstar-kimi3` on branch `k3-mtp`.

---

## 1. Architecture & Design

### Stage 1: Joint MTP Pretraining
- **Block Architecture (`src/models/mtp.py`)**:
  - Independent RMSNorm for previous hidden state $h_t$ and next-token embedding $e(x_{t+1})$.
  - Feature concatenation and bias-free projection $W_{\text{fuse}}: \mathbb{R}^{2d} \to \mathbb{R}^d$.
  - Shallow causal self-attention layer (1 layer with residual connection and pre-norm).
  - SwiGLU feed-forward layer (with residual connection and pre-norm).
  - Dedicated output RMSNorm owned by the MTP block.
  - Shares token embedding table and final LM head with the target model.
- **Token Alignment & Shifted Loss**:
  - Main causal input: $x_{1 \dots T-1}$, targets: $x_{2 \dots T}$.
  - Main model outputs hidden states $h_{1 \dots T-1}$.
  - MTP inputs: previous hidden $h_{1 \dots T-2}$, embedding of next ground-truth token $e(x_{2 \dots T-1})$.
  - MTP targets: $x_{3 \dots T}$.
  - Combined loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + \lambda_{\text{mtp}} \cdot \mathcal{L}_{\text{mtp}}$.
  - Validation metrics: validation loss and perplexity ($\exp(\mathcal{L}_{\text{main}})$) are evaluated strictly on the main next-token objective to prevent auxiliary objective contamination.
  - Parameter and FLOP accounting in `scripts/train.py` dynamically includes the MTP block overhead when enabled.

### Stage 2: Drafter Fine-Tuning against Frozen Target
- **Feature Fusion (`src/training/draft.py:FeatureProjection`)**:
  - Extracts 3 representative target hidden states: early, middle, and final layers (e.g. layers `[0, 1, 3]` in 4-layer tiny model, `[4, 13, 27]` in 28-layer 0.6B model).
  - Low/mid/high features are concatenated along the hidden dimension ($\mathbb{R}^{3d}$).
  - Projected via bias-free $W_{\text{proj}}: \mathbb{R}^{3d} \to \mathbb{R}^d$, initialized as $[0, 0, I]$.
  - Exact equivalence property: before any training, $W_{\text{proj}}([h_{\text{low}}, h_{\text{mid}}, h_{\text{high}}]) = h_{\text{high}}$ exactly, guaranteeing identical initial representations to Stage 1.
- **Recursive Unrolling & Teacher-Forced Trajectory**:
  - Recurrent hidden state feedback: the draft hidden state output from step $r-1$ feeds into the MTP block at step $r$.
  - Shifted ground-truth token embeddings: step $r$ consumes $e(x_{t+r})$ and supervises distribution over $x_{t+r+1}$.
  - Pure LK (acceptance-overlap) loss at temperature 1:
    $$p = \text{softmax}(z_{\text{target}}), \quad q = \text{softmax}(z_{\text{draft}})$$
    $$\text{overlap} = \sum_{v \in V} \min(p_v, q_v)$$
    $$\mathcal{L}_{\text{LK}} = -\mathbb{E}\left[\log(\max(\text{overlap}, \epsilon))\right]$$
  - Multi-step combination: supports uniform averaging or exponential decay $\sum_r \gamma^r \mathcal{L}_r$.
  - Target model parameters are frozen; only MTP block and feature projection are optimized.

---

## 2. Changed & Created Files

| File | Type | Description |
|---|---|---|
| `src/models/mtp.py` | New | `MTPBlock` with fusion, shallow causal attention, SwiGLU, and output norm. |
| `src/models/kimi_k3.py` | Modified | Added `mtp_block` initialization, intermediate feature extraction API, joint MTP loss computation, FLOP accounting. |
| `src/models/__init__.py` | Modified | Exported `MTPBlock`. |
| `src/training/__init__.py` | New | Package init exporting `DraftTrainer`, `DraftConfig`, `lk_loss`. |
| `src/training/draft.py` | New | `FeatureProjection`, pure LK loss, overlap & agreement metrics, recursive unroll module `DraftTrainer`. |
| `src/common.py` | Modified | Updated `evaluate()` to isolate `main_loss`, updated checkpoint loader for flexible/strict MTP key initialization. |
| `scripts/train.py` | Modified | Added MTP configuration, `--strict-resume` support, separate logging for `main_loss`, `mtp_loss`, and MTP gradient norms. |
| `scripts/train_draft.py` | New | Stage 2 training script: loads frozen target, initializes draft, trains with LK loss, saves draft checkpoint. |
| `configs/kimi_k3_tiny_mtp.json` | New | Tiny 4-layer config for testing both Stage 1 and Stage 2 pipelines. |
| `configs/kimi_k3_0.6B_mtp.json` | New | 0.6B production config with MTP enabled (feature layers `[4, 13, 27]`). |
| `tests/test_mtp.py` | New | Comprehensive Stage 1 test suite (14 unit/regression/overfit tests). |
| `tests/test_draft_training.py` | New | Comprehensive Stage 2 test suite (13 unit/regression/overfit tests). |
| `tests/test_speculative.py` | New | Speculative decoding requirements & test specification stub. |

---

## 3. Verification & Test Results

### Unit Tests
Executed via `/home/gpuuser/.venv/bin/python`:
1. `tests/test_mtp.py`:
   - `test_token_alignment`: **PASSED**
   - `test_causal_masking`: **PASSED**
   - `test_loss_decomposition`: **PASSED**
   - `test_shared_parameters`: **PASSED**
   - `test_gradient_flow`: **PASSED**
   - `test_mtp_disabled_compatibility`: **PASSED**
   - `test_checkpoint_round_trip`: **PASSED**
   - `test_target_only_checkpoint_loading`: **PASSED**
   - `test_short_sequence`: **PASSED**
   - `test_ignored_labels`: **PASSED**
   - `test_feature_extraction`: **PASSED**
   - `test_cpu_smoke_overfit`: **PASSED** (loss: 8.127 -> 0.249)
   - `test_inference_compatibility`: **PASSED** (max diff: 4.8e-07)
   - `test_estimate_flops_with_mtp`: **PASSED**
2. `tests/test_draft_training.py`:
   - `test_feature_projection_identity`: **PASSED**
   - `test_conversion_equivalence`: **PASSED** (max diff: 0.00e+00)
   - `test_lk_loss_identical_distributions`: **PASSED** (loss = 2.22e-15)
   - `test_lk_loss_mismatched_distributions`: **PASSED** (loss = 2.9312)
   - `test_lk_loss_sparse_overlap`: **PASSED** (loss = 18.4207)
   - `test_lk_loss_empty_mask`: **PASSED**
   - `test_overlap_and_agreement`: **PASSED**
   - `test_draft_gradients_and_freezing`: **PASSED** (target gradients strictly 0)
   - `test_recursive_alignment`: **PASSED** (4 steps, loss = 0.1373)
   - `test_short_sequence_draft`: **PASSED**
   - `test_draft_checkpoint_round_trip`: **PASSED**
   - `test_draft_overfit`: **PASSED** (loss: 0.136 -> 0.033, overlap: 0.873 -> 0.968)
   - `test_gradients_through_unroll`: **PASSED**

### End-to-End Smoke Training Runs
1. **Stage 1 Training (`scripts/train.py`)**:
   - Trained on `data/fineweb_edu` with `configs/kimi_k3_tiny_mtp.json`.
   - Loss decreased from 15.4711 (main: 11.8893, mtp: 11.9391) to 15.0079 (main: 11.5115, mtp: 11.6547).
   - Saved `out/test_mtp_tiny/ckpt_000010.pt`.
   - Verified resume from checkpoint at step 10 -> executed steps 11 and 12 smoothly.
2. **Stage 2 Drafter Training (`scripts/train_draft.py`)**:
   - Loaded frozen target and initialized drafter from `out/test_mtp_tiny/ckpt_000010.pt`.
   - Trained 4-step recursive unrolling on `data/fineweb_edu`.
   - Validation loss dropped from 0.0915 to 0.0484.
   - Mean overlap improved from 0.9126 to 0.9528.
   - Top-1 target agreement improved from 0.1889 to 0.6674.
   - Saved `out/test_draft_tiny/draft_000010.pt` (3.2 MB lightweight draft checkpoint).
   - Verified resume from checkpoint at step 10 -> executed steps 11 and 12 with further loss drop to 0.0474.
3. **Inference Compatibility (`scripts/generate.py`)**:
   - Ran text generation from `ckpt_000010.pt` via `src/engine.py`.
   - Successfully loaded checkpoint and generated tokens without errors.

---

## 4. Speculative Inference (Follow-On Work)

As specified in the milestone plan, speculative decoding is deferred to the subsequent inference milestone. Requirements and invariants are documented in `tests/test_speculative.py`:
- Target verification engine with speculative draft proposals at batch size 1.
- Exact token-for-token equality with target-only greedy generation.
- Transactional KV cache and KDA recurrent state cloning/rollback upon token rejection.
