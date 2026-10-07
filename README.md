# Kimi-K3: Multi-Token Prediction (MTP) & Speculative Decoding

A concise, educational PyTorch implementation of **Multi-Token Prediction (MTP)** and **EAGLE-style speculative decoding** for the **Kimi-K3** hybrid architecture (KDA linear/recurrent attention + GQA windowed self-attention), alongside the standard Transformer baseline.

---

## Architecture & Lifecycle Overview

The codebase is structured around three consecutive stages:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Joint MTP Pretraining (scripts/train.py)                      │
│ - Shared embedding & lm_head between backbone and MTP block            │
│ - MTP block predicts token t+2 from backbone final hidden & token t+1  │
│ - Loss = L_main + λ * L_mtp (λ = 0.3)                                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Pretrained target + MTP weights
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Recursive Drafter Training (scripts/train_draft.py)           │
│ - Frozen target model produces low/mid/high representations            │
│ - Feature projection Linear(3d, d) initialized as [0, 0, I]            │
│ - Teacher-forced recursive unroll (K depths)                           │
│ - DraftTTTCache: depth-1 causal prefix + depth>=2 diagonal extension   │
│ - NVIDIA EAGLE-3 LK loss (acceptance-overlap alpha objective)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Drafter weights (FeatureProj + MTP)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Greedy Speculative Decoding (src/speculative.py)              │
│ - Exact token-for-token identity with greedy autoregressive generation │
│ - Target cache forking & verification                                  │
│ - Full acceptance (1 + K commit) with bonus token                      │
│ - Partial rejection: rollback shadow state + replay accepted prefix    │
│ - Recurrent KDA state maintained transactionally across spec rounds    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
├── configs/
│   ├── kimi_k3_0.6B_mtp.json      # 0.6B Kimi-K3 with MTP (Stage 1 target)
│   ├── kimi_k3_tiny_mtp.json      # Tiny fast CPU/testing config
│   ├── transformer_0.6B.json      # 0.6B Standard Transformer baseline
│   └── transformer_tiny.json      # Tiny Transformer baseline
├── scripts/
│   ├── train.py                   # Pretraining loop (normal or with MTP)
│   ├── train_draft.py             # Stage 2 recursive drafter training (LK loss)
│   ├── generate.py                # Autoregressive generation from checkpoint
│   ├── benchmark_speculative.py   # Benchmark speculative decoding speedup
│   └── prepare_data.py            # FineWeb-Edu tokenization pipeline
├── src/
│   ├── models/
│   │   ├── kimi_k3.py             # Kimi-K3 causal LM with optional MTP block
│   │   ├── mtp.py                 # Standalone MTP block
│   │   └── transformer.py         # Standard Transformer baseline
│   ├── training/
│   │   └── draft.py               # DraftTrainer (FeatureProjection, LK loss)
│   ├── speculative.py             # SpeculativeEngine (verify, rollback, replay)
│   ├── cache.py                   # Cache, KVCache, KDACache, DraftTTTCache
│   ├── layers/                    # Attention, KDA, RMSNorm, RoPE, SwiGLU
│   ├── dataset.py                 # Sharded token loader
│   ├── optim.py                   # AdamW optimizer & cosine LR schedule
│   └── common.py                  # Distributed setup, evaluation, checkpointing
├── tests/
│   ├── test_mtp.py                # Stage 1 tests (alignment, loss, gradient flow)
│   ├── test_draft_training.py     # Stage 2 tests (unroll, LK loss, TTT cache)
│   ├── test_speculative.py        # Stage 3 tests (exact token identity, rollback)
│   └── helpers.py                 # Test trajectory & validation references
└── tools/
    ├── upload_to_hf.py            # Hugging Face checkpoint upload tool
    └── upload_to_hf.sh            # Upload launcher script
```

---

## Recommended Code Reading Order

For studying this codebase, read the core implementation in this order:

1. **[`src/models/mtp.py`](file:///home/gpuuser/gstar-kimi3-public/src/models/mtp.py)**: The MTP block structure (RMSNorm on trunk input, projection, attention/FFN layer, output RMSNorm).
2. **[`src/models/kimi_k3.py`](file:///home/gpuuser/gstar-kimi3-public/src/models/kimi_k3.py)**: How the main model feeds trunk hidden states and ground-truth shifted tokens to the MTP block during Stage 1.
3. **[`src/training/draft.py`](file:///home/gpuuser/gstar-kimi3-public/src/training/draft.py)**: Stage 2 recursive unroll, `FeatureProjection([0, 0, I])`, and the LK acceptance-overlap loss.
4. **[`src/cache.py`](file:///home/gpuuser/gstar-kimi3-public/src/cache.py)**: `DraftTTTCache` (depth-1 causal prefix + diagonal extension) and transactional cache methods (`fork`, `commit`, `rollback`).
5. **[`src/speculative.py`](file:///home/gpuuser/gstar-kimi3-public/src/speculative.py)**: `SpeculativeEngine` implementing verify-then-accept greedy generation, full match commit `1+K`, and rollback + replay on rejection.

---

## Quickstart & Commands

### 1. Stage 1 Pretraining (Joint MTP)

```bash
# Kimi-K3 with MTP
torchrun --standalone --nproc_per_node=4 -m scripts.train \
    --config configs/kimi_k3_0.6B_mtp.json \
    --data data/fineweb_edu

# Baseline Transformer architecture
torchrun --standalone --nproc_per_node=4 -m scripts.train \
    --config configs/transformer_0.6B.json \
    --data data/fineweb_edu
```

### 2. Stage 2 Drafter Fine-Tuning

```bash
torchrun --standalone --nproc_per_node=4 -m scripts.train_draft \
    --config configs/kimi_k3_0.6B_mtp.json \
    --target-checkpoint out/kimi_k3_0.6B_mtp/ckpt_019073.pt \
    --data data/fineweb_edu \
    --draft-steps 4 \
    --feature-layers "0,4,27"
```

### 3. Speculative Generation & Benchmarking

```bash
# Speculative decoding benchmark
python -m scripts.benchmark_speculative \
    --config configs/kimi_k3_0.6B_mtp.json \
    --target-checkpoint out/kimi_k3_0.6B_mtp/ckpt_019073.pt \
    --draft-checkpoint out/kimi_k3_0.6B_mtp_draft/draft_final.pt \
    --draft-steps 4 \
    --max-new-tokens 128
```

---

## Running the Complete Test Suite

The test suite thoroughly verifies all correctness invariants across all 3 stages:

```bash
# Stage 1: MTP token alignment, causal masking, shared params, gradient flow (14 tests)
python -m tests.test_mtp

# Stage 2: Feature projection, LK loss, recursive unroll, TTT attention cache (18 tests)
python -m tests.test_draft_training

# Stage 3: Speculative token identity, rollback/replay, KDA cache state (21 tests)
python -m tests.test_speculative

# Inference cache test (Transformer & Kimi-K3)
python -m tests.test_inference
```
