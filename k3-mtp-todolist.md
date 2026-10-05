# Kimi MTP: two-stage training implementation brief

Target repository: https://github.com/newturing/gstar-kimi3 on the current k3-mtp branch

## Task for the implementation agent

Implement both training stages below, following the repository's existing conventions and coding style. Complete and validate Stage 1 before integrating Stage 2. Inspect the current checkout first. Adapt reference implementations rather than importing their full frameworks. Writing clean codes. Read only the targeted symbols listed under References: three core files, plus one attention helper if needed.

The result should be a miniature **K3-style training pipeline**: joint next-token/MTP pretraining, followed by independent drafter fine-tuning against a frozen target. The current target uses KDA interleaved with ordinary attention and SwiGLU; it does not implement every production K3 component. Reuse those local primitives and document the architectural differences.

| Stage | Trainable parameters | Objective | Entry point |
|---|---|---|---|
| 1: joint pretraining | Target + one MTP block | Main token CE + `mtp_loss_weight * mtp_CE` | `scripts/train.py` |
| 2: drafter fine-tuning | Pretrained MTP block + feature projection | LK acceptance-overlap loss against frozen target | `scripts/train_draft.py` |

A single MTP block can be reused recursively to propose multiple tokens. Stage 2 trains this recursive hidden-state use. The target, not the MTP block, eventually verifies and accepts proposals.

## File layout

Preserve existing files and add package exports/`__init__.py` files as needed.

| Path | Responsibility |
|---|---|
| `src/models/kimi_k3.py` | Optional intermediate features, MTP integration and joint loss |
| `src/models/mtp.py` | One MTP/draft block and low/mid/high feature projection |
| `src/training/draft.py` | Recursive training unroll, alignment, masks, LK loss and metrics |
| `src/engine.py`, `src/cache.py` | Preserve ordinary inference; future speculative integration |
| `scripts/train.py` | Stage 1, including MTP config and separate loss logging |
| `scripts/train_draft.py` | Stage 2, target loading/freezing, training, evaluation and checkpoints |
| `scripts/generate.py` | Preserve current generation behavior |
| `tests/test_mtp.py` | Stage-1 alignment, loss, gradients and compatibility |
| `tests/test_draft_training.py` | Stage-2 fusion, recurrence, loss, freezing and checkpoints |
| `tests/test_speculative.py` | Future inference milestone; add substantive tests when implemented |

Also add tiny example configs and concise launch instructions. Speculative decoding is a later milestone, not required to finish these two training stages; do not create empty tests or claim it works.

## 1. Inspect the existing pipeline

- [x] Read `kimi_k3.py`, `scripts/train.py`, `src/dataset.py`, optimizer/checkpoint/evaluation helpers, attention and cache APIs, and existing tests.
- [x] Preserve the existing default forward behavior: inference returns logits; training with targets returns a scalar loss. Make richer outputs opt-in so other architectures and callers remain compatible.
- [x] Keep the current token loader, gradient accumulation, DDP, autocast, clipping, optimizer-step schedules, resume behavior and optional compilation working.

## 2. Stage 1: joint MTP pretraining

- [x] Add **one** MTP block in `src/models/mtp.py`. Separately normalize the prior hidden state and next-token embedding, concatenate them, project `2d -> d`, then run a shallow causal attention + dense MLP block. Use the target's embedding and vocabulary head; include any draft output normalization in the block's saved parameters. Inspect `MTPBlock`, `MTPModule`, and the alignment/loss methods in [R1].
- [x] Design its hidden-state input/output contract for repeated calls. Reuse local attention/MLP code; keep draft cache/state separate from target cache/state. Do not introduce multiple independent heads for this task.
- [x] Expose final hidden states and selected intermediate states through an optional model API. Avoid collecting every layer by default. Define clearly whether each feature is before or after normalization.
- [x] Add optional MTP enablement and `mtp_loss_weight` configuration. Compute main CE and shifted MTP CE in the same model forward so DDP tracks the trainable MTP parameters. Let MTP gradients reach the target in Stage 1.
- [x] Update the existing `train.py` and evaluation helpers to log `main_loss`, `mtp_loss`, and total loss separately. Target validation perplexity is `exp(main_loss)`; exponentiating the combined loss mixes two tasks. Include MTP overhead in parameter/FLOP reporting when enabled.
- [x] Save MTP configuration and weights with Stage-1 checkpoints. Make loading an older target-only checkpoint explicit: preserve its target weights and initialize only missing MTP parameters; do not silently ignore unrelated missing keys.

**Alignment with the existing loader:** for inputs of length `T`, `targets[:, t]` is the token following `inputs[:, t]`. Thus a convenient valid MTP slice is:

```python
mtp_hidden = final_hidden[:, :-1]
mtp_embeddings = embed_tokens(targets[:, :-1])
mtp_labels = targets[:, 1:]
```

For tokens `[10, 20, 30, 40, 50, 60]`, the main inputs are `[10, 20, 30, 40, 50]`, main labels `[20, 30, 40, 50, 60]`, MTP embedding inputs `[20, 30, 40, 50]`, and MTP labels `[30, 40, 50, 60]`. Apply masks before embedding ignored labels; handle short sequences and no valid positions safely. If document boundaries are represented, prevent supervision and attention from crossing them.

**Stage-1 checks**

- [x] Test that exact token alignment, causal masking, scalar loss decomposition, shared parameter identity, and gradient flow are correct.
- [x] Verify MTP-disabled training/inference compatibility and checkpoint round trips.
- [x] Run a tiny CPU smoke test and fixed-batch overfit test; both losses should decrease. Run a CUDA/DDP smoke test if available and report unavailable checks.

**CUDA testing on our 4 H100s:** you may stop the Kimi training job without asking again. First record its launch command and verify a completed checkpoint. **Ctrl+C does not save; unsaved steps must be repeated.**

After confirming `0:kimi` targets the training pane, run on the H100 host:

```bash
tmux send-keys -t '0:kimi' C-c
nvidia-smi
```

If the target is ambiguous, use `tmux list-panes -a` to identify its pane ID and substitute that ID. Wait until training workers exit and release GPU memory. After testing, resume the original command with `--resume /path/to/checkpoint.pt` using compatible code/config. Keep test outputs separate and leave unrelated processes running.

## 3. Stage 2: convert MTP into a drafter

- [x] Load the trained Stage-1 target and MTP weights. Freeze the target, shared embedding and shared LM head; set the target to evaluation mode and extract its features/logits under `no_grad`. Train a separate draft instance initialized from MTP, plus the new feature projection. Use [R2] for the draft structure and [R3] for the training loop.
- [x] Select configurable early/middle/final target features and concatenate them in that order. For the miniature model, use representative depths rather than assuming production AttnRes block numbers.
- [x] Add a bias-free `Linear(3d, d)` projection, initialized with zero low/middle blocks and an identity high block. The initial projected feature must equal the **exact same high feature** used in Stage 1, including normalization. Apply this initialization after generic weight initialization and preserve it on resume.
- [x] Preserve the Stage-1 MTP architecture and normalization during conversion. [R2] feeds concatenated features directly into attention, unlike our `2d -> d` projection; adapt its ideas without replacing the pretrained block with incompatible weights.
- [x] Implement a configurable recursive unroll in `src/training/draft.py`: start from fused target features, then feed each draft hidden output into the same block at the next step. Start with `draft_steps=4`; support 7. Preserve gradients through the unroll.
- [x] Use **shifted ground-truth token embeddings** during the initial training implementation, as in the NVIDIA TTT reference. Shift target distributions and valid masks together. For an anchor at `t`, step `r` (starting at 1) consumes token `x[t+r]` and supervises the distribution for `x[t+r+1]`.
- [x] Maintain causal draft attention history across recursive steps. Follow the shared `cache_hidden` lifecycle in [R3]. For the inherited attention math, read only `_eager_attention_forward` in [R4] if needed; do not read that entire module. Resetting the cache every step is not equivalent to this TTT procedure. Compare with a slow single-anchor incremental reference to catch future-token leakage.
- [x] Implement pure LK loss at temperature 1: `p = softmax(target_logits)`, `q = softmax(draft_logits)`, `overlap = minimum(p, q).sum(-1)`, then masked mean of `-log(overlap.clamp_min(eps))`. Use full vocabulary and float32 probability/loss calculations initially. Choose NVIDIA's **alpha** objective, not its optional hybrid objective. No auxiliary ground-truth CE is required for this stage.
- [x] Combine step losses with a normalized mean or configurable normalized decay weights. Log per-step loss, overlap and top-1 target agreement. Overlap is an expected sampling-acceptance measure; top-1 agreement is relevant to greedy verification. Neither is a measured end-to-end speedup.
- [x] Create `scripts/train_draft.py` using this repository's training/checkpoint utilities and `train.py` conventions. Support config/CLI overrides, validation, accumulation, clipping, save/resume and optional distributed execution. Only draft/projection parameters belong in its optimizer; a separate external orchestration recipe is unnecessary.
- [x] Save draft weights, fusion settings/feature indices, loss/unroll settings, optimizer/step/loader state and the exact target checkpoint identity. Do not overwrite the frozen target checkpoint. Add tiny runnable configs and example commands for both stages.

**Training distinction:** feed back draft hidden states, but use shifted ground-truth token embeddings, as [R3] does. If sampled-token training is added later, recompute target distributions on the sampled prefixes; distributions from a different prefix are not aligned supervision.

**Stage-2 checks**

- [x] Test projection identity and conversion equivalence before any fine-tuning.
- [x] Test recursive token/distribution/mask alignment, short sequences, attention causality, and gradients through multiple calls.
- [x] Test LK loss is approximately zero for identical distributions, positive for mismatched ones, and finite with sparse overlap or empty masks.
- [x] Verify draft/projection gradients and updates, no target gradients, and unchanged target/shared weights after an optimizer step.
- [x] Verify checkpoint round trips and resume; run tiny fixed-batch training and log per-step agreement/overlap against the initial drafter.

## 4. Later: speculative inference

Keep this documented as follow-on work, separate from the training deliverable.

- [x] Begin with greedy speculative decoding at **inference batch size 1**; ordinary training can remain batched. Require token-for-token equality with target-only greedy generation in `test_speculative.py`.
- [x] Define pending-token/cache-length invariants and add transactional fork/commit. KV buffers can share unused future slots; clone KDA state and convolution buffers. On rejection, discard shadow state and replay only the pending token plus accepted prefix. Reset/rebuild draft state too; reducing `seq_len` cannot undo recurrent updates.
- [ ] Add sampling and performance benchmarks in a separate follow-up. These training references do not replace verification of the inference algorithm.

## References

Read the following targeted sections in order. Do not scan the upstream repositories or follow every import. Paths were checked on 2026-10-02; record revisions when implementing and retain attribution/license notices for reused code.

| ID | Code implementation | Read only | Apply to our module |
|---|---|---|---|
| R1 — Stage 1 | [DeepSeek-v3-Lite `models/mtp.py`](https://github.com/atandra2000/DeepSeek-v3-Lite/blob/main/models/mtp.py) | `MTPBlock`, `MTPModule`, `MultiTokenPrediction.forward` and `compute_loss` | Normalize/fuse hidden + next-token embedding; shallow causal block; shared head; shifted CE. Configure one block and integrate with our existing trainer. |
| R2 — Stage 2 structure | [NVIDIA `draft_kimi_k3.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/speculative/eagle/draft_kimi_k3.py) | `Eagle3KimiK3Model.__init__` and `KimiK3Eagle3DecoderLayer.forward` | `3d -> d` feature projection and embedding/hidden fusion. Keep our pretrained MTP and local attention/MLP; production NoPE MLA is not required. |
| R3 — Stage 2 training | [NVIDIA `eagle/core.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/speculative/eagle/core.py) | `Eagle3TrainerModule` training forward and `_lk_step_loss` alpha branch | Reuse draft hidden outputs, shift teacher-forced tokens/distributions/masks together, retain draft history, and optimize masked acceptance overlap. Skip CP, vocabulary remapping and hybrid-loss branches. |
| R4 — conditional helper | [NVIDIA `draft_deepseek.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/speculative/eagle/draft_deepseek.py) | `_eager_attention_forward` only, if implementing/debugging vectorized TTT attention | Causal initial-prefix attention plus per-anchor continuation history. Read this because R2 inherits the attention helper; no need to port the full model. |

**Implementation essentials already distilled here:** one reusable block; Stage-1 shifted auxiliary CE; Stage-2 frozen target; low/mid/high fusion initialized to `[0, 0, I]`; recursive hidden feedback with teacher-forced tokens; aligned causal history; pure LK loss. Use this brief for these choices and the three core files for executable examples. All orchestration and checkpoint integration should follow our local repository.

## Completion criteria

- [x] Stage 1 trains and saves a target with its one-step MTP block.
- [x] Stage 2 loads that checkpoint, keeps the target fixed, trains recursive drafting with feature fusion and LK loss, and saves a reloadable draft.
- [x] Meaningful unit/smoke checks pass; validation limitations are reported.
- [x] Existing target-only training/generation works, and docs explain commands, config fields, checkpoint formats and the teacher-forced TTT distinction.
- [x] Final implementation report lists changed files, checks run, results and remaining speculative-inference work.
