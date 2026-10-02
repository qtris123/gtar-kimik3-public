# LLM Architecture from scratch

Minimal PyTorch codebase for pretraining and inference on language models.

## Structure

```
griffin/
├── configs/          # Model + training defaults (JSON)
├── scripts/
│   ├── prepare_data.py   # Tokenize HF datasets into .bin shards
│   ├── train.py          # Pretraining loop (single- or multi-GPU)
│   ├── generate.py       # Text generation from a checkpoint
│   └── *.sh              # Convenience launchers
├── src/
│   ├── models/       # Architectures: transformer, kimi_k3
│   ├── layers/       # Building blocks (Attention, KDA, RMSNorm, RoPE, SwiGLU, …)
│   ├── ops/          # Low-level kernels (flash attention, linear/delta attention, …)
│   ├── cache.py      # KV / recurrent state for inference
│   ├── engine.py     # Autoregressive generation
│   ├── dataset.py    # Token shard loader
│   ├── optim.py      # Optimizer + LR schedule
│   └── common.py     # Distributed setup, eval, checkpointing
└── tests/            # Unit tests and micro-benchmarks
```

## Quick start

```bash
uv venv
uv pip install torch tokenizers huggingface_hub pyarrow numpy

# Prepare data
uv run python -m scripts.prepare_data --dataset HuggingFaceFW/fineweb-edu --subset sample/10BT --out data/fineweb_edu

# Train (CPU smoke test)
uv run python -m scripts.train --config configs/transformer_tiny.json --data data/fineweb_edu --no-compile

# Generate
uv run python -m scripts.generate --config configs/transformer_tiny.json --checkpoint out/transformer_tiny/ckpt.pt \
    --prompt "The capital of France is"
```

Configs set the architecture via `"arch"` (`transformer` or `kimi_k3`) and model hyperparameters; the `"train"` section provides defaults overridable on the CLI.

## TODO

- [x] Training codebase — data prep, pretraining loop (single/multi-GPU), optimizer, checkpointing
- [x] Inference codebase — generation engine, KV Cache
- [x] Low-level kernels — Flash Attention 2, GDN, KDA, ...
- [x] Architecture support — Transformer++, Kimi-K3
- [ ] Architecture benchmarks — profiling, memory usage, inference cost, pros/cons vs transformer
- [ ] Eval scripts — perplexity, downstream task benchmarks
- [ ] Assignment guide — step-by-step docs with todo breakdown for each part
- [ ] Multi-token prediction in training
- [ ] SFT and RL pipelines# gtar-kimik3-public
