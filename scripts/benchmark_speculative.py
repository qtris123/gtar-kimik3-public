"""
Benchmark and compare inference performance of:
1. Target-only greedy generation (baseline Engine)
2. Stage 1 MTP Drafter (before Stage 2 fine-tuning, directly using target_model.mtp_block)
3. Stage 2 Fine-tuned Drafter (after Stage 2 fine-tuning, using DraftTrainer with feature projection)

Usage:
  python -m scripts.benchmark_speculative \
      --config configs/kimi_k3_0.6B_mtp.json \
      --target-checkpoint out/kimi_k3_0.6B_mtp/ckpt_019073.pt \
      --draft-checkpoint out/kimi_k3_0.6B_mtp_draft_0_4_27/draft_001907.pt \
      --draft-steps 7 \
      --max-new-tokens 128
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.engine import Engine
from src.models import ARCHITECTURES
from src.models.mtp import MTPBlock
from src.training.draft import DraftConfig, DraftTrainer
from src.speculative import SpeculativeEngine


DEFAULT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "In machine learning, speculative decoding is a technique that",
    "Once upon a time, in a quiet village surrounded by misty mountains,",
    "Solve the following step by step: 25 * 14 + 130 =",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Speculative Decoding (Stage 1 vs Stage 2)")
    parser.add_argument("--config", default="configs/kimi_k3_0.6B_mtp.json", help="Path to config json")
    parser.add_argument("--target-checkpoint", required=True, help="Stage 1 target checkpoint path")
    parser.add_argument("--draft-checkpoint", default=None, help="Stage 2 draft checkpoint path (optional)")
    parser.add_argument("--draft-steps", type=int, default=4, help="Tokens to draft per speculation round")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of tokens to generate per prompt")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B", help="Tokenizer name or path")
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompts", nargs="+", default=None, help="Custom prompt strings")
    return parser.parse_args()


def load_models(args):
    print("=" * 70)
    print("Loading Models")
    print("=" * 70)
    device = torch.device(args.device)

    target_path = Path(args.target_checkpoint)
    if not target_path.exists():
        raise FileNotFoundError(f"Target checkpoint not found: {target_path}")

    # 1. Load Target Model from Stage 1 checkpoint
    print(f"Loading Stage 1 target checkpoint from {target_path}...")
    target_ckpt = torch.load(target_path, map_location="cpu")
    config_file = json.loads(Path(args.config).read_text())
    Config, ForCausalLM = ARCHITECTURES[config_file["arch"]]

    target_config = Config(**target_ckpt["config"])
    with torch.device(device):
        target_model = ForCausalLM(target_config)
    target_model.load_state_dict(target_ckpt["model"])
    target_model = target_model.to(device).eval()
    print(f"✓ Target model loaded ({sum(p.numel() for p in target_model.parameters()):,} parameters)")

    # 2. Reconstruct Stage 2 Draft Trainer (optional)
    draft_trainer = None
    if args.draft_checkpoint:
        draft_path = Path(args.draft_checkpoint)
        if not draft_path.exists():
            raise FileNotFoundError(f"Draft checkpoint not found: {draft_path}")
        print(f"Loading Stage 2 draft checkpoint from {draft_path}...")
        draft_ckpt = torch.load(draft_path, map_location="cpu")
        draft_saved_cfg = draft_ckpt["draft_config"]
        draft_config = DraftConfig(
            draft_steps=args.draft_steps,
            feature_layer_indices=draft_saved_cfg.get("feature_layer_indices", target_config.feature_layer_indices),
            step_loss_weighting=draft_saved_cfg.get("step_loss_weighting", "uniform"),
            step_loss_decay=draft_saved_cfg.get("step_loss_decay", 0.9),
        )

        draft_mtp = MTPBlock(
            hidden_size=target_config.hidden_size,
            intermediate_size=target_config.intermediate_size,
            num_attention_heads=target_config.num_attention_heads,
            num_key_value_heads=target_config.num_key_value_heads,
            head_dim=target_config.head_dim,
            rms_norm_eps=target_config.rms_norm_eps,
        ).to(device)

        draft_trainer = DraftTrainer(draft_mtp, draft_config).to(device)
        draft_trainer.load_state_dict(draft_ckpt["draft_trainer"])
        draft_trainer.eval()
        print(f"✓ Stage 2 draft model loaded (Step {draft_ckpt.get('step', 'unknown')})")

    # 3. Load Tokenizer
    print(f"Loading tokenizer: {args.tokenizer}...")
    tokenizer = Tokenizer.from_pretrained(args.tokenizer)
    eos_token_id = tokenizer.token_to_id(args.eos_token)
    print("✓ Tokenizer loaded")

    return target_model, draft_trainer, tokenizer, eos_token_id


def time_generation(fn, *args, **kwargs):
    """Run generation with CUDA synchronization and timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    duration = t1 - t0
    return out, duration


def main():
    args = parse_args()
    device = torch.device(args.device)
    target_model, draft_trainer, tokenizer, eos_token_id = load_models(args)

    prompts = args.prompts if args.prompts is not None else DEFAULT_PROMPTS

    # Build Engines
    engine_baseline = Engine(target_model)
    # Stage 1: uses target_model.mtp_block directly (no feature projection)
    engine_stage1 = SpeculativeEngine(
        target_model=target_model,
        draft_block=target_model.mtp_block,
        feature_proj=None,
        draft_steps=args.draft_steps,
    )
    # Stage 2: uses the fine-tuned draft_trainer (mtp_block + feature_proj) if provided
    engine_stage2 = None
    if draft_trainer is not None:
        engine_stage2 = SpeculativeEngine(
            target_model=target_model,
            draft_block=draft_trainer,
            draft_steps=args.draft_steps,
        )

    print("\n" + "=" * 70)
    print("Warming up engines...")
    warmup_ids = torch.tensor([tokenizer.encode("Hello world").ids], device=device)
    _ = engine_baseline.generate(warmup_ids, max_new_tokens=4, temperature=0, eos_token_id=eos_token_id)
    _ = engine_stage1.generate(warmup_ids, max_new_tokens=4, eos_token_id=eos_token_id)
    if engine_stage2 is not None:
        _ = engine_stage2.generate(warmup_ids, max_new_tokens=4, eos_token_id=eos_token_id)
    print("✓ Warmup complete\n" + "=" * 70)

    # Aggregated stats across prompts
    all_baseline_tokens_per_sec = []
    all_stage1_tokens_per_sec = []
    all_stage2_tokens_per_sec = []

    all_stage1_acc_rates = []
    all_stage2_acc_rates = []

    all_stage1_mean_accepted = []
    all_stage2_mean_accepted = []

    stage1_step_accept_sum = [0] * args.draft_steps
    stage1_step_prop_sum = [0] * args.draft_steps
    stage2_step_accept_sum = [0] * args.draft_steps
    stage2_step_prop_sum = [0] * args.draft_steps

    print("\n" + "#" * 70)
    print(f"BENCHMARK: SPECULATIVE DECODING (draft_steps = {args.draft_steps}, max_new = {args.max_new_tokens})")
    print("#" * 70)

    for p_idx, prompt in enumerate(prompts, 1):
        prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
        print(f"\n[Prompt {p_idx}/{len(prompts)}]: {repr(prompt)}")

        # 1. Baseline Target Greedy
        gen_base, t_base = time_generation(
            engine_baseline.generate,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0,
            eos_token_id=eos_token_id,
        )
        base_tok_count = gen_base.shape[1]
        base_tok_s = base_tok_count / max(1e-6, t_base)
        all_baseline_tokens_per_sec.append(base_tok_s)

        # 2. Stage 1 MTP Drafter (Before Stage 2)
        (gen_s1, stats_s1), t_s1 = time_generation(
            engine_stage1.generate,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            return_stats=True,
        )
        s1_tok_count = gen_s1.shape[1]
        s1_tok_s = s1_tok_count / max(1e-6, t_s1)
        s1_speedup = t_base / max(1e-6, t_s1)
        all_stage1_tokens_per_sec.append(s1_tok_s)
        all_stage1_acc_rates.append(stats_s1["acceptance_rate"])
        all_stage1_mean_accepted.append(stats_s1["mean_accepted_per_round"])

        for i in range(args.draft_steps):
            stage1_step_accept_sum[i] += stats_s1["per_step_accepted"][i]
            stage1_step_prop_sum[i] += stats_s1["per_step_proposed"][i]

        # 3. Stage 2 Fine-Tuned Drafter (After Stage 2, optional)
        if engine_stage2 is not None:
            (gen_s2, stats_s2), t_s2 = time_generation(
                engine_stage2.generate,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_token_id,
                return_stats=True,
            )
            s2_tok_count = gen_s2.shape[1]
            s2_tok_s = s2_tok_count / max(1e-6, t_s2)
            s2_speedup = t_base / max(1e-6, t_s2)
            all_stage2_tokens_per_sec.append(s2_tok_s)
            all_stage2_acc_rates.append(stats_s2["acceptance_rate"])
            all_stage2_mean_accepted.append(stats_s2["mean_accepted_per_round"])

            for i in range(args.draft_steps):
                stage2_step_accept_sum[i] += stats_s2["per_step_accepted"][i]
                stage2_step_prop_sum[i] += stats_s2["per_step_proposed"][i]
            match_s2 = torch.equal(gen_base, gen_s2)
        else:
            match_s2 = "N/A"

        # Invariant Verification: speculative outputs MUST match baseline exactly
        match_s1 = torch.equal(gen_base, gen_s1)

        # Print Prompt Results
        text_generated = tokenizer.decode(gen_base[0].tolist())
        print(f"Generated text: {repr(text_generated[:100])}...")
        print(f"Verification: Stage 1 match = {match_s1} | Stage 2 match = {match_s2}")
        print(f"  * Baseline:   {base_tok_s:6.1f} tok/s ({t_base*1000:6.1f} ms)")
        print(f"  * Stage 1:    {s1_tok_s:6.1f} tok/s ({t_s1*1000:6.1f} ms) | Speedup: {s1_speedup:4.2f}x | Accept rate: {stats_s1['acceptance_rate']*100:5.1f}% (mean {stats_s1['mean_accepted_per_round']:.2f}/round)")
        if engine_stage2 is not None:
            print(f"  * Stage 2:    {s2_tok_s:6.1f} tok/s ({t_s2*1000:6.1f} ms) | Speedup: {s2_speedup:4.2f}x | Accept rate: {stats_s2['acceptance_rate']*100:5.1f}% (mean {stats_s2['mean_accepted_per_round']:.2f}/round)")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    avg_base_tok_s = sum(all_baseline_tokens_per_sec) / len(all_baseline_tokens_per_sec)
    avg_s1_tok_s = sum(all_stage1_tokens_per_sec) / len(all_stage1_tokens_per_sec)
    avg_s1_acc = sum(all_stage1_acc_rates) / len(all_stage1_acc_rates) * 100
    avg_s1_alpha = sum(all_stage1_mean_accepted) / len(all_stage1_mean_accepted)
    s1_avg_speedup = avg_s1_tok_s / avg_base_tok_s

    print("\n" + "=" * 78)
    print("OVERALL INFERENCE COMPARISON (Averaged across prompts)")
    print("=" * 78)
    print(f"{'Method':<32} | {'Throughput':<12} | {'Speedup':<9} | {'Accept Rate':<12} | {'Mean Accept/Round'}")
    print("-" * 78)
    print(f"{'1. Baseline (Target-Only Greedy)':<32} | {avg_base_tok_s:7.1f} tok/s | {'1.00x':<9} | {'N/A':<12} | {'N/A'}")
    print(f"{'2. Stage 1 MTP (Before Stage 2)':<32} | {avg_s1_tok_s:7.1f} tok/s | {f'{s1_avg_speedup:.2f}x':<9} | {f'{avg_s1_acc:.1f}%':<12} | {f'{avg_s1_alpha:.2f} tokens'}")
    if engine_stage2 is not None:
        avg_s2_tok_s = sum(all_stage2_tokens_per_sec) / len(all_stage2_tokens_per_sec)
        avg_s2_acc = sum(all_stage2_acc_rates) / len(all_stage2_acc_rates) * 100
        avg_s2_alpha = sum(all_stage2_mean_accepted) / len(all_stage2_mean_accepted)
        s2_avg_speedup = avg_s2_tok_s / avg_base_tok_s
        print(f"{'3. Stage 2 Draft (After Stage 2)':<32} | {avg_s2_tok_s:7.1f} tok/s | {f'{s2_avg_speedup:.2f}x':<9} | {f'{avg_s2_acc:.1f}%':<12} | {f'{avg_s2_alpha:.2f} tokens'}")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("PER-STEP ACCEPTANCE BREAKDOWN")
    print("=" * 78)
    if engine_stage2 is not None:
        print(f"{'Draft Step':<12} | {'Stage 1 (Before)':<25} | {'Stage 2 (After)':<25} | {'Delta':<10}")
        print("-" * 78)
        for k in range(args.draft_steps):
            s1_p = stage1_step_prop_sum[k]
            s1_a = stage1_step_accept_sum[k]
            s1_rate = (s1_a / max(1, s1_p)) * 100

            s2_p = stage2_step_prop_sum[k]
            s2_a = stage2_step_accept_sum[k]
            s2_rate = (s2_a / max(1, s2_p)) * 100

            delta = s2_rate - s1_rate
            sign = "+" if delta >= 0 else ""
            print(f"Step {k+1:<7} | {s1_rate:5.1f}% ({s1_a}/{s1_p:<5})         | {s2_rate:5.1f}% ({s2_a}/{s2_p:<5})         | {sign}{delta:5.1f}%")
    else:
        print(f"{'Draft Step':<12} | {'Stage 1 Acceptance Rate':<30}")
        print("-" * 78)
        for k in range(args.draft_steps):
            s1_p = stage1_step_prop_sum[k]
            s1_a = stage1_step_accept_sum[k]
            s1_rate = (s1_a / max(1, s1_p)) * 100
            print(f"Step {k+1:<7} | {s1_rate:5.1f}% ({s1_a}/{s1_p:<5})")
    print("=" * 78)


if __name__ == "__main__":
    main()
