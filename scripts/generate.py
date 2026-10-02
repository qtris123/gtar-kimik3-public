"""
Generate text from a checkpoint. From the repo root:

python -m scripts.generate --config configs/kimi_k3_0.6B.json --checkpoint out/kimi_k3_0.6B/ckpt_019073.pt \
    --prompt "The capital of France is" --max-new-tokens 64 --temperature 0.8 --top-k 50 --num-samples 2
"""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.engine import Engine
from src.models import ARCHITECTURES

parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
parser.add_argument("--config", required=True, help="json config the model was trained with")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--prompt", default="The capital of France is")
parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
parser.add_argument("--eos-token", default="<|endoftext|>")
parser.add_argument("--max-new-tokens", type=int, default=64)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

config_file = json.loads(Path(args.config).read_text())
Config, ForCausalLM = ARCHITECTURES[config_file["arch"]]
checkpoint = torch.load(args.checkpoint, map_location="cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.device(device):
    model = ForCausalLM(Config(**checkpoint["config"]))
model.load_state_dict(checkpoint["model"])

tokenizer = Tokenizer.from_pretrained(args.tokenizer)
eos_token_id = tokenizer.token_to_id(args.eos_token)
prompt_ids = torch.tensor([tokenizer.encode(args.prompt).ids] * args.num_samples)

torch.manual_seed(args.seed)
generated = Engine(model).generate(prompt_ids, args.max_new_tokens, args.temperature, args.top_k, eos_token_id)
for tokens in generated.tolist():
    tokens = tokens[: tokens.index(eos_token_id)] if eos_token_id in tokens else tokens
    print(args.prompt + tokenizer.decode(tokens))
    print("---")
