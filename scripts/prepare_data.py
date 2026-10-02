"""
Tokenize the parquet files of a Hugging Face dataset into flat shards of token ids. From the repo root:

python -m scripts.prepare_data --dataset HuggingFaceFW/fineweb-edu --subset sample/10BT --out data/fineweb_edu

Shard 0 is the validation split, the rest are training shards. Documents are separated by the EOS token.
"""

import argparse
import json
import os
from pathlib import Path

# If /scratch exists, redirect Hugging Face cache there so it doesn't fill up the root disk
if Path("/scratch").exists():
    os.environ.setdefault("HF_HOME", "/scratch/.cache/huggingface")
    Path("/scratch/.cache/huggingface").mkdir(parents=True, exist_ok=True)

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description="Tokenize a dataset into .bin shards")
parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
parser.add_argument("--subset", default="sample/10BT", help="path prefix of the parquet files inside the repo")
parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
parser.add_argument("--eos-token", default="<|endoftext|>")
parser.add_argument("--out", default="data/fineweb_edu")
parser.add_argument("--shard-size", type=int, default=100_000_000, help="tokens per train shard")
parser.add_argument("--val-size", type=int, default=20_000_000, help="tokens in the val shard")
parser.add_argument("--max-tokens", type=int, default=-1, help="stop after this many tokens (-1 = whole dataset)")
args = parser.parse_args()

tokenizer = Tokenizer.from_pretrained(args.tokenizer)
eos_id = tokenizer.token_to_id(args.eos_token)
vocab_size = tokenizer.get_vocab_size()
dtype = np.uint16 if vocab_size < 2**16 else np.uint32

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
meta = {"tokenizer": args.tokenizer, "vocab_size": vocab_size, "eos_id": eos_id, "dtype": np.dtype(dtype).name}
(out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def shard_path(index: int) -> Path:
    return out_dir / f"{'val' if index == 0 else 'train'}_{index:06d}.bin"


def document_batches(batch_size: int = 1024):
    files = HfApi().list_repo_files(args.dataset, repo_type="dataset")
    for file in sorted(f for f in files if f.startswith(args.subset) and f.endswith(".parquet")):
        path = hf_hub_download(args.dataset, file, repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size, columns=["text"]):
            yield batch.column("text").to_pylist()


docs, num_buffered, shard_idx, total_tokens = [], 0, 0, 0
for texts in document_batches():
    for encoding in tokenizer.encode_batch(texts):
        docs.append(np.array(encoding.ids + [eos_id], dtype=dtype))
        num_buffered += len(docs[-1])
    while num_buffered >= (capacity := args.val_size if shard_idx == 0 else args.shard_size):
        tokens = np.concatenate(docs)
        tokens[:capacity].tofile(shard_path(shard_idx))
        docs, num_buffered = [tokens[capacity:]], len(tokens) - capacity
        shard_idx, total_tokens = shard_idx + 1, total_tokens + capacity
        print(f"wrote {shard_path(shard_idx - 1)} | {total_tokens:,} tokens total")
    if 0 < args.max_tokens <= total_tokens:
        break
else:
    if num_buffered > 0:
        np.concatenate(docs).tofile(shard_path(shard_idx))
        print(f"wrote {shard_path(shard_idx)} | {total_tokens + num_buffered:,} tokens total")
