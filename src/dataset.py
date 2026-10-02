import json
from pathlib import Path

import numpy as np
import torch


class TokenLoader:
    def __init__(
        self,
        data_dir: str,
        split: str,
        batch_size: int,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        device: str | torch.device = "cpu",
    ):
        self.data_dir = Path(data_dir)
        self.meta = json.loads((self.data_dir / "meta.json").read_text())
        self.shards = sorted(self.data_dir.glob(f"{split}_*.bin"))
        assert self.shards, f"no {split} shards found in {data_dir}"
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.tokens_per_step = world_size * batch_size * seq_len
        self.load_state_dict({"shard_idx": 0, "pos": 0, "epoch": 1})

    def state_dict(self) -> dict:
        return {"shard_idx": self.shard_idx, "pos": self.pos, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        self.shard_idx, self.pos, self.epoch = state["shard_idx"], state["pos"], state["epoch"]
        self.tokens = np.memmap(self.shards[self.shard_idx], dtype=self.meta["dtype"], mode="r")
        assert len(self.tokens) > self.tokens_per_step, f"{self.shards[self.shard_idx]} is smaller than one step"

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pos + self.tokens_per_step + 1 > len(self.tokens):
            shard_idx = (self.shard_idx + 1) % len(self.shards)
            self.load_state_dict({"shard_idx": shard_idx, "pos": 0, "epoch": self.epoch + (shard_idx == 0)})
        start = self.pos + self.rank * self.batch_size * self.seq_len
        chunk = torch.from_numpy(self.tokens[start : start + self.batch_size * self.seq_len + 1].astype(np.int64))
        if self.device.type == "cuda":
            chunk = chunk.pin_memory()
        chunk = chunk.to(self.device, non_blocking=True)
        self.pos += self.tokens_per_step
        inputs = chunk[:-1].view(self.batch_size, self.seq_len)
        targets = chunk[1:].view(self.batch_size, self.seq_len)
        return inputs, targets
