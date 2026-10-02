import torch


class KVCache:
    def __init__(self, batch_size: int, num_key_value_heads: int, max_seq_len: int, head_dim: int, device, dtype):
        shape = (batch_size, num_key_value_heads, max_seq_len, head_dim)
        self.keys = torch.zeros(shape, device=device, dtype=dtype)
        self.values = torch.zeros(shape, device=device, dtype=dtype)
        self.seq_len = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new_len = self.seq_len + key_states.shape[2]
        self.keys[:, :, self.seq_len : new_len] = key_states
        self.values[:, :, self.seq_len : new_len] = value_states
        self.seq_len = new_len
        return self.keys[:, :, :new_len], self.values[:, :, :new_len]


class RecurrentCache:
    def __init__(self, batch_size: int, num_heads: int, head_dim: int, value_dim: int, inner_size: int, conv_size: int, device, dtype):
        self.state = torch.zeros(batch_size, num_heads, head_dim, value_dim, device=device, dtype=torch.float32)
        self.conv_states = [torch.zeros(batch_size, inner_size, conv_size - 1, device=device, dtype=dtype) for _ in range(3)]
        self.seq_len = 0

    def update(self, state: torch.Tensor, seq_len: int) -> None:
        self.state = state.contiguous()
        self.seq_len += seq_len


class Cache:
    def __init__(self, layers: list):
        self.layers = layers

    def __getitem__(self, layer_idx: int):
        return self.layers[layer_idx]

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len
