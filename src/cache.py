import torch


class KVCache:
    def __init__(self, batch_size: int, num_key_value_heads: int, max_seq_len: int, head_dim: int, device, dtype):
        shape = (batch_size, num_key_value_heads, max_seq_len, head_dim)
        self.keys = torch.zeros(shape, device=device, dtype=dtype)
        self.values = torch.zeros(shape, device=device, dtype=dtype)
        self.seq_len = 0
        self._snapshot_seq_len = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new_len = self.seq_len + key_states.shape[2]
        self.keys[:, :, self.seq_len : new_len] = key_states
        self.values[:, :, self.seq_len : new_len] = value_states
        self.seq_len = new_len
        return self.keys[:, :, :new_len], self.values[:, :, :new_len]

    def fork(self) -> None:
        """Snapshot the current seq_len for potential rollback.

        KV buffers share unused future slots -- no data is copied.
        Speculative tokens are written into the pre-allocated buffer
        and simply abandoned (overwritten) on rollback.
        """
        self._snapshot_seq_len = self.seq_len

    def commit(self, n_accepted: int) -> None:
        """Accept n_accepted speculative tokens after the snapshot point.

        Sets seq_len to snapshot + n_accepted and clears the snapshot.
        """
        assert self._snapshot_seq_len is not None, "commit() called without fork()"
        self.seq_len = self._snapshot_seq_len + n_accepted
        self._snapshot_seq_len = None

    def rollback(self) -> None:
        """Discard all tokens written since fork().

        Only resets seq_len; the buffer content beyond the snapshot is
        stale but will be overwritten on the next update.
        """
        assert self._snapshot_seq_len is not None, "rollback() called without fork()"
        self.seq_len = self._snapshot_seq_len
        self._snapshot_seq_len = None


class RecurrentCache:
    def __init__(self, batch_size: int, num_heads: int, head_dim: int, value_dim: int, inner_size: int, conv_size: int, device, dtype):
        self.state = torch.zeros(batch_size, num_heads, head_dim, value_dim, device=device, dtype=torch.float32)
        self.conv_states = [torch.zeros(batch_size, inner_size, conv_size - 1, device=device, dtype=dtype) for _ in range(3)]
        self.seq_len = 0
        self._snapshot = None

    def update(self, state: torch.Tensor, seq_len: int) -> None:
        self.state = state.contiguous()
        self.seq_len += seq_len

    def fork(self) -> None:
        """Deep-clone the recurrent state and convolution buffers.

        Unlike KV caches, recurrent state cannot be shared because
        reducing seq_len does not undo recurrent updates.
        """
        self._snapshot = {
            "state": self.state.clone(),
            "conv_states": [cs.clone() for cs in self.conv_states],
            "seq_len": self.seq_len,
        }

    def commit(self, n_accepted: int) -> None:
        """Accept the current recurrent state and discard the snapshot.

        Only valid when all verified tokens are accepted.
        """
        assert self._snapshot is not None, "commit() called without fork()"
        self._snapshot = None

    def rollback(self) -> None:
        """Restore the recurrent state and conv buffers from the snapshot."""
        assert self._snapshot is not None, "rollback() called without fork()"
        self.state = self._snapshot["state"].clone()
        for i, cs in enumerate(self._snapshot["conv_states"]):
            self.conv_states[i].copy_(cs)
        self.seq_len = self._snapshot["seq_len"]
        self._snapshot = None


class Cache:
    def __init__(self, layers: list):
        self.layers = layers

    def __getitem__(self, layer_idx: int):
        return self.layers[layer_idx]

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len

    def fork(self) -> None:
        """Fork all layer caches for speculative decoding."""
        for layer_cache in self.layers:
            layer_cache.fork()

    def commit(self, n_accepted: int) -> None:
        """Commit n_accepted speculative tokens across all layers."""
        for layer_cache in self.layers:
            layer_cache.commit(n_accepted)

    def rollback(self) -> None:
        """Rollback all layer caches to the snapshot."""
        for layer_cache in self.layers:
            layer_cache.rollback()


class DraftTTTCache:
    """TTT (Training-Time Test) attention cache for EAGLE-3 drafter.

    Preserves per-anchor K/V history across recursive speculative depths:
    - Step 1: stores full sequence K/V states (causal initial prefix).
    - Step r (r >= 2): stores depth-r K/V states.

    In vectorized training mode:
      At step r, query position t attends to:
      1. Prefix positions 0..t from step 1 (causal prefix).
      2. Same anchor position t from steps 2..r (diagonal extension).
      This avoids mixing different anchors' speculative trajectories.

    In single-anchor mode (speculative inference or sequential reference):
      At step r, query (length 1) attends to the anchor's causal prefix
      plus the same anchor's previous draft K/V states.
    """

    def __init__(
        self,
        prefix_k: torch.Tensor | None = None,
        prefix_v: torch.Tensor | None = None,
        mode: str = "vectorized",
    ):
        self.k_list: list[torch.Tensor] = []
        self.v_list: list[torch.Tensor] = []
        self.prefix_k = prefix_k
        self.prefix_v = prefix_v
        self.mode = mode
        self._snapshot_len: int | None = None

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if len(self.k_list) == 0 and self.prefix_k is not None and self.prefix_v is not None:
            # Step 1: concatenate persistent prefix with current anchor's key/value
            self.k_list.append(torch.cat([self.prefix_k, k], dim=2))
            self.v_list.append(torch.cat([self.prefix_v, v], dim=2))
        else:
            self.k_list.append(k)
            self.v_list.append(v)

    @property
    def step_count(self) -> int:
        return len(self.k_list)

    def reset(self) -> None:
        self.k_list.clear()
        self.v_list.clear()
        self._snapshot_len = None

    def fork(self) -> None:
        """Snapshot current depth for speculative transactional rollback."""
        self._snapshot_len = len(self.k_list)

    def commit(self) -> None:
        """Clear snapshot upon acceptance."""
        self._snapshot_len = None

    def rollback(self) -> None:
        """Rollback to the snapshot depth."""
        if self._snapshot_len is not None:
            self.k_list = self.k_list[: self._snapshot_len]
            self.v_list = self.v_list[: self._snapshot_len]
            self._snapshot_len = None
