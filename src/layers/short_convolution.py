import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConvolution(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size, padding=kernel_size - 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, conv_state: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)
        if conv_state is None:
            hidden_states = self.conv(hidden_states)[..., :seq_len]
        else:
            hidden_states = torch.cat([conv_state, hidden_states], dim=-1)
            conv_state.copy_(hidden_states[..., -conv_state.shape[-1] :])
            hidden_states = F.conv1d(hidden_states, self.conv.weight, groups=self.conv.groups)
        return F.silu(hidden_states).transpose(1, 2)
