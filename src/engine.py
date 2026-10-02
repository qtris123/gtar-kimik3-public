import torch
import torch.nn as nn


def sample(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k is not None:
        cutoff = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    return torch.multinomial(logits.softmax(-1), 1).squeeze(-1)


class Engine:
    def __init__(self, model: nn.Module):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        batch_size, prompt_len = input_ids.shape
        cache = self.model.make_cache(batch_size, prompt_len + max_new_tokens, self.dtype)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        generated = []
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self.model(input_ids, cache=cache)[:, -1]
            for _ in range(max_new_tokens):
                next_token = sample(logits.float(), temperature, top_k)
                if eos_token_id is not None:
                    next_token = torch.where(finished, eos_token_id, next_token)
                    finished |= next_token == eos_token_id
                generated.append(next_token)
                if finished.all():
                    break
                logits = self.model(next_token[:, None], cache=cache)[:, -1]
        return torch.stack(generated, dim=1)
