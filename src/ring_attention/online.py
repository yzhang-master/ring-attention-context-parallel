from dataclasses import dataclass

import torch


@dataclass
class OnlineAttentionState:
    running_max: torch.Tensor
    normalizer: torch.Tensor
    accumulator: torch.Tensor

    @classmethod
    def empty(cls, query: torch.Tensor) -> "OnlineAttentionState":
        prefix = query.shape[:-1]
        return cls(
            running_max=torch.full(prefix, -torch.inf, dtype=query.dtype, device=query.device),
            normalizer=torch.zeros(prefix, dtype=query.dtype, device=query.device),
            accumulator=torch.zeros_like(query),
        )

    def output(self) -> torch.Tensor:
        tiny = torch.finfo(self.normalizer.dtype).tiny
        return self.accumulator / self.normalizer.unsqueeze(-1).clamp_min(tiny)

    def log_normalizer(self) -> torch.Tensor:
        return self.running_max + self.normalizer.log()


def absorb_attention_block(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: OnlineAttentionState,
    allowed: torch.Tensor | None = None,
) -> OnlineAttentionState:
    """Merge one K/V block without materializing probabilities across other blocks."""
    scores = torch.matmul(query, key.transpose(-1, -2)) / (query.shape[-1] ** 0.5)
    if allowed is not None:
        scores = scores.masked_fill(~allowed, -torch.inf)

    block_max = scores.amax(dim=-1)
    new_max = torch.maximum(state.running_max, block_max)
    old_scale = torch.where(
        torch.isneginf(state.running_max),
        0.0,
        torch.exp(state.running_max - new_max),
    )
    if allowed is None:
        probabilities = torch.exp(scores - new_max.unsqueeze(-1))
    else:
        probabilities = torch.where(
            allowed,
            torch.exp(scores - new_max.unsqueeze(-1)),
            0.0,
        )

    normalizer = state.normalizer * old_scale + probabilities.sum(dim=-1)
    accumulator = (
        state.accumulator * old_scale.unsqueeze(-1)
        + torch.matmul(probabilities, value)
    )
    return OnlineAttentionState(new_max, normalizer, accumulator)
