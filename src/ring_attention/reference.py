import torch


def full_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-1, -2)) / (query.shape[-1] ** 0.5)
    if causal:
        query_positions = torch.arange(query.shape[-2], device=query.device)
        key_positions = torch.arange(key.shape[-2], device=query.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~allowed, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value)
