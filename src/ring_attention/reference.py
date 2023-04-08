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
        length = query.shape[-2]
        allowed = torch.ones(length, length, dtype=torch.bool, device=query.device).tril()
        scores = scores.masked_fill(~allowed, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value)
