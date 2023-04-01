import torch


def sequence_shard(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Return a contiguous rank-owned slice of the sequence dimension."""
    if tensor.ndim < 3:
        raise ValueError("expected [batch, heads, sequence, features] or equivalent")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid rank/world_size")
    if tensor.shape[-2] % world_size:
        raise ValueError("sequence length must be divisible by world_size")
    local_length = tensor.shape[-2] // world_size
    return tensor.narrow(-2, rank * local_length, local_length).contiguous()
