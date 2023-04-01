import pytest
import torch

from ring_attention import sequence_shard


def test_sequence_shard_owns_disjoint_contiguous_ranges() -> None:
    values = torch.arange(24).reshape(1, 1, 6, 4)
    first = sequence_shard(values, rank=0, world_size=2)
    second = sequence_shard(values, rank=1, world_size=2)
    assert torch.equal(torch.cat([first, second], dim=-2), values)
    assert first.is_contiguous()


def test_sequence_shard_rejects_partial_rank() -> None:
    with pytest.raises(ValueError, match="divisible"):
        sequence_shard(torch.empty(1, 1, 5, 4), rank=0, world_size=2)
