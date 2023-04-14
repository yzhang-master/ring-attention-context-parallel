import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ring_attention import full_attention, ring_attention, sequence_shard
from ring_attention.ring import ring_peers


def test_single_rank_matches_full_fp32_attention() -> None:
    torch.manual_seed(23)
    query = torch.randn(1, 2, 7, 4)
    key = torch.randn(1, 2, 7, 4)
    value = torch.randn_like(key)
    for causal in (False, True):
        actual = ring_attention(query, key, value, causal=causal)
        expected = full_attention(query, key, value, causal=causal)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    ("rank", "world_size", "expected"),
    [(0, 4, (3, 1)), (2, 4, (1, 3)), (3, 4, (2, 0))],
)
def test_ring_neighbors_wrap(rank: int, world_size: int, expected: tuple[int, int]) -> None:
    assert ring_peers(rank, world_size) == expected


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _two_rank_worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(37)
        query = torch.randn(1, 2, 8, 4, dtype=torch.float32)
        key = torch.randn(1, 2, 8, 4, dtype=torch.float32)
        value = torch.randn(1, 2, 8, 4, dtype=torch.float32)
        local_query = sequence_shard(query, rank, world_size)
        local_key = sequence_shard(key, rank, world_size)
        local_value = sequence_shard(value, rank, world_size)
        for causal in (False, True):
            actual = ring_attention(local_query, local_key, local_value, causal=causal)
            expected = full_attention(query, key, value, causal=causal)
            expected = sequence_shard(expected, rank, world_size)
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    finally:
        dist.destroy_process_group()


def test_two_rank_cpu_gloo_matches_full_fp32_attention() -> None:
    mp.spawn(_two_rank_worker, args=(2, _unused_tcp_port()), nprocs=2, join=True)
