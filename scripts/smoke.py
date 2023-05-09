import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ring_attention import full_attention, ring_attention, sequence_shard


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank: int, port: int) -> None:
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(101)
        query = torch.randn(1, 2, 8, 4)
        key = torch.randn(1, 2, 8, 4)
        value = torch.randn(1, 2, 8, 4)
        local = [sequence_shard(tensor, rank, 2) for tensor in (query, key, value)]
        worst = torch.tensor(0.0)
        for causal in (False, True):
            actual = ring_attention(*local, causal=causal)
            expected = sequence_shard(full_attention(query, key, value, causal=causal), rank, 2)
            worst = torch.maximum(worst, (actual - expected).abs().max())
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        if rank == 0:
            torch.testing.assert_close(worst, torch.tensor(0.0), atol=5e-6, rtol=0)
            print(f"two-rank gloo dense+causal smoke: ok (max abs error {worst.item():.3g})")
    finally:
        dist.destroy_process_group()


def main() -> None:
    mp.spawn(_worker, args=(_port(),), nprocs=2, join=True)


if __name__ == "__main__":
    main()
