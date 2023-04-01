import torch

from ring_attention import sequence_shard


def main() -> None:
    values = torch.arange(16).reshape(1, 1, 4, 4)
    shards = [sequence_shard(values, rank, 2) for rank in range(2)]
    torch.testing.assert_close(torch.cat(shards, dim=-2), values)
    print("sequence sharding smoke: ok")


if __name__ == "__main__":
    main()
