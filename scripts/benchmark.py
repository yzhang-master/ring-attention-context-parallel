import argparse
import json
import statistics
import time

import torch
import torch.distributed as dist

from ring_attention import ring_attention


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fixed-global-sequence ring attention")
    parser.add_argument("--global-sequence", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--baseline-ms", type=float)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--max-search", action="store_true", help="double sequence until CUDA OOM")
    args = parser.parse_args()

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        device = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    global_sequence = args.global_sequence
    largest_completed = 0
    try:
        while True:
            if global_sequence % world_size:
                raise ValueError("global sequence must be divisible by world size")
            local_length = global_sequence // world_size
            shape = (args.batch, args.heads, local_length, args.head_dim)
            try:
                query = torch.randn(shape, device=device, dtype=dtype)
                key = torch.randn(shape, device=device, dtype=dtype)
                value = torch.randn(shape, device=device, dtype=dtype)
                for _ in range(args.warmup):
                    ring_attention(query, key, value, causal=args.causal)
                _synchronize(device)
                samples = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    ring_attention(query, key, value, causal=args.causal)
                    _synchronize(device)
                    samples.append((time.perf_counter() - start) * 1000)
                largest_completed = global_sequence
            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print(json.dumps({"largest_completed_sequence": largest_completed}))
                break

            if args.max_search:
                if device.type != "cuda":
                    raise ValueError("--max-search requires CUDA")
                del query, key, value
                global_sequence *= 2
                continue

            milliseconds = statistics.median(samples)
            efficiency = None
            if args.baseline_ms is not None:
                efficiency = 100 * args.baseline_ms / (world_size * milliseconds)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "backend": dist.get_backend(),
                            "world_size": world_size,
                            "global_sequence": global_sequence,
                            "dtype": str(dtype).removeprefix("torch."),
                            "causal": args.causal,
                            "median_ms": milliseconds,
                            "strong_scaling_efficiency_pct": efficiency,
                            "samples_ms": samples,
                        }
                    )
                )
            break
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
