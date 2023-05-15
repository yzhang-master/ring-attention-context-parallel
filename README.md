# Ring Attention Context Parallel

Technique from 2023, reimplemented in 2026.

## Context

Long-context attention is limited by the memory needed for queries, keys, values, and attention intermediates. Context parallelism partitions the sequence across ranks instead of replicating the full sequence on every device.

## Problem

Compute each rank's local query outputs against every sequence shard without gathering full K/V tensors or running full attention independently on each rank. Causal masks must use global token positions, and communication should overlap block computation where the backend supports asynchronous work.

## Method

For rank `r` of `P`, each input has shape `[B, H, S/P, D]`. K/V rotate through `isend`/`irecv`; the block received from `(r-step) mod P` is used at iteration `step`:

```text
rank r: local K/V -> r+1 -> r+2 -> ... -> r-1
                 compute Q_r with current K/V while transfers are in flight
```

Every block merges into an online-softmax state `(m, l, a)`:

```text
m' = max(m, rowmax(scores))
l' = exp(m-m') l + sum(exp(scores-m'))
a' = exp(m-m') a + exp(scores-m') V
output = a'/l'
```

Fully masked causal blocks contribute zero probability and preserve `-inf` log normalizers. Communication uses actual `torch.distributed.Work` objects. Transfers for the next block are posted before QK/AV computation and awaited afterward. There is no full-K/V gather and no full-attention call inside the ring.

## Results

Local CPU correctness only:

- Real two-rank Gloo dense and causal attention agree with full fp32 attention (`atol=rtol=3e-6`).
- The end-to-end smoke observed maximum absolute error `1.19e-07`.
- Unit tests cover online rescaling, large scores, fully masked blocks, shard ownership, and ring wraparound.

GPU measurements are **pending**; no GPU numbers are claimed.

| Metric | Result | Command |
|---|---:|---|
| Maximum sequence, 2 GPUs | pending | `torchrun --standalone --nproc-per-node=2 scripts/benchmark.py --global-sequence 4096 --max-search` |
| Strong scaling, 1 GPU | pending baseline | `torchrun --standalone --nproc-per-node=1 scripts/benchmark.py --global-sequence 32768` |
| Strong scaling efficiency, 2 GPUs | pending | `torchrun --standalone --nproc-per-node=2 scripts/benchmark.py --global-sequence 32768 --baseline-ms <ONE_GPU_MS>` |

Efficiency is `100 * t1 / (world_size * tN)` at fixed global sequence length. The benchmark emits JSON with all samples.

## Reproduce

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest -q
PYTHONPATH=src python scripts/smoke.py
PYTHONPATH=src torchrun --standalone --nproc-per-node=2 scripts/benchmark.py \
  --global-sequence 4096 --causal
```

## Limitations

- Forward pass only; backward and autograd-aware communication are not implemented.
- All ranks use equal, contiguous sequence shards; the global sequence must be divisible by world size.
- Causal future blocks are masked but still computed, so causal load balancing is not optimized.
- Overlap depends on backend support for asynchronous point-to-point operations. CPU validation uses Gloo and fp32; CUDA uses NCCL and fp16 in the benchmark.
- `--max-search` is CUDA-only and stops at the first CUDA out-of-memory error.

## References

- Liu et al., [*Ring Attention with Blockwise Transformers for Near-Infinite Context*](https://arxiv.org/abs/2310.01889), 2023.
- Dao et al., [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135), 2022.
