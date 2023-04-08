import torch
import torch.distributed as dist

from ring_attention.online import OnlineAttentionState, absorb_attention_block


def ring_peers(rank: int, world_size: int) -> tuple[int, int]:
    return (rank - 1) % world_size, (rank + 1) % world_size


def causal_block_mask(
    query_start: int,
    key_start: int,
    query_length: int,
    key_length: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = query_start + torch.arange(query_length, device=device)
    key_positions = key_start + torch.arange(key_length, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


@torch.no_grad()
def ring_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    process_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Attend local queries to all K/V shards exchanged around a rank ring."""
    if query.ndim != 4 or key.shape != value.shape:
        raise ValueError("expected q [B,H,S,D] and identically shaped k/v")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("q/k/v batch, head, and feature dimensions must match")
    if query.shape[-2] != key.shape[-2]:
        raise ValueError("this implementation uses equal local query and K/V lengths")
    if not query.is_floating_point() or query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("q/k/v must use the same floating-point dtype")

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)
    else:
        if process_group is not None:
            raise RuntimeError("a process group requires torch.distributed initialization")
        rank, world_size = 0, 1

    previous_rank, next_rank = ring_peers(rank, world_size)
    local_length = query.shape[-2]
    current_key = key.contiguous()
    current_value = value.contiguous()
    state = OnlineAttentionState.empty(query)

    for step in range(world_size):
        source_rank = (rank - step) % world_size
        works: list[dist.Work] = []
        if step + 1 < world_size:
            received_key = torch.empty_like(current_key)
            received_value = torch.empty_like(current_value)
            works.extend(
                [
                    dist.irecv(received_key, src=previous_rank, group=process_group),
                    dist.irecv(received_value, src=previous_rank, group=process_group),
                    dist.isend(current_key, dst=next_rank, group=process_group),
                    dist.isend(current_value, dst=next_rank, group=process_group),
                ]
            )

        allowed = None
        if causal:
            allowed = causal_block_mask(
                rank * local_length,
                source_rank * local_length,
                local_length,
                local_length,
                query.device,
            )
        state = absorb_attention_block(
            query, current_key, current_value, state, allowed=allowed
        )

        for work in works:
            work.wait()
        if step + 1 < world_size:
            current_key, current_value = received_key, received_value

    return state.output()
