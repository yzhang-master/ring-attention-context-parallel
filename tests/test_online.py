import torch

from ring_attention.online import OnlineAttentionState, absorb_attention_block
from ring_attention.reference import full_attention


def test_online_merge_matches_full_attention_for_causal_and_dense() -> None:
    torch.manual_seed(11)
    query = torch.randn(2, 2, 5, 4, dtype=torch.float32)
    key = torch.randn(2, 2, 10, 4, dtype=torch.float32)
    value = torch.randn_like(key)
    for causal in (False, True):
        state = OnlineAttentionState.empty(query)
        for start in (0, 5):
            allowed = None
            if causal:
                query_positions = torch.arange(5).unsqueeze(1)
                key_positions = start + torch.arange(5).unsqueeze(0)
                allowed = key_positions <= query_positions
            state = absorb_attention_block(
                query,
                key[..., start : start + 5, :],
                value[..., start : start + 5, :],
                state,
                allowed,
            )
        expected = full_attention(query, key, value, causal=causal)
        torch.testing.assert_close(state.output(), expected, atol=2e-6, rtol=2e-6)


def test_fully_masked_block_leaves_empty_state_well_defined() -> None:
    query = torch.randn(1, 1, 2, 4)
    state = OnlineAttentionState.empty(query)
    state = absorb_attention_block(
        query,
        torch.randn(1, 1, 3, 4),
        torch.randn(1, 1, 3, 4),
        state,
        allowed=torch.zeros(2, 3, dtype=torch.bool),
    )
    assert torch.equal(state.output(), torch.zeros_like(query))
    assert torch.isneginf(state.log_normalizer()).all()


def test_online_softmax_rescales_large_scores_without_overflow() -> None:
    query = torch.full((1, 1, 1, 2), 1e10)
    key = torch.full((1, 1, 4, 2), 1e10)
    value = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1).expand(1, 1, 4, 2)
    state = OnlineAttentionState.empty(query)
    for index in range(4):
        state = absorb_attention_block(
            query, key[..., index : index + 1, :], value[..., index : index + 1, :], state
        )
    assert torch.isfinite(state.log_normalizer()).all()
    torch.testing.assert_close(state.output(), value.mean(dim=-2, keepdim=True))
