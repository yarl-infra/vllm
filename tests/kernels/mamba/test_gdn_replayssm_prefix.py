# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.gdn_replayssm_prefix import (
    GDNReplaySSMPrefixCommitContext,
)


def _serial_state(
    initial: torch.Tensor,
    d: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    length: int,
) -> torch.Tensor:
    result = initial.float().clone()
    value_heads_per_key_head = result.shape[0] // k.shape[0]
    for pos in range(length):
        result.mul_(torch.exp(g[:, pos]).unsqueeze(-1).unsqueeze(-1))
        for value_head in range(result.shape[0]):
            key_head = value_head // value_heads_per_key_head
            result[value_head].add_(
                d[value_head, pos].float().unsqueeze(1) * k[key_head, pos].float()
            )
    return result


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_materialize_keeps_exact_boundary_state_in_optimistic_running_page() -> None:
    """Acceptance can end one page behind the preselected running page.

    Align preprocessing chooses a page from the whole speculative window.  If
    sampling accepts exactly to the preceding page boundary, the accepted
    prefix checkpoint and the live running page are distinct physical blocks.
    RecoverSSM must populate both before it clears the private replay ring.
    """
    torch.manual_seed(7)
    device = torch.device("cuda")
    blocks, value_heads, key_heads = 5, 2, 1
    value_dim = key_dim = 16
    ring_len, spec_len, history_len = 8, 3, 4
    conv_len = history_len + spec_len - 1

    # Default SD layout: [block, state_len, dim].  The prefix context exposes
    # a transposed [block, dim, state_len] view to its compaction kernel.
    conv = torch.randn(blocks, conv_len, 12, device=device, dtype=torch.bfloat16)
    state = torch.randn(
        blocks,
        value_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    )
    d_cache = (
        torch.randn(
            blocks,
            value_heads,
            ring_len,
            value_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.03
    )
    k_cache = (
        torch.randn(
            blocks,
            key_heads,
            ring_len,
            key_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.03
    )
    g_cache = (
        -torch.rand(
            blocks,
            value_heads,
            ring_len,
            device=device,
            dtype=torch.float32,
        )
        * 0.05
    )

    optimistic_running, prefix_boundary = 2, 1
    prior_len, accepted = 3, 1
    initial = state[optimistic_running].clone()
    expected_state = _serial_state(
        initial,
        d_cache[optimistic_running].clone(),
        k_cache[optimistic_running].clone(),
        g_cache[optimistic_running].clone(),
        prior_len + accepted,
    )
    expected_conv = conv[optimistic_running, :history_len].clone()

    write_pos = torch.zeros(blocks, dtype=torch.int32, device=device)
    cache_base = torch.zeros_like(write_pos)
    is_flush = torch.zeros(blocks, dtype=torch.int8, device=device)
    write_pos[optimistic_running] = prior_len
    context = GDNReplaySSMPrefixCommitContext.create(
        [SimpleNamespace(kv_cache=[conv, state, d_cache, k_cache, g_cache])],
        write_pos=write_pos,
        cache_base=cache_base,
        is_flush=is_flush,
        spec_query_len=spec_len,
        max_num_reqs=1,
        max_cache_len=ring_len,
    )

    # The full three-token verify window selected logical page 1, but accepting
    # one token from num_computed=15 lands exactly at the end of logical page 0.
    materialized = context.commit(
        torch.tensor([accepted], dtype=torch.int32, device=device),
        torch.tensor([optimistic_running], dtype=torch.int32, device=device),
        torch.tensor([0, spec_len], dtype=torch.int32, device=device),
        request_indices=None,
        block_table=torch.tensor(
            [[prefix_boundary, optimistic_running, 3]],
            dtype=torch.int32,
            device=device,
        ),
        num_computed_tokens=torch.tensor([15], dtype=torch.int32, device=device),
        mamba_block_size=16,
    )
    torch.cuda.synchronize()

    assert materialized.item()
    torch.testing.assert_close(
        state[prefix_boundary], expected_state, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        state[optimistic_running], expected_state, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(conv[prefix_boundary, :history_len], expected_conv)
    torch.testing.assert_close(conv[optimistic_running, :history_len], expected_conv)
    assert write_pos[optimistic_running].item() == 0


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_guard_does_not_reset_unallocated_stale_next_page() -> None:
    """A T=1 guard must not touch a next page that was not scheduled."""
    device = torch.device("cuda")
    blocks, value_heads, key_heads = 6, 2, 1
    value_dim = key_dim = 16
    ring_len, spec_len, history_len = 8, 3, 4
    conv_len = history_len + spec_len - 1

    conv = torch.zeros(
        blocks, conv_len, 12, device=device, dtype=torch.bfloat16
    )
    state = torch.zeros(
        blocks,
        value_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    )
    d_cache = torch.zeros(
        blocks,
        value_heads,
        ring_len,
        value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k_cache = torch.zeros(
        blocks,
        key_heads,
        ring_len,
        key_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    g_cache = torch.zeros(
        blocks, value_heads, ring_len, device=device, dtype=torch.float32
    )

    running, stale_next = 1, 4
    write_pos = torch.zeros(blocks, dtype=torch.int32, device=device)
    cache_base = torch.zeros_like(write_pos)
    is_flush = torch.zeros(blocks, dtype=torch.int8, device=device)
    write_pos[running] = 2
    write_pos[stale_next] = 7
    cache_base[stale_next] = 3
    context = GDNReplaySSMPrefixCommitContext.create(
        [SimpleNamespace(kv_cache=[conv, state, d_cache, k_cache, g_cache])],
        write_pos=write_pos,
        cache_base=cache_base,
        is_flush=is_flush,
        spec_query_len=spec_len,
        max_num_reqs=1,
        max_cache_len=ring_len,
    )

    # At context 14 a one-token draft-less step ends at 15, so logical page 1
    # is not allocated. The second table entry deliberately represents stale
    # row-tail data left by another request.
    context.commit(
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([running], dtype=torch.int32, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        request_indices=None,
        block_table=torch.tensor(
            [[running, stale_next]], dtype=torch.int32, device=device
        ),
        num_computed_tokens=torch.tensor([14], dtype=torch.int32, device=device),
        mamba_block_size=16,
    )
    torch.cuda.synchronize()

    assert write_pos[stale_next].item() == 7
    assert cache_base[stale_next].item() == 3
    assert is_flush[stale_next].item() == 0
