# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-boundary recovery for speculative GDN ReplaySSM."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.layers.mamba.ops.replayssm_config import (
    get_replayssm_config,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.heuristics(
    {"HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None}
)
@triton.jit
def _prepare_recovery_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    write_pos_ptr,
    cache_base_ptr,
    is_flush_ptr,
    commit_lens_ptr,
    recovery_base_ptr,
    final_recovery_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    boundary_conv_lens_ptr,
    next_state_indices_ptr,
    materialize_mask_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    RING_LEN: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(
        tl.int64
    )
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(
            request_indices_ptr + spec_idx * stride_request_indices
        ).to(tl.int64)

    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted).to(
        tl.int32
    )
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)

    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    write_pos = tl.load(write_pos_ptr + source_state_idx, mask=valid, other=0).to(
        tl.int32
    )
    cache_base = tl.load(cache_base_ptr + source_state_idx, mask=valid, other=0).to(
        tl.int32
    )
    was_flushed = tl.load(is_flush_ptr + source_state_idx, mask=valid, other=0).to(
        tl.int32
    )

    prior_recovery_len = tl.where(was_flushed != 0, 0, write_pos)
    recovery_base = tl.where(
        was_flushed != 0,
        (cache_base + write_pos) & (RING_LEN - 1),
        cache_base,
    )
    final_recovery_len = prior_recovery_len + commit_len

    # The metadata snapshot is computed before the forward as
    # seq_lens - query_lens, in batch order.
    num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(
        tl.int32
    )
    final_num_computed = num_computed + commit_len
    current_col = num_computed // mamba_block_size
    next_boundary = (current_col + 1) * mamba_block_size
    crosses_boundary = final_num_computed >= next_boundary
    # Align mode copies only the checkpoint/conv state, not ReplaySSM's private
    # ring. Materialize before the first speculative window can select the next
    # page so that the subsequent pre-copy starts from a self-contained state.
    in_boundary_guard = (final_num_computed < next_boundary) & (
        (next_boundary - final_num_computed) < SPEC_QUERY_LEN
    )
    materialize = valid & (crosses_boundary | in_boundary_guard)

    # Mamba state column ``p`` represents the recurrent state after at most
    # ``(p + 1) * block_size`` tokens.  In particular, an exact boundary still
    # belongs to the completed page; the next page is selected only after the
    # first token beyond that boundary.  Keep this identical to
    # compute_aligned_state_indices()'s ceil-div-minus-one invariant.
    final_col = tl.minimum(
        (tl.maximum(final_num_computed, 1) - 1) // mamba_block_size,
        block_table_width - 1,
    )
    final_state_idx = tl.load(
        block_table_ptr
        + request_idx * stride_block_table_row
        + final_col * stride_block_table_col,
        mask=materialize,
        other=null_block_id,
    ).to(tl.int64)

    boundary_col = next_boundary // mamba_block_size - 1
    boundary_state_idx = tl.load(
        block_table_ptr
        + request_idx * stride_block_table_row
        + boundary_col * stride_block_table_col,
        mask=materialize & crosses_boundary & (boundary_col < block_table_width),
        other=null_block_id,
    ).to(tl.int64)
    boundary_recovery_len = prior_recovery_len + next_boundary - num_computed

    next_col = next_boundary // mamba_block_size
    next_state_idx = tl.load(
        block_table_ptr
        + request_idx * stride_block_table_row
        + next_col * stride_block_table_col,
        mask=materialize & (next_col < block_table_width),
        other=null_block_id,
    ).to(tl.int64)

    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    tl.store(recovery_base_ptr + spec_idx, recovery_base)
    tl.store(
        final_recovery_lens_ptr + spec_idx,
        tl.where(materialize, final_recovery_len, 0),
    )
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(materialize, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(materialize & crosses_boundary, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_recovery_lens_ptr + spec_idx,
        tl.where(materialize & crosses_boundary, boundary_recovery_len, 0),
    )
    tl.store(
        boundary_conv_lens_ptr + spec_idx,
        tl.where(materialize & crosses_boundary, next_boundary - num_computed, 0),
    )
    tl.store(
        next_state_indices_ptr + spec_idx,
        tl.where(materialize, next_state_idx, null_block_id),
    )
    tl.store(materialize_mask_ptr + spec_idx, materialize)


@triton.jit
def _compact_conv_state_kernel(
    conv_state_ref_ptr,
    conv_state_base_addrs_ptr,
    conv_state_block_strides_ptr,
    conv_state_dim_strides_ptr,
    conv_state_token_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_conv_lens_ptr,
    materialize_mask_ptr,
    null_block_id,
    conv_dim,
    conv_history_len,
    stride_state_indices,
    BLOCK_D: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_l = tl.program_id(2)
    if tl.load(materialize_mask_ptr + pid_b) == 0:
        return

    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
        tl.int64
    )
    if source_state_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_conv_len = tl.load(boundary_conv_lens_ptr + pid_b)

    base_addr = tl.load(conv_state_base_addrs_ptr + pid_l)
    block_stride = tl.load(conv_state_block_strides_ptr + pid_l)
    dim_stride = tl.load(conv_state_dim_strides_ptr + pid_l)
    token_stride = tl.load(conv_state_token_strides_ptr + pid_l)
    conv_state_ptr = base_addr.to(tl.pointer_type(conv_state_ref_ptr.dtype.element_ty))
    source_ptr = conv_state_ptr + source_state_idx * block_stride

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_HISTORY)
    mask = (offs_d[:, None] < conv_dim) & (offs_h[None, :] < conv_history_len)
    final_values = tl.load(
        source_ptr
        + offs_d[:, None] * dim_stride
        + (commit_len - 1 + offs_h[None, :]) * token_stride,
        mask=mask,
    )
    if boundary_state_idx > null_block_id:
        boundary_values = tl.load(
            source_ptr
            + offs_d[:, None] * dim_stride
            + (boundary_conv_len - 1 + offs_h[None, :]) * token_stride,
            mask=mask,
        )
        boundary_ptr = conv_state_ptr + boundary_state_idx * block_stride
        tl.store(
            boundary_ptr
            + offs_d[:, None] * dim_stride
            + offs_h[None, :] * token_stride,
            boundary_values,
            mask=mask,
        )
    final_ptr = conv_state_ptr + final_state_idx * block_stride
    tl.store(
        final_ptr + offs_d[:, None] * dim_stride + offs_h[None, :] * token_stride,
        final_values,
        mask=mask & (final_state_idx > null_block_id),
    )
    # Align-mode preprocessing chooses the running page from the whole
    # speculative window, before acceptance is known.  That optimistic page is
    # ``source_state_idx`` and remains the runner's page on the next step.  If
    # acceptance stops before (or exactly at) the page boundary,
    # ``final_state_idx`` can point at the preceding logical page.  Keep a
    # compact copy in both places: ``final_state_idx`` is the prefix-cache
    # checkpoint, while ``source_state_idx`` is the live running state.
    tl.store(
        source_ptr + offs_d[:, None] * dim_stride + offs_h[None, :] * token_stride,
        final_values,
        mask=mask,
    )


@triton.jit
def _materialize_gdn_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    d_ref_ptr,
    d_base_addrs_ptr,
    d_block_strides_ptr,
    k_ref_ptr,
    k_base_addrs_ptr,
    k_block_strides_ptr,
    g_ref_ptr,
    g_base_addrs_ptr,
    g_block_strides_ptr,
    state_indices_ptr,
    recovery_base_ptr,
    final_recovery_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    materialize_mask_ptr,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_d_head,
    stride_d_pos,
    stride_d_v,
    stride_k_head,
    stride_k_pos,
    stride_k_k,
    stride_g_head,
    stride_g_pos,
    stride_state_indices,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    NK: tl.constexpr,
    BKT: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    RING_LEN: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    if tl.load(materialize_mask_ptr + pid_b) == 0:
        return

    pid_l = pid_lh // HV
    pid_hv = pid_lh % HV
    pid_h = pid_hv // (HV // H)
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
        tl.int64
    )
    if source_state_idx <= null_block_id:
        return
    recovery_base = tl.load(recovery_base_ptr + pid_b).to(tl.int32)
    final_recovery_len = tl.load(final_recovery_lens_ptr + pid_b).to(tl.int32)
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b).to(tl.int32)

    state_base = tl.load(state_base_addrs_ptr + pid_l)
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    state_ptr = state_base.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_ptr = (
        state_ptr + source_state_idx * state_block_stride + pid_hv * stride_state_head
    )
    d_base = tl.load(d_base_addrs_ptr + pid_l)
    d_block_stride = tl.load(d_block_strides_ptr + pid_l)
    d_ptr = d_base.to(tl.pointer_type(d_ref_ptr.dtype.element_ty))
    d_ptr += source_state_idx * d_block_stride + pid_hv * stride_d_head
    k_base = tl.load(k_base_addrs_ptr + pid_l)
    k_block_stride = tl.load(k_block_strides_ptr + pid_l)
    k_ptr = k_base.to(tl.pointer_type(k_ref_ptr.dtype.element_ty))
    k_ptr += source_state_idx * k_block_stride + pid_h * stride_k_head
    g_base = tl.load(g_base_addrs_ptr + pid_l)
    g_block_stride = tl.load(g_block_strides_ptr + pid_l)
    g_ptr = g_base.to(tl.pointer_type(g_ref_ptr.dtype.element_ty))
    g_ptr += source_state_idx * g_block_stride + pid_hv * stride_g_head

    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_c = tl.arange(0, BC)
    mask_v = offs_v < V
    physical_pos = (recovery_base + offs_c) & (RING_LEN - 1)

    final_valid = offs_c < final_recovery_len
    final_g = tl.load(
        g_ptr + physical_pos * stride_g_pos, mask=final_valid, other=0.0
    ).to(tl.float32)
    final_g_prefix = tl.cumsum(final_g, axis=0)
    final_g_total = tl.sum(final_g, axis=0)
    final_decay = tl.where(final_valid, tl.exp(final_g_total - final_g_prefix), 0.0)
    final_d = tl.load(
        d_ptr + physical_pos[None, :] * stride_d_pos + offs_v[:, None] * stride_d_v,
        mask=mask_v[:, None] & final_valid[None, :],
        other=0.0,
    ).to(tl.float32)

    has_boundary = boundary_state_idx > null_block_id
    boundary_valid = (offs_c < boundary_recovery_len) & has_boundary
    boundary_g = tl.load(
        g_ptr + physical_pos * stride_g_pos,
        mask=boundary_valid,
        other=0.0,
    ).to(tl.float32)
    boundary_g_prefix = tl.cumsum(boundary_g, axis=0)
    boundary_g_total = tl.sum(boundary_g, axis=0)
    boundary_decay = tl.where(
        boundary_valid,
        tl.exp(boundary_g_total - boundary_g_prefix),
        0.0,
    )
    boundary_d = tl.load(
        d_ptr + physical_pos[None, :] * stride_d_pos + offs_v[:, None] * stride_d_v,
        mask=mask_v[:, None] & boundary_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    # Match the normal ReplaySSM flush kernel's K tiling.  Besides reducing
    # register pressure, keeping the same tl.dot shapes minimizes numerical
    # drift when a checkpoint is materialized solely for prefix caching.
    for kk in range(NK):
        offs_k = kk * BKT + tl.arange(0, BKT)
        mask_k = offs_k < K
        mask_state = mask_v[:, None] & mask_k[None, :]
        source_state_ptrs = (
            source_ptr
            + offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
        )
        initial_state = tl.load(source_state_ptrs, mask=mask_state, other=0.0).to(
            tl.float32
        )
        final_k = tl.load(
            k_ptr + physical_pos[:, None] * stride_k_pos + offs_k[None, :] * stride_k_k,
            mask=final_valid[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        final_state = tl.dot(
            final_d * final_decay[None, :],
            final_k,
            acc=tl.exp(final_g_total) * initial_state,
            input_precision="tf32",
        )

        boundary_k = tl.load(
            k_ptr + physical_pos[:, None] * stride_k_pos + offs_k[None, :] * stride_k_k,
            mask=boundary_valid[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        boundary_state = tl.dot(
            boundary_d * boundary_decay[None, :],
            boundary_k,
            acc=tl.exp(boundary_g_total) * initial_state,
            input_precision="tf32",
        )
        boundary_ptr = (
            state_ptr
            + boundary_state_idx * state_block_stride
            + pid_hv * stride_state_head
            + offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
        )
        tl.store(boundary_ptr, boundary_state, mask=mask_state & has_boundary)

        final_ptrs = (
            state_ptr
            + final_state_idx * state_block_stride
            + pid_hv * stride_state_head
            + offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
        )
        tl.store(
            final_ptrs,
            final_state,
            mask=mask_state & (final_state_idx > null_block_id),
        )
        # See the conv-state copy above.  The source page is the optimistic
        # running page selected before sampling.  Preserve the accepted state
        # there as well as in the logical prefix checkpoint so the next decode
        # step and prefix restoration observe the same recurrence state.
        tl.store(source_state_ptrs, final_state, mask=mask_state)


@triton.jit
def _reset_materialized_cursors_kernel(
    write_pos_ptr,
    cache_base_ptr,
    is_flush_ptr,
    state_indices_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    next_state_indices_ptr,
    materialize_mask_ptr,
    n_rows,
    stride_state_indices,
    null_block_id,
    INIT_FLUSH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    row_mask = rows < n_rows
    materialize = tl.load(materialize_mask_ptr + rows, mask=row_mask, other=0).to(
        tl.int1
    )
    source = tl.load(
        state_indices_ptr + rows * stride_state_indices,
        mask=row_mask,
        other=null_block_id,
    ).to(tl.int64)
    final_state = tl.load(
        final_state_indices_ptr + rows,
        mask=row_mask,
        other=null_block_id,
    ).to(tl.int64)
    boundary = tl.load(
        boundary_state_indices_ptr + rows,
        mask=row_mask,
        other=null_block_id,
    ).to(tl.int64)
    next_state = tl.load(
        next_state_indices_ptr + rows,
        mask=row_mask,
        other=null_block_id,
    ).to(tl.int64)
    valid_source = row_mask & materialize & (source > null_block_id)
    tl.store(write_pos_ptr + source, 0, mask=valid_source)
    tl.store(cache_base_ptr + source, 0, mask=valid_source)
    tl.store(is_flush_ptr + source, INIT_FLUSH, mask=valid_source)

    valid_final = row_mask & materialize & (final_state > null_block_id)
    tl.store(write_pos_ptr + final_state, 0, mask=valid_final)
    tl.store(cache_base_ptr + final_state, 0, mask=valid_final)
    tl.store(is_flush_ptr + final_state, INIT_FLUSH, mask=valid_final)

    valid_boundary = row_mask & materialize & (boundary > null_block_id)
    tl.store(write_pos_ptr + boundary, 0, mask=valid_boundary)
    tl.store(cache_base_ptr + boundary, 0, mask=valid_boundary)
    tl.store(is_flush_ptr + boundary, INIT_FLUSH, mask=valid_boundary)

    valid_next = row_mask & materialize & (next_state > null_block_id)
    tl.store(write_pos_ptr + next_state, 0, mask=valid_next)
    tl.store(cache_base_ptr + next_state, 0, mask=valid_next)
    tl.store(is_flush_ptr + next_state, INIT_FLUSH, mask=valid_next)


@dataclass
class GDNReplaySSMPrefixCommitContext:
    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    checkpoints: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    d_caches: tuple[torch.Tensor, ...]
    d_base_addrs: torch.Tensor
    d_block_strides: torch.Tensor
    k_caches: tuple[torch.Tensor, ...]
    k_base_addrs: torch.Tensor
    k_block_strides: torch.Tensor
    g_caches: tuple[torch.Tensor, ...]
    g_base_addrs: torch.Tensor
    g_block_strides: torch.Tensor
    write_pos: torch.Tensor
    cache_base: torch.Tensor
    is_flush: torch.Tensor
    commit_lens: torch.Tensor
    recovery_base: torch.Tensor
    final_recovery_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    boundary_conv_lens: torch.Tensor
    next_state_indices: torch.Tensor
    materialize_mask: torch.Tensor
    spec_query_len: int
    max_cache_len: int

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        write_pos: torch.Tensor,
        cache_base: torch.Tensor,
        is_flush: torch.Tensor,
        spec_query_len: int,
        max_num_reqs: int,
        max_cache_len: int,
    ) -> "GDNReplaySSMPrefixCommitContext":
        if not layers:
            raise ValueError("GDN ReplaySSM prefix commit requires at least one layer")
        if any(len(layer.kv_cache) != 5 for layer in layers):
            raise ValueError(
                "GDN ReplaySSM pages must contain conv, checkpoint, d, k, and g"
            )

        conv_states = [layer.kv_cache[0] for layer in layers]
        if not is_conv_state_dim_first():
            conv_states = [state.transpose(-1, -2) for state in conv_states]
        checkpoints = [layer.kv_cache[1] for layer in layers]
        d_caches = [layer.kv_cache[2] for layer in layers]
        k_caches = [layer.kv_cache[3] for layer in layers]
        g_caches = [layer.kv_cache[4] for layer in layers]

        state_ref = checkpoints[0]
        d_ref = d_caches[0]
        k_ref = k_caches[0]
        g_ref = g_caches[0]
        if state_ref.ndim != 4 or d_ref.ndim != 4 or k_ref.ndim != 4:
            raise ValueError("GDN ReplaySSM recurrent cache layouts are invalid")
        if g_ref.ndim != 3:
            raise ValueError("GDN ReplaySSM gate cache layout is invalid")
        num_blocks, num_value_heads, value_dim, key_dim = state_ref.shape
        ring_len = d_ref.shape[2]
        num_key_heads = k_ref.shape[1]
        expected_d = (num_blocks, num_value_heads, ring_len, value_dim)
        expected_k = (num_blocks, num_key_heads, ring_len, key_dim)
        expected_g = (num_blocks, num_value_heads, ring_len)
        if ring_len & (ring_len - 1):
            raise ValueError("GDN ReplaySSM ring length must be a power of two")
        if max_cache_len > ring_len:
            raise ValueError("GDN ReplaySSM logical cache exceeds the physical ring")

        def _validate_layouts(
            tensors: list[torch.Tensor], ref: torch.Tensor, shape: tuple[int, ...]
        ) -> None:
            for tensor in tensors:
                if (
                    tensor.shape != shape
                    or tensor.dtype != ref.dtype
                    or tensor.device != ref.device
                    or tensor.stride()[1:] != ref.stride()[1:]
                ):
                    raise ValueError("GDN ReplaySSM layers need matching cache layouts")

        _validate_layouts(checkpoints, state_ref, tuple(state_ref.shape))
        _validate_layouts(d_caches, d_ref, expected_d)
        _validate_layouts(k_caches, k_ref, expected_k)
        _validate_layouts(g_caches, g_ref, expected_g)

        conv_ref = conv_states[0]
        if conv_ref.ndim != 3:
            raise ValueError("GDN ReplaySSM conv state must be three-dimensional")
        conv_dim, conv_state_len = conv_ref.shape[1:]
        conv_history_len = conv_state_len - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("GDN ReplaySSM conv state is shorter than its window")
        for state in conv_states:
            if (
                state.shape != conv_ref.shape
                or state.dtype != conv_ref.dtype
                or state.device != state_ref.device
                or state.shape[0] != num_blocks
            ):
                raise ValueError("GDN ReplaySSM layers need matching conv layouts")

        device = state_ref.device

        def _base_addrs(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensor.data_ptr() for tensor in tensors],
                dtype=torch.int64,
                device=device,
            )

        def _block_strides(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensor.stride(0) for tensor in tensors],
                dtype=torch.int64,
                device=device,
            )

        if any(
            tensor.device != device or tensor.shape[0] != num_blocks
            for tensor in (write_pos, cache_base, is_flush)
        ):
            raise ValueError("GDN ReplaySSM cursor buffers are incompatible")

        return cls(
            conv_states=tuple(conv_states),
            conv_state_base_addrs=_base_addrs(conv_states),
            conv_state_block_strides=_block_strides(conv_states),
            conv_state_dim_strides=torch.tensor(
                [state.stride(1) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_state_token_strides=torch.tensor(
                [state.stride(2) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_history_len=conv_history_len,
            checkpoints=tuple(checkpoints),
            state_base_addrs=_base_addrs(checkpoints),
            state_block_strides=_block_strides(checkpoints),
            d_caches=tuple(d_caches),
            d_base_addrs=_base_addrs(d_caches),
            d_block_strides=_block_strides(d_caches),
            k_caches=tuple(k_caches),
            k_base_addrs=_base_addrs(k_caches),
            k_block_strides=_block_strides(k_caches),
            g_caches=tuple(g_caches),
            g_base_addrs=_base_addrs(g_caches),
            g_block_strides=_block_strides(g_caches),
            write_pos=write_pos,
            cache_base=cache_base,
            is_flush=is_flush,
            commit_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            recovery_base=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            final_recovery_lens=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            final_state_indices=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            boundary_state_indices=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            boundary_recovery_lens=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            boundary_conv_lens=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            next_state_indices=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            materialize_mask=torch.empty(max_num_reqs, dtype=torch.bool, device=device),
            spec_query_len=spec_query_len,
            max_cache_len=max_cache_len,
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        *,
        request_indices: torch.Tensor | None,
        block_table: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        mamba_block_size: int,
    ) -> torch.Tensor:
        """Advance the ring and materialize states needed by prefix caching.

        ``num_computed_tokens`` is the pre-forward, batch-ordered metadata
        snapshot (``seq_lens - query_lens``).
        """
        batch = state_indices.shape[0]
        if batch == 0:
            return self.materialize_mask[:0]
        if batch > self.commit_lens.shape[0]:
            raise ValueError("GDN ReplaySSM commit batch exceeds its capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("GDN ReplaySSM commit metadata is incompatible")
        if request_indices is not None and request_indices.shape[0] < batch:
            raise ValueError("GDN ReplaySSM request mapping is too short")
        if block_table.ndim != 2:
            raise ValueError("GDN ReplaySSM block table must be two-dimensional")
        device = self.checkpoints[0].device
        inputs = (
            num_accepted_tokens,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
        )
        if any(tensor.device != device for tensor in inputs) or (
            request_indices is not None and request_indices.device != device
        ):
            raise ValueError("GDN ReplaySSM commit inputs must share a device")

        block_table_stride = block_table.stride()
        _prepare_recovery_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.write_pos,
            self.cache_base,
            self.is_flush,
            self.commit_lens,
            self.recovery_base,
            self.final_recovery_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            self.boundary_conv_lens,
            self.next_state_indices,
            self.materialize_mask,
            NULL_BLOCK_ID,
            mamba_block_size,
            block_table.shape[1],
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            block_table_stride[0],
            block_table_stride[1],
            num_computed_tokens.stride(0),
            RING_LEN=self.d_caches[0].shape[2],
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )

        from vllm.third_party.flash_linear_attention.ops.gdn_replayssm_spec_decode import (  # noqa: E501
            commit_gdn_replayssm_spec,
        )

        commit_gdn_replayssm_spec(
            self.write_pos,
            self.cache_base,
            self.is_flush,
            self.commit_lens[:batch],
            state_indices,
            max_cache_len=self.max_cache_len,
            max_spec_len=self.spec_query_len,
            cache_buf_len=self.d_caches[0].shape[2],
        )

        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        _compact_conv_state_kernel[
            (triton.cdiv(conv_dim, 256), batch, len(self.conv_states))
        ](
            conv_ref,
            self.conv_state_base_addrs,
            self.conv_state_block_strides,
            self.conv_state_dim_strides,
            self.conv_state_token_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_conv_lens,
            self.materialize_mask,
            NULL_BLOCK_ID,
            conv_dim,
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=256,
            BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
            num_warps=4,
        )

        state_ref = self.checkpoints[0]
        _, num_value_heads, value_dim, key_dim = state_ref.shape
        num_key_heads = self.k_caches[0].shape[1]
        block_k = triton.next_power_of_2(key_dim)
        block_v, num_warps, configured_k_tiles, num_stages = get_replayssm_config(
            "gdn_spec_flush",
            max_spec_len=self.spec_query_len,
            head_k_dim=key_dim,
        )
        num_k_tiles = min(configured_k_tiles, max(1, block_k // 16))
        if block_k % num_k_tiles != 0:
            raise ValueError(
                f"ReplaySSM flush K tiling requires nk={num_k_tiles} to divide "
                f"BK={block_k}."
            )
        block_k_tile = block_k // num_k_tiles
        if block_k_tile < 16:
            raise ValueError(
                f"ReplaySSM flush K tile must be at least 16, got {block_k_tile}."
            )
        ring_len = self.d_caches[0].shape[2]
        grid = (
            triton.cdiv(value_dim, block_v),
            batch,
            len(self.checkpoints) * num_value_heads,
        )
        _materialize_gdn_state_kernel[grid](
            state_ref,
            self.state_base_addrs,
            self.state_block_strides,
            self.d_caches[0],
            self.d_base_addrs,
            self.d_block_strides,
            self.k_caches[0],
            self.k_base_addrs,
            self.k_block_strides,
            self.g_caches[0],
            self.g_base_addrs,
            self.g_block_strides,
            state_indices,
            self.recovery_base,
            self.final_recovery_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            self.materialize_mask,
            NULL_BLOCK_ID,
            state_ref.stride(1),
            state_ref.stride(2),
            state_ref.stride(3),
            self.d_caches[0].stride(1),
            self.d_caches[0].stride(2),
            self.d_caches[0].stride(3),
            self.k_caches[0].stride(1),
            self.k_caches[0].stride(2),
            self.k_caches[0].stride(3),
            self.g_caches[0].stride(1),
            self.g_caches[0].stride(2),
            state_indices.stride(0),
            H=num_key_heads,
            HV=num_value_heads,
            K=key_dim,
            V=value_dim,
            BK=block_k,
            NK=num_k_tiles,
            BKT=block_k_tile,
            BV=block_v,
            BC=max(
                16,
                triton.next_power_of_2(
                    max(1, self.max_cache_len - self.spec_query_len)
                ),
            ),
            RING_LEN=ring_len,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        block = triton.next_power_of_2(max(1, batch))
        _reset_materialized_cursors_kernel[(1,)](
            self.write_pos,
            self.cache_base,
            self.is_flush,
            state_indices,
            self.final_state_indices,
            self.boundary_state_indices,
            self.next_state_indices,
            self.materialize_mask,
            batch,
            state_indices.stride(0),
            NULL_BLOCK_ID,
            INIT_FLUSH=1 if 2 * self.spec_query_len > self.max_cache_len else 0,
            BLOCK=block,
            num_warps=1,
        )
        return self.materialize_mask[:batch]


__all__ = ["GDNReplaySSMPrefixCommitContext"]
