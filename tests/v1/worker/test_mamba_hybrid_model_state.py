# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


def test_prepare_attn_forwards_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=0)
    state.max_model_len = 8192
    state._align_mode = False
    state._use_replayssm = False
    state.recoverssm = None

    positions = torch.tensor([1536], dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        num_reqs_after_padding=1,
        num_tokens_after_padding=1,
        query_start_loc_np=torch.tensor([0, 1], dtype=torch.int32).numpy(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        num_scheduled_tokens=torch.tensor([1], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([1537], dtype=torch.int32),
        seq_lens=torch.tensor([1537], dtype=torch.int32),
        is_prefilling_np=torch.tensor([False]).numpy(),
        dcp_local_seq_lens=None,
        positions=positions,
        prompt_lens=torch.tensor([1024], dtype=torch.int32),
    )
    expected_metadata = {"layer": object()}
    build_attn_metadata = Mock(return_value=expected_metadata)
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)

    metadata = state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )

    assert metadata is expected_metadata
    assert build_attn_metadata.call_args.kwargs["positions"] is positions


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    group = SimpleNamespace(layer_names=["layer"])

    state.record_step({"layer": metadata}, [[group]], for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once()
    (committed_counts,) = metadata.commit_recoverssm_state.call_args.args
    torch.testing.assert_close(committed_counts, num_sampled)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_recoverssm_all_groups_see_same_accepted_token_snapshot() -> None:
    """One group's align reset must not shorten later groups' commits."""
    state = RecoverSSMState()
    seen: list[torch.Tensor] = []

    def commit(value: torch.Tensor) -> RecoverSSMPostprocessMetadata:
        seen.append(value.clone())
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=1,
            request_indices=None,
            num_computed_tokens=torch.tensor([7], dtype=torch.int32, device="cuda"),
            block_size=8,
            block_table=torch.zeros((1, 4), dtype=torch.int32, device="cuda"),
            reset_mask=torch.ones(1, dtype=torch.bool, device="cuda"),
        )

    first = Mock(spec=RecoverSSMMetadata)
    second = Mock(spec=RecoverSSMMetadata)
    first.commit_recoverssm_state.side_effect = commit
    second.commit_recoverssm_state.side_effect = commit
    groups = [
        [SimpleNamespace(layer_names=["gdn.0"])],
        [SimpleNamespace(layer_names=["gdn.1"])],
    ]
    state.record_step({"gdn.0": first, "gdn.1": second}, groups, for_capture=False)

    # Match GPUModelRunner: ``num_sampled`` is a view into the exact buffer
    # passed as ``num_accepted_tokens`` below.
    num_accepted = torch.tensor([3], dtype=torch.int32, device="cuda")
    num_sampled = num_accepted[:1]
    state_indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    state.commit_step(
        num_sampled,
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        state_indices=state_indices,
        num_accepted_tokens=num_accepted,
    )

    assert [value.item() for value in seen] == [3, 3]
    # The optimistic running page is preserved; both groups only consume the
    # immutable accepted-count snapshot and mark the row as materialized.
    assert state_indices.item() == 0
    assert num_accepted.item() == -1


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_recoverssm_snapshot_handles_mixed_acceptance_across_groups() -> None:
    """Mixed accepted lengths must stay immutable for every cache group."""
    state = RecoverSSMState()
    seen: list[torch.Tensor] = []
    num_computed = torch.tensor([7, 15, 7, 15], dtype=torch.int32, device="cuda")
    reset_mask = torch.ones(4, dtype=torch.bool, device="cuda")

    def commit(value: torch.Tensor) -> RecoverSSMPostprocessMetadata:
        seen.append(value.clone())
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=4,
            request_indices=None,
            num_computed_tokens=num_computed,
            block_size=8,
            block_table=torch.zeros((4, 4), dtype=torch.int32, device="cuda"),
            reset_mask=reset_mask,
        )

    metadata = [Mock(spec=RecoverSSMMetadata) for _ in range(3)]
    for item in metadata:
        item.commit_recoverssm_state.side_effect = commit
    groups = [
        [SimpleNamespace(layer_names=[f"gdn.{idx}"])] for idx in range(len(metadata))
    ]
    state.record_step(
        {f"gdn.{idx}": item for idx, item in enumerate(metadata)},
        groups,
        for_capture=False,
    )

    num_accepted = torch.tensor([1, 2, 3, 2], dtype=torch.int32, device="cuda")
    state_indices = torch.zeros(4, dtype=torch.int32, device="cuda")
    state.commit_step(
        num_accepted[:],
        torch.arange(4, dtype=torch.int32, device="cuda"),
        state_indices=state_indices,
        num_accepted_tokens=num_accepted,
    )

    expected_counts = [1, 2, 3, 2]
    assert [value.tolist() for value in seen] == [expected_counts] * 3
    assert state_indices.tolist() == [0, 0, 0, 0]
    assert num_accepted.tolist() == [-1, -1, -1, -1]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias() -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor([6, 7], dtype=torch.int32, device="cuda"),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
    )
    num_sampled = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    group = SimpleNamespace(layer_names=["layer"])

    state.recoverssm.record_step({"layer": metadata}, [[group]], for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, -1, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    # This fixture intentionally omits the generic align postprocess that
    # consumes RecoverSSM's private -1 sentinel and restores the public value
    # to one.
    expected_accepted = [9, -1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted
