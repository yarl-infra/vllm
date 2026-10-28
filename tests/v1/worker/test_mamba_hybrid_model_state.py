# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.v1.attention.test_gdn_metadata_builder import (
    BLOCK_SIZE,
    DEVICE,
    _create_gdn_builder,
)
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.utils import AttentionGroup


def _new_mamba_hybrid_state(
    num_speculative_tokens: int = 0, mamba_block_size: int | None = None
) -> MambaHybridModelState:
    config = create_vllm_config(
        max_model_len=8192, max_num_seqs=4, max_num_batched_tokens=32
    )
    config.cache_config.mamba_cache_mode = (
        "align" if mamba_block_size is not None else "none"
    )
    config.cache_config.mamba_block_size = mamba_block_size
    if num_speculative_tokens:
        config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=num_speculative_tokens
        )
    # These tests exercise request and attention metadata, not model forwards.
    return MambaHybridModelState(config, torch.nn.Identity(), None, torch.device("cpu"))


def _recoverssm_attn_groups(layer_name: str) -> list[list[AttentionGroup]]:
    """RecoverSSMState.record_step reads only each group's ``layer_names``."""
    group = SimpleNamespace(layer_names=[layer_name])
    return [[group]]  # type: ignore[list-item]  # layer_names-only stand-in


GPU_DEVICE = current_platform.device_type
HAS_GPU = current_platform.is_cuda_alike() or current_platform.is_xpu()


def test_prepare_attn_forwards_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=0)
    state.model_config = SimpleNamespace(max_model_len=8192)
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
        max_query_len=None,
        seq_lens_cpu_upper_bound=torch.tensor([1537], dtype=torch.int32),
        seq_lens=torch.tensor([1537], dtype=torch.int32),
        is_prefilling_np=torch.tensor([False]).numpy(),
        prefill_runs_as_decode_np=None,
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


def test_add_request_seeds_state_idx_in_mamba_blocks() -> None:
    """A drafter group can lower cache_config.block_size below the mamba
    block size; the seeded column must still be in mamba blocks."""
    state = _new_mamba_hybrid_state(mamba_block_size=880)
    state.add_request(
        1,
        NewRequestData(
            req_id="resumed",
            prompt_token_ids=[1],
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([],),
            num_computed_tokens=107_360,
            lora_request=None,
        ),
    )

    assert state._mamba_state_idx_gpu[1] == 121


def test_padded_prompt_tail_builds_as_spec_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-token prompt tail over prior state, padded with K placeholder
    drafts (e.g. a P/D decode-node arrival), must reach the GDN builder as a
    spec-decode row. Built as a prefill, the placeholder tokens are folded into
    the recurrent state and can't be rolled back.
    """
    k = 3
    state = _new_mamba_hybrid_state(num_speculative_tokens=k)

    # A verify decode, a padded prompt tail (128 of 129 prompt tokens already
    # computed), and a fresh prompt chunk of the same length.
    query_lens = [k + 1] * 3
    seq_lens = [50, 128 + k + 1, k + 1]
    is_prefilling = [False, True, True]
    query_start_loc = np.array([0, 4, 8, 12], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=3,
        num_tokens=12,
        num_reqs_after_padding=3,
        num_tokens_after_padding=12,
        idx_mapping=torch.arange(3),
        query_start_loc_np=query_start_loc,
        query_start_loc=torch.from_numpy(query_start_loc),
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        num_draft_tokens_per_req=np.array([k, k, 0], dtype=np.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        is_prefilling_np=np.array(is_prefilling),
        prefill_runs_as_decode_np=np.array([False, True, False]),
        dcp_local_seq_lens=None,
        positions=torch.zeros(12, dtype=torch.int64),
        prompt_lens=None,
    )
    build_attn_metadata = Mock(return_value={})
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)
    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )
    mamba_metadata = build_attn_metadata.call_args.kwargs[
        "model_specific_attn_metadata"
    ]
    assert mamba_metadata.is_prefilling.tolist() == [False, False, True]

    builder = _create_gdn_builder(num_speculative_tokens=k)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens), BLOCK_SIZE, DEVICE
    ).replace(**mamba_metadata.get_extra_common_attn_kwargs(0, 3))
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=mamba_metadata.num_accepted_tokens,
        num_decode_draft_tokens_cpu=mamba_metadata.num_decode_draft_tokens_cpu,
    )

    # Only the fresh prompt chunk needs the prefill kernels.
    assert meta.num_spec_decodes == 2
    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == k + 1


def _mamba_hybrid_state(num_speculative_tokens: int) -> MambaHybridModelState:
    state = _new_mamba_hybrid_state(num_speculative_tokens=num_speculative_tokens)
    # A request that accepted 3 drafts (plus the bonus token) on the previous
    # step; the others have never had a draft accepted.
    state.num_accepted_tokens_gpu[0] = 4
    return state


def _input_batch(
    num_scheduled_tokens: list[int],
    num_draft_tokens_per_req: list[int] | None,
    is_prefilling: list[bool],
    num_reqs_after_padding: int | None = None,
) -> SimpleNamespace:
    """`num_draft_tokens_per_req=None` is a batch where nobody proposed a
    draft this step, e.g. because ngram found no match for anyone."""
    num_reqs = len(num_scheduled_tokens)
    query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
    query_start_loc[1:] = np.cumsum(num_scheduled_tokens)
    num_tokens = int(query_start_loc[-1])
    seq_lens = torch.full((num_reqs,), 10, dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs_after_padding or num_reqs,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        query_start_loc_np=query_start_loc,
        num_scheduled_tokens=np.array(num_scheduled_tokens, dtype=np.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=seq_lens,
        is_prefilling_np=np.array(is_prefilling),
        prefill_runs_as_decode_np=None,
        num_draft_tokens_per_req=None
        if num_draft_tokens_per_req is None
        else np.array(num_draft_tokens_per_req, dtype=np.int32),
        idx_mapping=torch.arange(num_reqs),
        query_start_loc=torch.from_numpy(query_start_loc),
        seq_lens=seq_lens,
        dcp_local_seq_lens=None,
        positions=torch.zeros(num_tokens, dtype=torch.int64),
        prompt_lens=None,
    )


def _prepare_attn_metadata(
    state: MambaHybridModelState,
    monkeypatch,
    input_batch: SimpleNamespace | None = None,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
) -> SimpleNamespace:
    captured = {}

    def fake_build_attn_metadata(**kwargs):
        captured.update(kwargs)
        return Mock()

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_states.mamba_hybrid.build_attn_metadata",
        fake_build_attn_metadata,
    )

    if input_batch is None:
        input_batch = _input_batch([1, 1], None, [False, False])
    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=cudagraph_mode,
        block_tables=(),
        slot_mappings={},
        attn_groups=[],
        kv_cache_config=Mock(),
    )
    return captured["model_specific_attn_metadata"]


def test_prepare_attn_keeps_the_accepted_offset_when_the_batch_drafted_nothing(
    monkeypatch,
):
    """`num_accepted_tokens` must reach the attention builders whenever the
    speculative config is on, even when *this* batch scheduled no draft
    tokens at all. Before the fix, `use_spec_decode` was gated on this step's
    scheduled drafts (`len(scheduled_spec_decode_tokens) > 0` in the V1
    runner) rather than on the speculative config, so a request that accepted
    drafts on a previous step would silently lose that offset the moment
    ngram found no match for anyone in the batch.
    """
    state = _mamba_hybrid_state(num_speculative_tokens=3)

    metadata = _prepare_attn_metadata(state, monkeypatch)

    assert metadata.num_accepted_tokens is not None
    assert metadata.num_accepted_tokens.tolist() == [4, 1]
    assert metadata.num_decode_draft_tokens_cpu is not None


def test_prepare_attn_omits_the_offset_without_a_speculative_config(monkeypatch):
    """Mirrors the check above: with speculative decoding off entirely, there
    is no accepted-token offset to carry, and the metadata must say so
    explicitly (`None`) rather than a stale or default value."""
    state = _mamba_hybrid_state(num_speculative_tokens=0)

    metadata = _prepare_attn_metadata(state, monkeypatch)

    assert metadata.num_accepted_tokens is None
    assert metadata.num_decode_draft_tokens_cpu is None


def test_zero_draft_decode_row_is_marked_as_a_speculative_row(monkeypatch):
    """A decode row that got no drafts still has to reach the speculative path.

    The recurrent backends only apply a row's accepted-token offset on that
    path, and a row that accepted drafts on the previous step keeps that offset
    even when the current step proposes nothing (the scheduler drops drafts
    whenever the token budget truncates the step). Marking such a row with the
    -1 sentinel sends it to the plain decode path, which reads the state at
    column 0 and ignores the offset.
    """
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([3, 1, 1], [2, 0, 0], [False, False, True])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [
        2,  # decode row with drafts
        0,  # decode row with no drafts: speculative path, offset applied
        -1,  # one-token prefill tail: no offset to apply, stays a plain row
    ]


def test_batch_without_any_drafts_still_marks_its_decode_rows(monkeypatch):
    """`num_draft_tokens_per_req` is None whenever no row in the batch drafted
    anything. Those rows are ordinary decode rows carrying an offset from the
    previous step, so the absence of drafts in this step must not be read as
    speculative decoding being off."""
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([1, 1], None, [False, True])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [0, -1]


def test_padded_rows_keep_the_sentinel(monkeypatch):
    """Full CUDA-graph capture pads the batch; padded rows describe no request
    and must not be picked up as speculative rows."""
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([1, 1], [0, 0], [False, False], num_reqs_after_padding=4)

    metadata = _prepare_attn_metadata(
        state, monkeypatch, input_batch, cudagraph_mode=CUDAGraphMode.FULL
    )

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [0, 0, -1, -1]


def test_rows_wider_than_the_speculative_state_window_keep_the_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _mamba_hybrid_state(num_speculative_tokens=7)
    input_batch = _input_batch([12, 8, 1], None, [False, False, False])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [-1, 0, 0]


def test_chunked_prefill_tail_of_two_or_three_tokens_keeps_the_sentinel(
    monkeypatch,
):
    """A prefill continuation is excluded by `is_prefilling`, not by its token
    count, so tails wider than one token are covered by the same rule."""
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([2, 3], [0, 0], [True, True])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [-1, -1]


@pytest.mark.skipif(not HAS_GPU, reason="Requires CUDA or XPU")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device=GPU_DEVICE
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device=GPU_DEVICE)

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device=GPU_DEVICE
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    attn_groups = _recoverssm_attn_groups("layer")

    state.record_step({"layer": metadata}, attn_groups, for_capture=False)
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


@pytest.mark.skipif(not HAS_GPU, reason="Requires CUDA or XPU")
@pytest.mark.parametrize(
    ("computed", "sampled", "expected_column"),
    [(4, 3, 0), (4, 4, 0), (4, 5, 1), (8, 8, 1), (8, 9, 2)],
)
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias(
    computed: int, sampled: int, expected_column: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full(
        (5,), -1, dtype=torch.int32, device=GPU_DEVICE
    )
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device=GPU_DEVICE
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device=GPU_DEVICE),
        num_computed_tokens=torch.tensor(
            [6, computed], dtype=torch.int32, device=GPU_DEVICE
        ),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device=GPU_DEVICE),
    )
    num_sampled = torch.tensor([2, sampled], dtype=torch.int32, device=GPU_DEVICE)
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device=GPU_DEVICE)
    attn_groups = _recoverssm_attn_groups("layer")

    state.recoverssm.record_step({"layer": metadata}, attn_groups, for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, -1, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    # This fixture intentionally omits the generic align postprocess that
    # consumes RecoverSSM's private -1 sentinel and restores the public value
    # to one.
    expected_accepted = [9, -1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted
