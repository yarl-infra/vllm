# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 ReplaySSM decode write-position derivation in
BaseMambaAttentionMetadataBuilder: write_pos and is_flush computed from the
per-request ring origin (replayssm_decode_base) and num_computed.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    MockMambaBuilder,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.mamba import MambaBackendEnum
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec


def test_gdn_replay_admission_reset_includes_decode_promoted_prompt_tail():
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    from vllm.v1.worker.utils import AttentionGroup

    config = create_vllm_config(max_num_seqs=4, max_num_batched_tokens=32)
    config.cache_config.use_replayssm_spec = True
    config.cache_config.mamba_cache_mode = "none"
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )
    device = torch.device("cuda")
    # Only request/attention metadata is exercised; no model forward is needed.
    state = MambaHybridModelState(config, torch.nn.Identity(), None, device)
    spec = MambaSpec(
        block_size=16,
        shapes=((7, 8), (1, 8, 8)),
        dtypes=(torch.float32, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    group = AttentionGroup(
        MambaAttentionBackendEnum.GDN_ATTN.get_class(), ["gdn"], spec, 0
    )
    group.create_metadata_builders(config, device)
    kv = KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["gdn"], spec)],
    )
    batch = InputBatch.make_dummy(1, 4, InputBuffers(4, 32, device), is_padding=False)
    batch = replace(
        batch,
        seq_lens=torch.tensor([20], dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor([20], dtype=torch.int32),
        is_prefilling_np=np.array([True]),
        has_prefill=True,
        prefill_runs_as_decode_np=np.array([True]),
        num_draft_tokens=3,
        num_draft_tokens_per_req=np.array([3], dtype=np.int32),
    )
    state.add_request(
        0,
        NewRequestData(
            req_id=batch.req_ids[0],
            prompt_token_ids=[1] * 17,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([1, 2],),
            num_computed_tokens=16,
            lora_request=None,
        ),
    )
    block_table = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=device)

    def build(current):
        return state.prepare_attn(
            current,
            CUDAGraphMode.NONE,
            (block_table,),
            torch.zeros((1, 4), dtype=torch.int64, device=device),
            [[group]],
            kv,
        )["gdn"]

    tail = build(batch)
    assert tail.num_spec_decodes == 1 and tail.num_prefills == 0
    block_id = tail.spec_state_indices_tensor[0, 0].item()
    assert tail.spec_write_pos_d[block_id].item() == 0
    # The admission reset must be consumed by the promoted prompt tail, not
    # delayed until the next step, which would discard accepted ring entries.
    decode = replace(
        batch,
        is_prefilling_np=np.array([False]),
        has_prefill=False,
        prefill_runs_as_decode_np=None,
    )
    assert build(decode).spec_write_pos_d[block_id].item() == 1
    assert build(decode).spec_write_pos_d[block_id].item() == 2


def test_gdn_replay_metadata_keeps_group_local_ring_cursors():
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata
    from vllm.v1.worker.utils import AttentionGroup

    config = create_vllm_config(max_num_seqs=4, max_num_batched_tokens=32)
    config.cache_config.use_replayssm_spec = True
    config.cache_config.mamba_cache_mode = "none"
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )
    device = torch.device("cuda")
    spec = MambaSpec(
        block_size=16,
        shapes=((7, 8), (1, 8, 8)),
        dtypes=(torch.float32, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    groups = [
        AttentionGroup(
            MambaAttentionBackendEnum.GDN_ATTN.get_class(), [f"gdn{i}"], spec, i
        )
        for i in range(2)
    ]
    for group in groups:
        group.create_metadata_builders(config, device)
    kv = KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(g.layer_names, spec) for g in groups],
    )
    metadata = build_attn_metadata(
        [[g] for g in groups],
        num_reqs=1,
        num_tokens=4,
        query_start_loc_gpu=torch.tensor([0, 4], device=device, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        max_query_len=4,
        seq_lens=torch.tensor([20], device=device, dtype=torch.int32),
        max_seq_len=20,
        block_tables=[
            torch.tensor([[b]], device=device, dtype=torch.int32) for b in (2, 8)
        ],
        slot_mappings=torch.zeros((2, 4), device=device, dtype=torch.int64),
        kv_cache_config=kv,
        seq_lens_cpu_upper_bound=torch.tensor([20], dtype=torch.int32),
        model_specific_attn_metadata=MambaHybridAttnMetadata(
            is_prefilling=torch.tensor([False]),
            num_accepted_tokens=torch.tensor([1], device=device, dtype=torch.int32),
            num_decode_draft_tokens_cpu=torch.tensor([3], dtype=torch.int32),
            replayssm_reset_ring_cpu=torch.tensor([0], dtype=torch.int8),
        ),
    )
    for i in range(2):
        meta = metadata[f"gdn{i}"]
        block_id = meta.spec_state_indices_tensor[0, 0].item()
        assert meta.spec_write_pos_d[block_id].item() == 1
    assert (
        metadata["gdn0"].spec_write_pos_d.data_ptr()
        != metadata["gdn1"].spec_write_pos_d.data_ptr()
    )


BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


def _make_spec_replay_config(num_spec: int, mode: str):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            use_replayssm=mode == "decode",
            use_replayssm_spec=mode == "spec",
            replayssm_buffer_len=16,
            mamba_cache_mode="align",
        ),
        mamba_config=SimpleNamespace(backend=MambaBackendEnum.TRITON),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            max_cudagraph_capture_size=0,
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=num_spec, parallel_drafting=False
        ),
        num_speculative_tokens=num_spec,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )


@pytest.mark.parametrize("num_spec", [2, 3])
@pytest.mark.parametrize("mode", ["none", "decode", "spec"])
def test_ple_spec_metadata_with_global_replayssm(
    num_spec: int, mode: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    config = _make_spec_replay_config(num_spec, mode)
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((9 + num_spec, 8),),
        dtypes=(torch.float32,),
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
    )
    builder = PleShortConvAttentionMetadataBuilder(spec, ["ple"], config, DEVICE)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[32, 48], query_lens=[num_spec + 1] * 2),
        BLOCK_SIZE,
        DEVICE,
    )
    accepted = torch.tensor([1, num_spec], dtype=torch.int32)
    metadata = builder.build(
        0,
        common,
        num_accepted_tokens=accepted,
        num_decode_draft_tokens_cpu=torch.tensor([num_spec] * 2),
    )
    assert metadata.num_spec_decodes == 2
    assert metadata.spec_query_start_loc is not None
    assert metadata.spec_query_start_loc.tolist() == [
        0,
        num_spec + 1,
        2 * (num_spec + 1),
    ]
    assert torch.equal(metadata.num_accepted_tokens, accepted)
    assert torch.equal(
        metadata.spec_state_indices_tensor, common.block_table_tensor[[0, 1], [1, 2]]
    )
    assert not builder.use_replayssm
    assert not builder.use_replayssm_spec
    assert config.cache_config.use_replayssm_spec == (mode == "spec")


@pytest.mark.parametrize("valid_page", [False, True])
def test_mamba2_spec_replay_still_validates_page(valid_page: bool):
    config = _make_spec_replay_config(2, "spec")
    shapes = ((4, 2), (1, 1, 1), (32, 2), (1, 32))
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=shapes if valid_page else shapes[:2],
        dtypes=(torch.float32,) * (4 if valid_page else 2),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
    )
    if not valid_page:
        with pytest.raises(ValueError, match="4-tensor Mamba2 page"):
            MockMambaBuilder(spec, ["mamba"], config, DEVICE)
    else:
        builder = MockMambaBuilder(spec, ["mamba"], config, DEVICE)
        assert builder.use_replayssm_spec
        assert builder.decode_spec_bc_pre is not None


def test_spec_replay_reset_uses_exact_device_context_not_cpu_upper_bound():
    config = create_vllm_config(model_name="Qwen/Qwen3.5-0.8B", max_num_seqs=4)
    config.cache_config.use_replayssm_spec = True
    config.cache_config.replayssm_buffer_len = 16
    config.cache_config.mamba_cache_mode = "none"
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=2
    )
    device = torch.device("cuda")
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((4, 3), (1, 1, 1), (32, 2), (1, 32)),
        dtypes=(torch.float32,) * 4,
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
    )
    builder = Mamba2AttentionMetadataBuilder(spec, ["layer0"], config, device)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[7, 9, 5], query_lens=[3, 3, 1]),
        BLOCK_SIZE,
        device,
        arange_block_indices=True,
    ).replace(
        seq_lens_cpu_upper_bound=torch.tensor([17, 19, 15], dtype=torch.int32),
        is_prefilling=torch.tensor([False, False, True]),
        replayssm_decode_base_cpu=torch.tensor([4, 5, 5], dtype=torch.int32),
    )
    metadata = builder.build(
        0,
        common,
        num_accepted_tokens=torch.tensor([2, 2, 1], device=device),
        num_decode_draft_tokens_cpu=torch.tensor([2, 2, 0]),
    )
    assert metadata.spec_write_pos_d is not None
    assert metadata.spec_is_flush_d is not None
    # First decode resets, continuing decode commits, and a prompt tail flushes.
    assert metadata.spec_write_pos_d[:3].tolist() == [0, 2, 0]
    assert metadata.spec_is_flush_d[:3].tolist() == [0, 0, 1]


@dataclass
class ReplaySSMBuildCase:
    """A decode batch and its expected per-row write_pos / is_flush.

    num_computed = seq_len - query_len; write_pos =
    (num_computed - decode_base) % buffer_len; is_flush = write_pos ==
    buffer_len - 1 (or a forced one-token flush when num_computed < decode_base).
    """

    seq_lens: list[int]
    query_lens: list[int]
    is_prefilling: list[bool]
    decode_base: list[int]
    buffer_len: int
    expected_write_pos: list[int]
    expected_is_flush: list[int]
    mamba_cache_mode: str = "none"


REPLAYSSM_BUILD_CASES = {
    # decode_base == num_prompt (fresh request).
    "fresh_decode": ReplaySSMBuildCase(
        seq_lens=[106],
        query_lens=[1],
        is_prefilling=[False],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[5],
        expected_is_flush=[0],
    ),
    # decode_base > num_prompt anchors write_pos at the resume point.
    "resumed_reanchors_to_zero": ReplaySSMBuildCase(
        seq_lens=[106],
        query_lens=[1],
        is_prefilling=[False],
        decode_base=[105],
        buffer_len=16,
        expected_write_pos=[0],
        expected_is_flush=[0],
    ),
    # write_pos == buffer_len - 1 flushes.
    "flush_boundary": ReplaySSMBuildCase(
        seq_lens=[116],
        query_lens=[1],
        is_prefilling=[False],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[15],
        expected_is_flush=[1],
    ),
    # Resumed request landing on a flush boundary.
    "resumed_flush_boundary": ReplaySSMBuildCase(
        seq_lens=[121],
        query_lens=[1],
        is_prefilling=[False],
        decode_base=[105],
        buffer_len=16,
        expected_write_pos=[15],
        expected_is_flush=[1],
    ),
    # Per-row write_pos / is_flush are independent.
    "mixed_rows": ReplaySSMBuildCase(
        seq_lens=[104, 106, 216],
        query_lens=[1, 1, 1],
        is_prefilling=[False, False, False],
        decode_base=[100, 105, 200],
        buffer_len=16,
        expected_write_pos=[3, 0, 15],
        expected_is_flush=[0, 0, 1],
    ),
    # write_pos wraps within the buffer (6 % 4 == 2).
    "small_buffer_wrap": ReplaySSMBuildCase(
        seq_lens=[112],
        query_lens=[1],
        is_prefilling=[False],
        decode_base=[105],
        buffer_len=4,
        expected_write_pos=[2],
        expected_is_flush=[0],
    ),
    # Single-token prefill-as-decode still in the prompt (num_computed <
    # decode_base): forced one-token flush.
    "leftover_prompt_one_token_flush": ReplaySSMBuildCase(
        seq_lens=[100],
        query_lens=[1],
        is_prefilling=[True],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[0],
        expected_is_flush=[1],
    ),
    # Align mode (block_size 16). Past the first boundary the ring re-anchors at
    # the block start: num_computed 117 -> block_start 112, write_pos 5 (vs 1 in
    # none mode).
    "align_reanchor_past_boundary": ReplaySSMBuildCase(
        seq_lens=[118],
        query_lens=[1],
        is_prefilling=[False],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[5],
        expected_is_flush=[0],
        mamba_cache_mode="align",
    ),
    # First-block boundary (num_computed+1 == 112) forces a flush even though
    # write_pos (11) != buffer_len - 1.
    "align_first_block_boundary_flush": ReplaySSMBuildCase(
        seq_lens=[112],
        query_lens=[1],
        is_prefilling=[False],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[11],
        expected_is_flush=[1],
        mamba_cache_mode="align",
    ),
    # First step of a new block re-anchors write_pos to 0.
    "align_new_block_start_zero": ReplaySSMBuildCase(
        seq_lens=[113],
        query_lens=[1],
        is_prefilling=[False],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[0],
        expected_is_flush=[0],
        mamba_cache_mode="align",
    ),
    # block_size % buffer_len == 0: a later boundary lands on write_pos ==
    # buffer_len - 1, so the boundary flush coincides with the natural flush.
    "align_boundary_coincides_natural_flush": ReplaySSMBuildCase(
        seq_lens=[128],
        query_lens=[1],
        is_prefilling=[False],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[15],
        expected_is_flush=[1],
        mamba_cache_mode="align",
    ),
    # block_size % buffer_len != 0 (buffer_len 6): the boundary step still flushes
    # although write_pos (3) != buffer_len - 1.
    "align_unaligned_buffer_forces_flush": ReplaySSMBuildCase(
        seq_lens=[128],
        query_lens=[1],
        is_prefilling=[False],
        decode_base=[100],
        buffer_len=6,
        expected_write_pos=[3],
        expected_is_flush=[1],
        mamba_cache_mode="align",
    ),
    # Per-row independence in align mode: partial-block / new-block / boundary.
    "align_mixed_rows": ReplaySSMBuildCase(
        seq_lens=[105, 113, 112],
        query_lens=[1, 1, 1],
        is_prefilling=[False, False, False],
        decode_base=[100, 100, 100],
        buffer_len=16,
        expected_write_pos=[4, 0, 11],
        expected_is_flush=[0, 0, 1],
        mamba_cache_mode="align",
    ),
}


def _make_mamba_spec(
    buffer_len: int,
    mamba_backend: MambaBackendEnum,
    num_speculative_tokens: int = 0,
) -> MambaSpec:
    ring_buffer_len = buffer_len + (
        1 + num_speculative_tokens
        if mamba_backend == MambaBackendEnum.FLASHINFER
        else 0
    )
    shapes = (
        (1, 1),
        (1, 1, 1),
        (1, ring_buffer_len, 1),
        (1, ring_buffer_len),
        (1, ring_buffer_len, 1),
    )
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=shapes,
        dtypes=(torch.float32,),
    )


def _create_replayssm_builder(
    buffer_len: int,
    mamba_cache_mode: str = "none",
    *,
    mamba_backend: MambaBackendEnum = MambaBackendEnum.TRITON,
    num_speculative_tokens: int = 0,
) -> MockMambaBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    # Set the flags after construction to skip validate_mamba_cached_kernel
    # (it requires a real SupportsReplaySSM model) on the mock model.
    vllm_config.cache_config.use_replayssm = True
    vllm_config.cache_config.replayssm_buffer_len = buffer_len
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    vllm_config.mamba_config.backend = mamba_backend
    if num_speculative_tokens > 0:
        # Triton ReplaySSM requires exact synchronous CPU metadata.
        vllm_config.scheduler_config.async_scheduling = False
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    return MockMambaBuilder(
        _make_mamba_spec(buffer_len, mamba_backend, num_speculative_tokens),
        ["layer0"],
        vllm_config,
        DEVICE,
    )


def _build(
    builder: MockMambaBuilder,
    case: ReplaySSMBuildCase,
    num_accepted_tokens: torch.Tensor | None = None,
):
    batch = BatchSpec(seq_lens=case.seq_lens, query_lens=case.query_lens)
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor(case.is_prefilling, dtype=torch.bool),
        replayssm_decode_base_cpu=torch.tensor(case.decode_base, dtype=torch.int32),
    )
    return builder.build(0, common, num_accepted_tokens=num_accepted_tokens)


@pytest.mark.parametrize(
    "case", REPLAYSSM_BUILD_CASES.values(), ids=REPLAYSSM_BUILD_CASES.keys()
)
def test_replayssm_write_pos(case: ReplaySSMBuildCase):
    builder = _create_replayssm_builder(case.buffer_len, case.mamba_cache_mode)
    meta = _build(builder, case)

    assert meta.write_pos_d is not None
    assert meta.is_flush_d is not None
    n = len(case.expected_write_pos)
    assert meta.write_pos_d[:n].tolist() == case.expected_write_pos
    assert meta.is_flush_d[:n].tolist() == case.expected_is_flush


def test_resumed_request_differs_from_fresh():
    """Same token count, different decode_base: fresh (base 100) -> write_pos 5,
    resumed (base 105) -> write_pos 0."""
    builder = _create_replayssm_builder(16)
    batch = BatchSpec(seq_lens=[106, 106], query_lens=[1, 1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor([False, False]),
        replayssm_decode_base_cpu=torch.tensor([100, 105], dtype=torch.int32),
    )
    meta = builder.build(0, common)

    assert meta.write_pos_d is not None
    assert meta.is_flush_d is not None
    assert meta.write_pos_d.tolist()[:2] == [5, 0]
    assert meta.is_flush_d.tolist()[:2] == [0, 0]


def test_spec_decode_single_token_chunk_synthesizes_acceptance_metadata():
    builder = _create_replayssm_builder(16, num_speculative_tokens=3)
    case = REPLAYSSM_BUILD_CASES["leftover_prompt_one_token_flush"]

    meta = _build(builder, case)

    assert meta.query_start_loc_d is not None
    assert meta.query_start_loc_d.tolist() == [0, 1]
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.tolist() == [1]


def test_flashinfer_replayssm_state_indices_are_stable_for_full_cudagraph():
    checkpointing_ssu = pytest.importorskip("flashinfer.mamba.checkpointing_ssu")
    if not hasattr(checkpointing_ssu, "allocate_checkpointing_ssu_scratch"):
        pytest.skip("requires FlashInfer ReplaySSM autotuning support")

    builder = _create_replayssm_builder(
        16,
        mamba_backend=MambaBackendEnum.FLASHINFER,
        num_speculative_tokens=3,
    )
    builder.compilation_config.cudagraph_mode = CUDAGraphMode.FULL

    first = _build(
        builder,
        ReplaySSMBuildCase(
            seq_lens=[106, 106],
            query_lens=[1, 1],
            is_prefilling=[False, False],
            decode_base=[100, 100],
            buffer_len=16,
            expected_write_pos=[],
            expected_is_flush=[],
        ),
    )
    first_indices = first.replayssm_state_indices_d
    assert first_indices is not None
    assert first_indices.is_contiguous()
    first_ptr = first_indices.data_ptr()

    second = _build(
        builder,
        ReplaySSMBuildCase(
            seq_lens=[122, 122],
            query_lens=[1, 1],
            is_prefilling=[False, False],
            decode_base=[116, 116],
            buffer_len=16,
            expected_write_pos=[],
            expected_is_flush=[],
        ),
    )
    second_indices = second.replayssm_state_indices_d
    assert second_indices is not None
    assert second_indices.data_ptr() == first_ptr
    assert torch.equal(second_indices, second.state_indices_tensor_d[:, 0])
