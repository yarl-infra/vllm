# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointBuilder,
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import MambaSpec

# FlashInfer's GDN prefill emits states every N tokens, N a multiple of its
# 64-token chunk, so checkpoint offsets must be multiples of 64.
FLASHINFER_GDN_CHECKPOINT_ALIGNMENT = 64


def plan_flashinfer_checkpoints(
    offsets: list[int], query_lens: list[int]
) -> tuple[int, list[int], list[int]]:
    """Lay out internal checkpoints as FlashInfer GDN prefill emits them.

    Args:
        offsets: Checkpoint offset into each checkpoint row's query, 0 for
            none. The last ``len(query_lens)`` rows are the prefill kernel's.
        query_lens: Query length of each sequence the prefill kernel runs.

    Returns:
        ``(every_n_tokens, cu_starts, rows)``: the stride shared by every
        checkpoint in the batch, cumulative per-sequence checkpoint counts,
        and the index of each checkpoint row's state among the emitted ones.

    """
    every_n_tokens = math.gcd(*offsets)
    cu_starts = [0]
    for query_len in query_lens:
        cu_starts.append(cu_starts[-1] + query_len // every_n_tokens)
    first = len(offsets) - len(query_lens)
    rows = [0] * len(offsets)
    for i, offset in enumerate(offsets[first:]):
        if offset:
            rows[first + i] = cu_starts[i] + offset // every_n_tokens - 1
    return every_n_tokens, cu_starts, rows


@dataclass
class GDNFlashInferCheckpointPlan:
    every_n_tokens: int
    num_checkpoints: int
    cu_starts: torch.Tensor
    # Per checkpoint row, its state among the ``num_checkpoints`` emitted.
    rows: torch.Tensor


if TYPE_CHECKING:
    from vllm.third_party.flash_linear_attention.ops.gdn_replayssm_prefix import (
        GDNReplaySSMPrefixCommitContext,
    )


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type[GDNAttentionMetadataBuilder]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNReplaySSMAlignMetadata:
    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    block_size: int


@dataclass
class GDNReplaySSMCommitMetadata:
    state_indices: torch.Tensor
    query_start_loc: torch.Tensor
    request_indices: torch.Tensor | None
    align: GDNReplaySSMAlignMetadata


@dataclass
class GDNAttentionMetadata(RecoverSSMMetadata):
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    checkpoint: MambaPrefillCheckpointMetadata | None = None
    flashinfer_checkpoint_plan: GDNFlashInferCheckpointPlan | None = None
    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_sequence_masks_cpu: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
    uniform_spec_sequence_length: int | None = None  # None for ragged batches

    # ReplaySSM spec ring cursors: block-keyed (num_blocks,) buffers indexed by
    # spec_state_indices_tensor[:, 0], advanced on-device once per step by
    # commit_gdn_replayssm_spec. None unless use_replayssm_spec is enabled.
    spec_write_pos_d: torch.Tensor | None = None
    spec_cache_base_d: torch.Tensor | None = None
    spec_is_flush_d: torch.Tensor | None = None

    replayssm_commit: GDNReplaySSMCommitMetadata | None = None
    replayssm_context: GDNReplaySSMPrefixCommitContext | None = field(
        default=None, repr=False, compare=False
    )

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None
    aiter_prefill_metadata: object | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        commit = self.replayssm_commit
        if commit is None:
            return None
        context = self.replayssm_context
        assert context is not None
        align = commit.align
        reset_mask = context.commit(
            num_accepted_tokens,
            commit.state_indices[: self.num_spec_decodes, 0],
            commit.query_start_loc[: self.num_spec_decodes + 1],
            request_indices=commit.request_indices,
            block_table=align.block_table,
            num_computed_tokens=align.num_computed_tokens,
            mamba_block_size=align.block_size,
        )
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=self.num_spec_decodes,
            request_indices=commit.request_indices,
            block_table=align.block_table,
            num_computed_tokens=align.num_computed_tokens,
            block_size=align.block_size,
            reset_mask=reset_mask,
        )


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    kv_cache_spec: MambaSpec
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1
    mamba_aligned_state_indices: torch.Tensor | None = None

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.checkpoint_builder = MambaPrefillCheckpointBuilder(
            vllm_config, kv_cache_spec
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal[
            "triton", "flashinfer", "cutedsl", "aiter_flydsl", "sycl"
        ]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)
        self._check_chunk_metadata_override(type(self), self.gdn_prefill_backend)
        # Only the Qwen GDN layer opts in on FlashInfer; KDA layers sharing this
        # backend export checkpoints their own way.
        checkpoint_alignment = kv_cache_spec.prefill_checkpoint_alignment
        self.plan_flashinfer_checkpoints = (
            self.gdn_prefill_backend == "flashinfer"
            and checkpoint_alignment is not None
            and checkpoint_alignment % FLASHINFER_GDN_CHECKPOINT_ALIGNMENT == 0
        )

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_replayssm_spec: bool = vllm_config.cache_config.use_replayssm_spec
        self.use_replayssm_prefix = (
            self.use_replayssm_spec
            and vllm_config.cache_config.mamba_cache_mode == "align"
        )
        self.replayssm_buffer_len: int = vllm_config.cache_config.replayssm_buffer_len
        self.max_spec_len: int = 1 + self.num_spec
        # Flush once the next window would not fit: threshold L = B + max_spec_len,
        # physical ring next_pow2(L). Cursors are sized from num_gpu_blocks, which
        # is only known after profiling, so they are allocated on the first build.
        self.spec_flush_threshold = self.replayssm_buffer_len + self.max_spec_len
        self.spec_ring_len = MambaStateShapeCalculator.replayssm_spec_ring_len(
            self.replayssm_buffer_len, self.num_spec
        )
        self.spec_write_pos: torch.Tensor | None = None
        self.spec_cache_base: torch.Tensor | None = None
        self.spec_is_flush: torch.Tensor | None = None
        self.replayssm_context: GDNReplaySSMPrefixCommitContext | None = None

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        # update_block_table() keeps the source group's batch-level FULL graph
        # buffers, so only MRV2, which also reuses metadata at capture, may use
        # it.
        self.supports_update_block_table = (
            vllm_config.use_v2_model_runner
            and device.type == "cuda"
            # Not isinstance: KDA's RecoverSSM/checkpoint metadata is per group.
            and type(self) is GDNAttentionMetadataBuilder
            # ReplaySSM also carries group-local cursors and prefix commit
            # context. Regathering only state indices would retain another
            # group's ring and update the wrong checkpoint pages.
            and not self.use_replayssm_spec
        )
        if self.supports_update_block_table:
            # Opts into MRV2's CUDA-only aligned-index precompute.
            self.mamba_aligned_state_indices: torch.Tensor | None = None

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    @staticmethod
    def _check_chunk_metadata_override(
        builder_cls: type[GDNAttentionMetadataBuilder], backend: str
    ) -> None:
        """Reject a subclass whose chunk metadata the AITER path would skip.

        AITER brings its own varlen prefill metadata and never calls
        ``_build_chunk_metadata``, so a builder that overrides it would lose
        that override without a word. No in-tree subclass can get here --
        ``_resolve_gdn_prefill_backend`` only selects this backend for models
        with 128-dim GDN key and value heads, which the KDA builders are not --
        but a future one should be told rather than quietly ignored.
        """
        if backend != "aiter_flydsl":
            return
        if (
            builder_cls._build_chunk_metadata
            is GDNAttentionMetadataBuilder._build_chunk_metadata
        ):
            return
        raise RuntimeError(
            f"{builder_cls.__name__} builds its own FLA chunk metadata, which "
            "the 'aiter_flydsl' GDN prefill backend does not use. Select a "
            "different gdn_prefill_backend for this model."
        )

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            assert prefill_query_start_loc is not None
            assert prefill_query_start_loc_cpu is not None
            total_tokens = int(prefill_query_start_loc_cpu[-1].item())
            return prepare_metadata_cutedsl(
                prefill_query_start_loc,
                total_tokens,
                FLA_CHUNK_SIZE,
            )

        # Only prefill batches use FLA chunk ops.
        # Pre-compute on CPU and async-copy to GPU to avoid
        # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        assert prefill_query_start_loc_cpu is not None
        return (
            async_tensor_h2d(
                prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
            async_tensor_h2d(
                prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        replayssm_reset_ring_cpu: torch.Tensor | None = None,
        replayssm_boundary_reset: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        aligned_state_indices = getattr(self, "mamba_aligned_state_indices", None)
        if aligned_state_indices is not None:
            block_table_tensor = aligned_state_indices[: m.num_reqs]
        else:
            block_table_tensor = mamba_get_block_table_tensor(
                m.block_table_tensor,
                m.seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )

        uniform_spec_sequence_length = None
        spec_sequence_masks_cpu: torch.Tensor | None = None
        if self.use_replayssm_spec and num_accepted_tokens is not None:
            # Every post-prefill row runs the spec kernel, a draft-less row as a
            # T=1 window. The baseline decode path reads the checkpoint page,
            # which lags the committed ring, so routing a decode row there
            # corrupts the state. num_decode_draft_tokens_cpu cannot drive this
            # mask: it is stale on draft-less steps and -1 for decode rows whose
            # drafts were dropped.
            is_prefilling_cpu = m.is_prefilling
            assert is_prefilling_cpu is not None
            query_lens_cpu_all = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            spec_sequence_masks_cpu = (
                (~is_prefilling_cpu[: query_lens_cpu_all.shape[0]])
                & (query_lens_cpu_all > 0)
                & (query_lens_cpu_all <= self.max_spec_len)
            )
            num_spec_decodes = int(spec_sequence_masks_cpu.sum().item())
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
                )
        elif not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            # A batch whose rows all drafted nothing still has to run the
            # speculative path: that is the only path that applies each row's
            # accepted-token offset to the recurrent state.
            if num_spec_decodes == 0:
                num_spec_decodes = 0
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
                )

        if spec_sequence_masks is None:
            # V2 already excludes prefills from full decode graphs via
            # has_prefill. Classify first chunks as prefills to mask recycled
            # state; resumed one-token chunks can still use the decode kernels.
            assert m.seq_lens_cpu_upper_bound is not None
            query_lens_cpu = query_start_loc_cpu.diff()
            no_prior_state = (query_lens_cpu > 0) & (
                m.seq_lens_cpu_upper_bound <= query_lens_cpu
            )
            # Capture batches also have seq_len == query_len, but are not
            # prefills.
            if m.is_prefilling is not None:
                no_prior_state &= m.is_prefilling
            else:
                no_prior_state = torch.zeros_like(no_prior_state)
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    m.replace(is_prefilling=no_prior_state),
                    decode_threshold=1,
                    treat_short_extends_as_decodes=False,
                )
            )
            # Exclude trailing padding from both prefill counts.
            if num_prefills:
                num_prefills -= int((query_lens_cpu[num_decodes:] == 0).sum())
                num_prefill_tokens = (
                    int(query_start_loc_cpu[num_decodes + num_prefills])
                    - num_decode_tokens
                )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            spec_query_lens_cpu = query_lens_cpu[spec_sequence_masks_cpu]
            if spec_query_lens_cpu.numel() > 0:
                first_spec_sequence_length = int(spec_query_lens_cpu[0])
                if first_spec_sequence_length > 0 and bool(
                    torch.all(spec_query_lens_cpu == first_spec_sequence_length)
                ):
                    uniform_spec_sequence_length = first_spec_sequence_length

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[non_spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Padded sequences trail the spec decodes, so slice them off
                # rather than gather with the host mask (an H2D copy + a kernel).
                spec_state_indices_tensor = block_table_tensor[
                    :num_spec_decodes, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    non_spec_sequence_masks_cpu, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            if num_prefills == 0 and num_decodes == 0:
                num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]
            else:
                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        aiter_prefill_metadata: object | None = None
        if num_prefills > 0:
            # In a mixed non-spec batch, decodes are peeled off to the recurrent
            # kernel (decode-first front slice), so build chunk metadata from the
            # rebased prefill-only cu_seqlens; otherwise use the full non-spec one.
            # _forward_core keys off the same condition, so they agree.
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                prefill_query_start_loc_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            if self.gdn_prefill_backend == "aiter_flydsl":
                # AITER carries its own reusable varlen metadata and has no use
                # for FLA's chunk indices, so it replaces them rather than
                # extending what _build_chunk_metadata returns.
                assert prefill_query_start_loc_cpu is not None
                aiter_prefill_metadata = (
                    rocm_aiter_ops.build_gdn_flydsl_prefill_metadata(
                        torch.diff(prefill_query_start_loc_cpu).tolist(),
                        cu_seqlens=prefill_query_start_loc,
                    )
                )
            elif self.gdn_prefill_backend != "flashinfer":
                # FlashInfer's chunk kernel consumes cu_seqlens directly and
                # never reads chunk_indices/chunk_offsets, so skip the FLA
                # metadata (CPU compute + H2D copies) entirely.
                chunk_indices, chunk_offsets = self._build_chunk_metadata(
                    prefill_query_start_loc,
                    prefill_query_start_loc_cpu,
                    query_start_loc.device,
                )

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        checkpoint: MambaPrefillCheckpointMetadata | None = None
        if num_prefills > 0:
            request_rows = list(range(m.num_reqs))
            if spec_sequence_masks_cpu is not None:
                request_rows = (~spec_sequence_masks_cpu).nonzero().flatten().tolist()
            checkpoint = self.checkpoint_builder.build(m, request_rows)

        flashinfer_checkpoint_plan: GDNFlashInferCheckpointPlan | None = None
        if checkpoint is not None and self.plan_flashinfer_checkpoints:
            assert checkpoint.offsets is not None
            assert checkpoint.query_lens is not None
            # The prefill kernel runs the trailing checkpoint rows.
            num_prefill_rows = prefill_query_start_loc_cpu.shape[0] - 1
            every_n_tokens, cu_starts, rows = plan_flashinfer_checkpoints(
                checkpoint.offsets, checkpoint.query_lens[-num_prefill_rows:]
            )
            plan = async_tensor_h2d(
                cu_starts + rows, query_start_loc.device, torch.int64
            )
            flashinfer_checkpoint_plan = GDNFlashInferCheckpointPlan(
                every_n_tokens=every_n_tokens,
                num_checkpoints=cu_starts[-1],
                cu_starts=plan[: len(cu_starts)],
                rows=plan[len(cu_starts) :],
            )

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        spec_write_pos_d = None
        spec_cache_base_d = None
        spec_is_flush_d = None
        if self.use_replayssm_spec and num_spec_decodes > 0:
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            assert spec_sequence_masks_cpu is not None
            spec_write_pos_d, spec_cache_base_d, spec_is_flush_d = (
                self._commit_replayssm_spec_cursors(
                    m,
                    spec_state_indices_tensor,
                    num_accepted_tokens,
                    spec_sequence_masks_cpu,
                    context_lens_tensor,
                    replayssm_reset_ring_cpu,
                    replayssm_boundary_reset,
                )
            )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if self._stage_spec_decode(
            num_prefills, num_decodes, num_spec_decodes, num_spec_decode_tokens
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if self._stage_decode(num_prefills, num_decodes, num_spec_decodes):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        replayssm_commit = None
        if self.use_replayssm_prefix and num_spec_decodes > 0:
            assert spec_state_indices_tensor is not None
            assert spec_query_start_loc is not None
            spec_request_indices = None
            if num_prefills > 0 or num_decodes > 0:
                assert spec_sequence_masks_cpu is not None
                spec_request_indices = async_tensor_h2d(
                    spec_sequence_masks_cpu.nonzero(as_tuple=True)[0],
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
            replayssm_commit = GDNReplaySSMCommitMetadata(
                state_indices=spec_state_indices_tensor,
                query_start_loc=spec_query_start_loc,
                request_indices=spec_request_indices,
                align=GDNReplaySSMAlignMetadata(
                    block_table=m.block_table_tensor,
                    num_computed_tokens=context_lens_tensor,
                    block_size=self.kv_cache_spec.block_size,
                ),
            )

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            checkpoint=checkpoint,
            flashinfer_checkpoint_plan=flashinfer_checkpoint_plan,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            aiter_prefill_metadata=aiter_prefill_metadata,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_sequence_masks_cpu=spec_sequence_masks_cpu,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            uniform_spec_sequence_length=uniform_spec_sequence_length,
            spec_write_pos_d=spec_write_pos_d,
            spec_cache_base_d=spec_cache_base_d,
            spec_is_flush_d=spec_is_flush_d,
            replayssm_commit=replayssm_commit,
            replayssm_context=(
                self._get_replayssm_prefix_context()
                if replayssm_commit is not None
                else None
            ),
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def _stage_spec_decode(
        self,
        num_prefills: int,
        num_decodes: int,
        num_spec_decodes: int,
        num_spec_decode_tokens: int,
    ) -> bool:
        """Whether spec-decode metadata goes into the FULL cudagraph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _stage_decode(
        self, num_prefills: int, num_decodes: int, num_spec_decodes: int
    ) -> bool:
        """Whether decode metadata goes into the FULL cudagraph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        )

    def update_block_table(
        self,
        metadata: GDNAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> GDNAttentionMetadata:
        """Re-gather this group's state indices. The other fields are
        batch-level and stay shared with ``metadata``."""
        m = metadata
        checkpoint = (
            m.checkpoint.regather_state_indices(blk_table)
            if m.checkpoint is not None
            else None
        )
        if self.vllm_config.cache_config.mamba_cache_mode == "align":
            assert self.mamba_aligned_state_indices is not None
            blk_table = self.mamba_aligned_state_indices
        masks = m.spec_sequence_masks_cpu
        spec_indices = non_spec_indices = prefill_indices = None
        if masks is None:
            non_spec_indices = blk_table[:, 0]
            if m.num_prefills > 0:
                prefill_indices = non_spec_indices[m.num_decodes :]
        elif m.num_prefills == 0:
            # Same as build(): padded sequences trail the spec decodes.
            spec_indices = blk_table[: m.num_spec_decodes, : self.num_spec + 1]
        else:
            spec_indices = blk_table[masks, : self.num_spec + 1]
            non_spec_indices = prefill_indices = blk_table[~masks, 0]

        if self._stage_spec_decode(
            m.num_prefills, m.num_decodes, m.num_spec_decodes, m.num_spec_decode_tokens
        ):
            assert m.spec_state_indices_tensor is not None
            batch_size = m.spec_state_indices_tensor.shape[0]
            self.spec_state_indices_tensor[: m.num_spec_decodes].copy_(
                spec_indices, non_blocking=True
            )
            spec_indices = self.spec_state_indices_tensor[:batch_size]
            spec_indices[m.num_spec_decodes :].fill_(NULL_BLOCK_ID)

        if self._stage_decode(m.num_prefills, m.num_decodes, m.num_spec_decodes):
            assert m.non_spec_state_indices_tensor is not None
            batch_size = m.non_spec_state_indices_tensor.shape[0]
            self.non_spec_state_indices_tensor[: m.num_decodes].copy_(
                non_spec_indices, non_blocking=True
            )
            non_spec_indices = self.non_spec_state_indices_tensor[:batch_size]
            non_spec_indices[m.num_decodes :].fill_(NULL_BLOCK_ID)

        return replace(
            m,
            spec_state_indices_tensor=spec_indices,
            non_spec_state_indices_tensor=non_spec_indices,
            prefill_state_indices=prefill_indices,
            checkpoint=checkpoint,
        )

    def _get_replayssm_prefix_context(self) -> GDNReplaySSMPrefixCommitContext:
        context = self.replayssm_context
        if context is not None:
            return context
        assert self.spec_write_pos is not None
        assert self.spec_cache_base is not None
        assert self.spec_is_flush is not None
        from vllm.third_party.flash_linear_attention.ops.gdn_replayssm_prefix import (
            GDNReplaySSMPrefixCommitContext,
        )

        forward_context = self.vllm_config.compilation_config.static_forward_context
        layers = [forward_context[layer_name] for layer_name in self.layer_names]
        context = GDNReplaySSMPrefixCommitContext.create(
            layers,
            write_pos=self.spec_write_pos,
            cache_base=self.spec_cache_base,
            is_flush=self.spec_is_flush,
            spec_query_len=self.max_spec_len,
            max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            max_cache_len=self.spec_flush_threshold,
        )
        self.replayssm_context = context
        return context

    def _commit_replayssm_spec_cursors(
        self,
        m: CommonAttentionMetadata,
        spec_state_indices_tensor: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spec_sequence_masks_cpu: torch.Tensor,
        context_lens_tensor: torch.Tensor,
        reset_ring_cpu: torch.Tensor | None,
        boundary_reset: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the block-keyed spec ring cursors for this step.

        Commits the previous step's accepted tokens and decides this step's
        is_flush on-device, then resets rows entering decode. Both run as fixed
        launches so the captured graph is identical every step.
        """
        from vllm.third_party.flash_linear_attention.ops.gdn_replayssm_spec_decode import (  # noqa: E501
            commit_gdn_replayssm_spec,
            reset_gdn_replayssm_spec_cursors,
        )

        device = spec_state_indices_tensor.device
        if self.spec_write_pos is None:
            num_gpu_blocks = self.vllm_config.cache_config.num_gpu_blocks
            if not num_gpu_blocks:
                raise ValueError(
                    "--use-replayssm-spec needs num_gpu_blocks at build time to "
                    "size the block-keyed cursor buffers"
                )
            self.spec_write_pos = torch.zeros(
                num_gpu_blocks, dtype=torch.int32, device=device
            )
            self.spec_cache_base = torch.zeros(
                num_gpu_blocks, dtype=torch.int32, device=device
            )
            self.spec_is_flush = torch.zeros(
                num_gpu_blocks, dtype=torch.int8, device=device
            )
        assert self.spec_cache_base is not None and self.spec_is_flush is not None

        block_ids = spec_state_indices_tensor[:, 0]
        # Commit first: acceptance is only known after the previous step's
        # sampling, so folding it into this build keeps every cursor update
        # on-device and avoids a device-to-host sync on num_accepted_tokens.
        if not self.use_replayssm_prefix:
            commit_gdn_replayssm_spec(
                self.spec_write_pos,
                self.spec_cache_base,
                self.spec_is_flush,
                num_accepted_tokens.to(torch.int32),
                block_ids,
                max_cache_len=self.spec_flush_threshold,
                max_spec_len=self.max_spec_len,
                cache_buf_len=self.spec_ring_len,
            )

        if reset_ring_cpu is not None:
            # V2 passes an explicit one-shot admission bit.  Its async context
            # accounting can expose both prefill_len - 1 and prefill_len on the
            # first two decode builds, so length comparisons are ambiguous.
            reset_ring_full = reset_ring_cpu.to(device, non_blocking=True)
        else:
            # V1 keeps the admission origin in GPUInputBatch and has exact
            # context accounting, so retain its established equality fallback.
            decode_base_cpu = m.replayssm_decode_base_cpu
            if decode_base_cpu is None:
                raise ValueError(
                    "--use-replayssm-spec requires an admission reset signal"
                )
            decode_base_d = decode_base_cpu.to(device, non_blocking=True)
            reset_ring_full = (
                context_lens_tensor[: decode_base_d.shape[0]] == decode_base_d
            ).to(torch.int8)
        if boundary_reset is not None:
            # Generic align pre-copy migrates checkpoint/conv state but not the
            # private ReplaySSM ring.  Reset the destination on the exact step
            # where preprocess_state changed the running page.  A page-number
            # lookahead remains true for several adjacent speculative windows
            # and would repeatedly discard accepted tokens from the new ring.
            reset_ring_full = torch.maximum(
                reset_ring_full[: boundary_reset.shape[0]], boundary_reset
            )
        spec_row_idx = spec_sequence_masks_cpu.nonzero(as_tuple=True)[0].to(
            device, non_blocking=True
        )
        reset_gdn_replayssm_spec_cursors(
            self.spec_write_pos,
            self.spec_cache_base,
            self.spec_is_flush,
            reset_ring_full.index_select(0, spec_row_idx),
            block_ids,
            max_cache_len=self.spec_flush_threshold,
            max_spec_len=self.max_spec_len,
        )
        return self.spec_write_pos, self.spec_cache_base, self.spec_is_flush

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = torch.diff(m.query_start_loc_cpu).sub_(1)
        assert num_decode_draft_tokens_cpu.shape == num_accepted_tokens.shape
        replayssm_reset_ring_cpu = (
            torch.zeros(m.num_reqs, dtype=torch.int8)
            if self.use_replayssm_spec
            else None
        )

        return self.build(
            0,
            m,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            replayssm_reset_ring_cpu,
            None,
        )
