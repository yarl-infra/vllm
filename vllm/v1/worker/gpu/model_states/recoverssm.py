# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
    postprocess_recoverssm_align_kernel,
)
from vllm.v1.worker.utils import AttentionGroup


class RecoverSSMState:
    """Coordinates RecoverSSM metadata between attention and postprocessing."""

    def __init__(self) -> None:
        self._step: tuple[RecoverSSMMetadata, ...] | None = None
        self._num_sampled_snapshot: torch.Tensor | None = None

    def record_step(
        self,
        attn_metadata: dict[str, Any],
        attn_groups: list[list[AttentionGroup]],
        *,
        for_capture: bool,
    ) -> None:
        if for_capture:
            self._step = None
            return

        step: list[RecoverSSMMetadata] = []
        for group_list in attn_groups:
            for group in group_list:
                metadata = attn_metadata[group.layer_names[0]]
                if isinstance(metadata, RecoverSSMMetadata):
                    step.append(metadata)
        self._step = tuple(step)

    def commit_step(
        self,
        num_sampled: torch.Tensor | int,
        idx_mapping: torch.Tensor,
        *,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        step = self._step
        self._step = None
        if isinstance(num_sampled, int) or step is None:
            return

        # Every cache group must commit the same accepted-token snapshot.  The
        # align postprocess below neutralizes ``num_accepted_tokens`` to one for
        # the next pre-copy.  ``num_sampled`` can be a view of that exact buffer,
        # so keep a private snapshot for both phases: otherwise the first group
        # shortens later groups' ReplaySSM histories and their page-index update
        # whenever MTP accepts more than one token.
        snapshot_storage = self._num_sampled_snapshot
        if (
            snapshot_storage is None
            or snapshot_storage.device != num_accepted_tokens.device
            or snapshot_storage.shape != num_accepted_tokens.shape
        ):
            snapshot_storage = torch.empty_like(num_accepted_tokens)
            self._num_sampled_snapshot = snapshot_storage
        num_sampled_snapshot = snapshot_storage[: num_sampled.shape[0]]
        num_sampled_snapshot.copy_(num_sampled)
        postprocess_steps: list[RecoverSSMPostprocessMetadata] = []
        for metadata in step:
            postprocess_meta = metadata.commit_recoverssm_state(num_sampled_snapshot)
            if postprocess_meta is None:
                continue
            postprocess_steps.append(postprocess_meta)

        for postprocess_meta in postprocess_steps:
            assert state_indices is not None
            # RecoverSSM already restored the accepted state. Update its running
            # column and reset the next-step copy bias to the neutral value.
            postprocess_recoverssm_align_kernel[(postprocess_meta.num_spec_decodes,)](
                idx_mapping,
                num_sampled_snapshot,
                postprocess_meta.request_indices,
                postprocess_meta.reset_mask,
                postprocess_meta.num_computed_tokens,
                state_indices,
                num_accepted_tokens,
                MAMBA_BLOCK_SIZE=postprocess_meta.block_size,
                BLOCK_TABLE_WIDTH=postprocess_meta.block_table.shape[1],
            )
