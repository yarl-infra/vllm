# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "HAS_RESET_MASK": lambda args: args["reset_mask_ptr"] is not None,
    }
)
@triton.jit
def postprocess_recoverssm_align_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    request_indices_ptr,
    reset_mask_ptr,
    num_computed_ptr,
    state_idx_ptr,
    num_accepted_ptr,
    HAS_REQUEST_INDICES: tl.constexpr,
    HAS_RESET_MASK: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WIDTH: tl.constexpr,
):
    """Restore align bookkeeping after accepted-state materialization."""
    spec_idx = tl.program_id(0)
    if HAS_RESET_MASK and not tl.load(reset_mask_ptr + spec_idx):
        return
    batch_idx = spec_idx
    if HAS_REQUEST_INDICES:
        batch_idx = tl.load(request_indices_ptr + spec_idx)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_state_idx < 0:
        return
    # Keep the optimistic running column selected by align preprocessing.  The
    # ReplaySSM materializer writes the accepted state both to its logical
    # prefix page and to that running page.  Replacing this column with the
    # acceptance-derived page would disagree with V1's CPU-side page tracker
    # and suppress the next required pre-copy at an exact boundary.
    # RecoverSSM has already materialized both the accepted running state and,
    # when needed, the exact prefix-boundary checkpoint.  Mark this request so
    # the generic Mamba align postprocess that runs immediately afterwards does
    # not copy the newly materialized running page back over the boundary page.
    # The generic postprocess consumes this private sentinel and restores the
    # public neutral accepted-token value (1) before returning to the runner.
    tl.store(num_accepted_ptr + req_state_idx, -1)


@dataclass(frozen=True)
class RecoverSSMPostprocessMetadata:
    """Metadata used during postprocessing for align-mode prefix caching."""

    num_spec_decodes: int
    request_indices: torch.Tensor | None
    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    block_size: int
    reset_mask: torch.Tensor | None = None


class RecoverSSMMetadata(abc.ABC):
    @abc.abstractmethod
    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        raise NotImplementedError
