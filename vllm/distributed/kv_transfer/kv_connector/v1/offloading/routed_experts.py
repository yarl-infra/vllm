# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side routing storage for the native CPU KV offload pool."""

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsManager,
    )

logger = init_logger(__name__)


class RoutedExpertsOffloadManager:
    """Mirror routing by physical CPU offload slot, using the KV transfer specs.

    The worker already returns routing to the scheduler, so no additional GPU
    buffer or transfer is needed. Only the full-attention group chosen by
    RoutedExpertsManager is copied, including for hybrid attention/SSM models.

    CPU IDs have the same lifetime as the native KV pool: eviction reuses IDs,
    and reset invalidates lookup metadata rather than clearing backing bytes.
    Thus storage is bounded and needs no separate hash cache/eviction policy.
    KV stores become available for lookup only after the corresponding output
    (and hence its routing snapshot) has been processed.
    """

    def __init__(
        self,
        routed_experts: "RoutedExpertsManager",
        num_cpu_blocks: int,
        blocks_per_chunk: int,
    ) -> None:
        self.routed_experts = routed_experts
        self.blocks_per_chunk = blocks_per_chunk
        slot_buffer = routed_experts.routed_experts_by_slot
        self.offloaded_experts = np.empty(
            (
                num_cpu_blocks * blocks_per_chunk * routed_experts.block_size,
                *slot_buffer.shape[1:],
            ),
            dtype=slot_buffer.dtype,
        )
        self._steps: deque[OffloadingConnectorMetadata] = deque()
        logger.info(
            "Routed experts CPU offload buffer: %.2f GB (dtype=%s). "
            "This is additional to the KV offloading memory budget.",
            self.offloaded_experts.nbytes / 1e9,
            self.offloaded_experts.dtype.name,
        )

    def on_schedule(self, metadata: KVConnectorMetadata) -> None:
        assert isinstance(metadata, OffloadingConnectorMetadata)
        # The worker pops early-flushed stores from metadata.store_jobs. Retain
        # a scheduler-owned snapshot, also when using the in-process executor.
        self._steps.append(
            OffloadingConnectorMetadata(
                load_jobs=dict(metadata.load_jobs),
                store_jobs=dict(metadata.store_jobs),
                jobs_to_flush=set(metadata.jobs_to_flush or ()),
            )
        )

    def before_step(self) -> None:
        metadata = self._steps[0]
        # Finished/preempted requests can have a last store whose source blocks
        # are reused in this very step. Preserve it before loads/forward writes.
        for job_id in metadata.jobs_to_flush or ():
            job = metadata.store_jobs.pop(job_id, None)
            if job is not None:
                self._copy(job, is_store=True)
        for job in metadata.load_jobs.values():
            self._copy(job, is_store=False)

    def after_step(self) -> None:
        # Ordinary stores include routing computed by this step's forward.
        metadata = self._steps.popleft()
        for job in metadata.store_jobs.values():
            self._copy(job, is_store=True)

    def _copy(self, job: TransferJob, *, is_store: bool) -> None:
        gpu = job.src_spec if is_store else job.dst_spec
        cpu = job.dst_spec if is_store else job.src_spec
        assert isinstance(gpu, GPULoadStoreSpec)
        assert isinstance(cpu, CPULoadStoreSpec)
        block_size = self.routed_experts.block_size
        blocks_per_chunk = self.blocks_per_chunk
        gpu_offset = cpu_offset = 0
        for gid, (count, block_index) in enumerate(
            zip(gpu.group_sizes, gpu.block_indices)
        ):
            if count == 0:
                continue
            skip = block_index % blocks_per_chunk
            cpu_count = (skip + count + blocks_per_chunk - 1) // blocks_per_chunk
            if gid == self.routed_experts.attn_gid:
                gpu_blocks = gpu.block_ids[gpu_offset : gpu_offset + count]
                cpu_chunks = cpu.block_ids[cpu_offset : cpu_offset + cpu_count]
                assert len(gpu_blocks) == count
                assert len(cpu_chunks) == cpu_count
                cpu_blocks = (
                    cpu_chunks[:, None] * blocks_per_chunk + np.arange(blocks_per_chunk)
                ).reshape(-1)[skip : skip + count]
                offsets = np.arange(block_size)
                gpu_slots = (gpu_blocks[:, None] * block_size + offsets).reshape(-1)
                cpu_slots = (cpu_blocks[:, None] * block_size + offsets).reshape(-1)
                slots = self.routed_experts.routed_experts_by_slot
                if is_store:
                    self.offloaded_experts[cpu_slots] = slots[gpu_slots]
                else:
                    slots[gpu_slots] = self.offloaded_experts[cpu_slots]
                return
            gpu_offset += count
            cpu_offset += cpu_count
