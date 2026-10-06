# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import numpy as np

from vllm.v1.kv_offload.base import OffloadKey

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsManager,
    )


class OffloadedRoutedExperts:
    """CPU routes with the same key lifetime as native CPU KV chunks."""

    def __init__(self, manager: "RoutedExpertsManager") -> None:
        self.manager = manager
        self.cache: dict[OffloadKey, np.ndarray] = {}

    def snapshot(self, blocks: dict[OffloadKey, int]) -> dict[OffloadKey, np.ndarray]:
        size = self.manager.block_size
        buffer = self.manager.routed_experts_by_slot
        return {
            key: buffer[block * size : (block + 1) * size].copy()
            for key, block in blocks.items()
        }

    def restore(
        self, blocks: dict[OffloadKey, int], data: dict[OffloadKey, np.ndarray]
    ) -> None:
        size = self.manager.block_size
        buffer = self.manager.routed_experts_by_slot
        for key, block in blocks.items():
            buffer[block * size : (block + 1) * size] = data[key]
