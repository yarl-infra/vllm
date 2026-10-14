# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run inside the patched container with pytest; these tests do not use a GPU."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.routed_experts import (
    RoutedExpertsOffloadManager,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.v1.kv_cache_interface import KVCacheGroupRole
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


def manager(dtype=np.uint8, group=0, chunks=1):
    routes = RoutedExpertsManager.__new__(RoutedExpertsManager)
    routes.attn_gid = group
    routes.block_size = 4
    routes.routed_experts_by_slot = np.zeros((64, 2, 2), dtype=dtype)
    return RoutedExpertsOffloadManager(routes, 8, chunks)


def job(gpu_ids, cpu_ids, *, store=True, sizes=None, indices=None):
    sizes = [len(gpu_ids)] if sizes is None else sizes
    indices = [0] * len(sizes) if indices is None else indices
    gpu = GPULoadStoreSpec(gpu_ids, group_sizes=sizes, block_indices=indices)
    cpu = CPULoadStoreSpec(cpu_ids)
    return TransferJob("request", gpu if store else cpu, cpu if store else gpu)


def step(mgr, *, stores=None, loads=None, flush=(), forward=None):
    mgr.on_schedule(OffloadingConnectorMetadata(loads or {}, stores or {}, set(flush)))
    mgr.before_step()
    if forward:
        forward()
    mgr.after_step()


def config_stub(**kwargs: Any) -> VllmConfig:
    return cast(VllmConfig, SimpleNamespace(**kwargs))


def test_connector_sizes_routing_buffer_from_cpu_chunks():
    connector: Any = OffloadingConnector.__new__(OffloadingConnector)
    connector.connector_scheduler = object()
    connector._kv_cache_config = SimpleNamespace(
        hisparse_host_num_blocks=None, kv_cache_groups=[object()]
    )
    spec = CPUOffloadingSpec.__new__(CPUOffloadingSpec)
    spec.num_chunks = 8
    spec.blocks_per_chunk = 2
    connector._offloading_spec = spec
    routes = manager().routed_experts

    mgr = connector.create_routed_experts_offload_manager(routes)

    assert mgr is not None
    assert mgr.offloaded_experts.shape == (8 * 2 * routes.block_size, 2, 2)


def test_connector_skips_routing_when_attention_group_is_not_offloaded():
    connector: Any = OffloadingConnector.__new__(OffloadingConnector)
    connector.connector_scheduler = object()
    connector._kv_cache_config = SimpleNamespace(
        hisparse_host_num_blocks=8,
        kv_cache_groups=[
            SimpleNamespace(role=KVCacheGroupRole.HISPARSE_SOURCE),
            SimpleNamespace(role=KVCacheGroupRole.HISPARSE_INDEXER),
        ],
    )
    routes = manager().routed_experts

    assert connector.create_routed_experts_offload_manager(routes) is None


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_evicted_gpu_prefix_is_restored_exactly(dtype):
    mgr = manager(dtype)
    routes = mgr.routed_experts
    expected = np.arange(32, dtype=dtype).reshape(8, 2, 2)
    if dtype == np.uint16:
        expected += 300
    routes.store_batch(expected, np.arange(4, 12))
    step(mgr, stores={1: job([1, 2], [3, 5])})
    routes.routed_experts_by_slot[:] = 99  # GPU eviction / physical slot reuse
    step(mgr, loads={2: job([7, 9], [3, 5], store=False)})
    np.testing.assert_array_equal(routes.get([7, 9], 8), expected)
    np.testing.assert_array_equal(routes.get([7, 9], 8, token_start=3), expected[3:])
    assert mgr.offloaded_experts.dtype == dtype


def test_regular_store_sees_current_forward_and_load_only_step_works():
    mgr = manager()
    slots = mgr.routed_experts.routed_experts_by_slot
    step(
        mgr,
        stores={1: job([2], [0])},
        forward=lambda: slots.__setitem__(slice(8, 12), 17),
    )
    step(mgr, loads={2: job([3], [0], store=False)})
    np.testing.assert_array_equal(slots[12:16], 17)
    assert not mgr._steps


def test_flush_precedes_load_and_forward_despite_worker_mutation():
    mgr = manager()
    slots = mgr.routed_experts.routed_experts_by_slot
    slots[4:8] = 11
    slots[8:12] = 22
    step(mgr, stores={1: job([2], [0])})
    metadata = OffloadingConnectorMetadata(
        {3: job([1], [0], store=False)},
        {2: job([1], [1]), 4: job([3], [2])},
        {2},
    )
    mgr.on_schedule(metadata)
    metadata.store_jobs.pop(2)  # Actual worker handle_preemptions behavior.
    assert metadata.jobs_to_flush is not None
    metadata.jobs_to_flush.clear()
    mgr.before_step()
    np.testing.assert_array_equal(slots[4:8], 22)
    slots[12:16] = 33
    mgr.after_step()
    step(mgr, loads={5: job([4, 5], [1, 2], store=False)})
    np.testing.assert_array_equal(slots[16:20], 11)
    np.testing.assert_array_equal(slots[20:24], 33)


def test_hybrid_groups_and_unaligned_multiblock_chunk_load():
    mgr = manager(group=1, chunks=3)
    slots = mgr.routed_experts.routed_experts_by_slot
    # Group 0 uses two chunks, group 1 one, and group 2 one.
    for bid in range(1, 10):
        slots[bid * 4 : bid * 4 + 4] = bid
    step(
        mgr,
        stores={
            1: job(
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [5, 6, 2, 4],
                sizes=[4, 3, 2],
                indices=[0, 0, 0],
            )
        },
    )
    slots[:] = 99
    # Skip the first GPU block in the matching attention CPU chunk.
    step(
        mgr,
        loads={
            2: job(
                [1, 10, 11, 12],
                [6, 2, 4],
                store=False,
                sizes=[1, 2, 1],
                indices=[3, 1, 0],
            )
        },
    )
    np.testing.assert_array_equal(slots[40:44], 6)
    np.testing.assert_array_equal(slots[44:48], 7)
    np.testing.assert_array_equal(slots[4:8], 99)  # SSM group is not routing.
    np.testing.assert_array_equal(slots[48:52], 99)


def test_zero_sized_preceding_group_does_not_consume_cpu_chunk():
    mgr = manager(group=1, chunks=2)
    slots = mgr.routed_experts.routed_experts_by_slot
    slots[4:12] = 42
    step(mgr, stores={1: job([1, 2], [5], sizes=[0, 2], indices=[1, 0])})
    slots[:] = 0
    step(mgr, loads={2: job([3], [5], store=False, sizes=[0, 1], indices=[1, 1])})
    np.testing.assert_array_equal(slots[12:16], 42)


def test_async_fifo_and_cpu_slot_reuse_keep_storage_bounded():
    mgr = manager()
    slots = mgr.routed_experts.routed_experts_by_slot
    original_buffer = mgr.offloaded_experts
    # Schedule ahead, then consume the corresponding forward results in order.
    for i in range(20):
        mgr.on_schedule(OffloadingConnectorMetadata({}, {i: job([1], [0])}, set()))
    for i in range(20):
        mgr.before_step()
        slots[4:8] = i
        mgr.after_step()
    # Same physical CPU ID can be reused after eviction or cache reset.
    step(mgr, loads={20: job([4], [0], store=False)})
    np.testing.assert_array_equal(slots[16:20], 19)
    assert mgr.offloaded_experts is original_buffer
    assert mgr.offloaded_experts.nbytes == 8 * 4 * 2 * 2


@pytest.mark.parametrize(
    "changes",
    [
        {"kv_connector": "NixlConnector", "kv_role": "kv_producer"},
        {"kv_connector": "NixlConnector", "kv_role": "kv_consumer"},
        {"kv_connector": "NixlConnector"},
        {"kv_connector": "LMCacheMPConnector"},
        {"kv_connector": "SimpleCPUOffloadConnector"},
        {"kv_role": "kv_producer"},
        {"kv_role": "kv_consumer"},
        {"kv_connector_module_path": "custom.connector"},
        {"kv_connector_extra_config": {"spec_name": "TieringOffloadingSpec"}},
        {"kv_connector_extra_config": {"spec_module_path": "custom.spec"}},
    ],
)
def test_unsupported_connectors_remain_rejected(changes, monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    config = config_stub(
        model_config=SimpleNamespace(enable_return_routed_experts=True),
        kv_transfer_config=KVTransferConfig(
            **dict(
                {"kv_connector": "OffloadingConnector", "kv_role": "kv_both"}, **changes
            )
        ),
    )
    with pytest.raises(ValueError, match="native CPU KV cache offload"):
        VllmConfig._verify_kv_transfer_compat(config)


@pytest.mark.parametrize("extra", [{}, {"spec_name": "CPUOffloadingSpec"}])
def test_native_cpu_connector_is_allowed(extra, monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    config = config_stub(
        model_config=SimpleNamespace(enable_return_routed_experts=True),
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config=extra,
        ),
    )
    VllmConfig._verify_kv_transfer_compat(config)


def test_unrelated_connector_is_unchanged_when_capture_disabled(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    config = config_stub(
        model_config=SimpleNamespace(enable_return_routed_experts=False),
        kv_transfer_config=KVTransferConfig(
            kv_connector="NixlConnector", kv_role="kv_both"
        ),
    )
    VllmConfig._verify_kv_transfer_compat(config)


@pytest.mark.parametrize(
    "backend,simple,allowed",
    [
        ("native", False, True),
        ("native", True, False),
        ("lmcache", False, False),
    ],
)
def test_derived_offload_flags_are_validated(backend, simple, allowed, monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setenv("VLLM_USE_SIMPLE_KV_OFFLOAD", str(int(simple)))
    config = config_stub(
        model_config=SimpleNamespace(enable_return_routed_experts=True),
        cache_config=SimpleNamespace(
            kv_offloading_size=10, kv_offloading_backend=backend
        ),
        kv_transfer_config=None,
    )
    VllmConfig._post_init_kv_transfer_config(config)
    if allowed:
        VllmConfig._verify_kv_transfer_compat(config)
    else:
        with pytest.raises(ValueError, match="native CPU KV cache offload"):
            VllmConfig._verify_kv_transfer_compat(config)
