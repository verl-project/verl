# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import importlib.util
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from verl.workers.rollout.sglang_rollout import utils
from verl.workers.rollout.sglang_rollout.utils import DEEPSEEK_V4_FUSION_GROUPS, get_named_tensor_buckets


def _tensor(nbytes):
    return torch.empty(nbytes, dtype=torch.uint8)


def _buckets(items, fusion_groups=()):
    async def collect():
        return [bucket async for bucket in get_named_tensor_buckets(iter(items), 1024, fusion_groups)]

    return asyncio.run(collect())


@pytest.mark.parametrize("group", DEEPSEEK_V4_FUSION_GROUPS)
def test_deepseek_v4_fusion_group_stays_in_one_bucket(group):
    prefix = "model.layers.7"
    items = [("filler", _tensor(1000)), *((prefix + suffix, _tensor(64)) for suffix in group)]
    buckets = _buckets(items, DEEPSEEK_V4_FUSION_GROUPS)
    names = [[name for name, _ in bucket] for bucket in buckets]
    assert any(all(prefix + suffix in bucket for suffix in group) for bucket in names)


def test_fusion_grouping_is_opt_in_for_non_deepseek_models():
    name = "model.layers.7.attn.wq_a.weight"
    assert [[item_name for item_name, _ in bucket] for bucket in _buckets([(name, _tensor(64))])] == [[name]]


def test_deepseek_v4_incomplete_group_fails_loudly():
    with pytest.raises(ValueError, match="never completed"):
        _buckets(
            [("model.layers.7.attn.wq_a.weight", _tensor(64))],
            DEEPSEEK_V4_FUSION_GROUPS,
        )


def test_overflow_yields_before_cloning_next_view(monkeypatch):
    compacted = []
    compact = utils._compact_for_bucket

    def track_compaction(tensor):
        compacted.append(tensor)
        return compact(tensor)

    monkeypatch.setattr(utils, "_compact_for_bucket", track_compaction)
    views = [torch.arange(8, dtype=torch.uint8)[:] for _ in range(2)]

    async def consume():
        buckets = get_named_tensor_buckets(iter(zip(("a", "b"), views, strict=True)), 8)
        first = await anext(buckets)
        assert len(compacted) == 1
        assert first[0][1]._base is None
        torch.testing.assert_close(first[0][1], views[0])
        second = await anext(buckets)
        assert len(compacted) == 2
        assert second[0][1]._base is None
        torch.testing.assert_close(second[0][1], views[1])
        await buckets.aclose()

    asyncio.run(consume())


def _load_rollout_without_sglang(monkeypatch):
    # Exercise the real adapter method on CPU; only the SGLang imports and
    # HTTP adapter are stand-ins. No engine, distributed group, or CUDA is needed.
    modules = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.entrypoints",
        "sglang.srt.entrypoints.engine",
        "sglang.srt.server_args",
        "sglang.srt.utils",
        "sglang.srt.weight_sync",
        "sglang.srt.weight_sync.utils",
        "verl.workers.rollout.sglang_rollout.http_server_engine",
    ):
        module = modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, module)
    modules["sglang.srt.server_args"].ServerArgs = object
    for name in (
        "MultiprocessingSerializer",
        "assert_pkg_version",
        "is_cuda",
        "set_prometheus_multiproc_dir",
        "set_ulimit",
    ):
        setattr(modules["sglang.srt.utils"], name, None)
    for name in ("_preprocess_tensor_for_update_weights", "update_weights"):
        setattr(modules["sglang.srt.weight_sync.utils"], name, None)
    modules["verl.workers.rollout.sglang_rollout.http_server_engine"].AsyncHttpServerAdapter = object
    path = Path(utils.__file__).with_name("sglang_rollout.py")
    spec = importlib.util.spec_from_file_location("_sglang_bucket_lifetime_rollout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapter_releases_consumed_bucket_after_await(monkeypatch):
    rollout = _load_rollout_without_sglang(monkeypatch)
    consumed = []
    updates = []

    async def init_adapter():
        pass

    async def update_weights(*, params_batch, **kwargs):
        refs = [weakref.ref(tensor) for _, tensor in params_batch]
        updates.append([name for name, _ in params_batch])
        await asyncio.sleep(0)
        assert all(ref() is not None for ref in refs)
        consumed.extend(refs)

    def source():
        for index in range(4):
            assert all(ref() is None for ref in consumed), "adapter retained a consumed bucket"
            yield str(index), torch.zeros(1 << 20, dtype=torch.uint8)[:]

    monkeypatch.setattr(rollout, "sgl_update_weights", update_weights)
    monkeypatch.setattr(rollout, "_uses_quantized_refit", lambda *args: False)
    monkeypatch.setattr(rollout, "_to_ipc_device", lambda tensor: tensor)
    adapter = SimpleNamespace(
        _init_server_adapter=init_adapter,
        _engine=None,
        config=SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1)),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test")),
        device_mesh=None,
    )
    asyncio.run(rollout.ServerAdapter.update_weights(adapter, source()))
    assert updates == [[str(index)] for index in range(4)]
    assert all(ref() is None for ref in consumed)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("async_source", [False, True])
def test_consumed_bucket_released_before_next_compaction(monkeypatch, fused, async_source):
    consumed = []
    compact = utils._compact_for_bucket

    def check_compaction(tensor):
        assert all(ref() is None for ref in consumed), "consumed bucket still retains tensors"
        return compact(tensor)

    monkeypatch.setattr(utils, "_compact_for_bucket", check_compaction)

    def source():
        if fused:
            for suffix in DEEPSEEK_V4_FUSION_GROUPS[0]:
                yield "model.layers.0" + suffix, torch.arange(4, dtype=torch.uint8)[:]
        else:
            yield "first", torch.arange(8, dtype=torch.uint8)[:]
        for name in ("second", "third", "fourth"):
            yield name, torch.arange(8, dtype=torch.uint8)[:]

    async def async_weights():
        for item in source():
            yield item

    async def consume():
        weights = async_weights() if async_source else source()
        async for bucket in get_named_tensor_buckets(weights, 8, DEEPSEEK_V4_FUSION_GROUPS):
            consumed.extend(weakref.ref(tensor) for _, tensor in bucket)
            del bucket
        assert all(ref() is None for ref in consumed)

    asyncio.run(consume())


@pytest.mark.parametrize("fused", [False, True])
def test_reused_full_span_views_preserve_values(fused):
    backing = torch.empty(8, dtype=torch.uint8)

    def source():
        for index in range(6):
            backing.fill_(index)
            name = "model.layers." + str(index // 2) + DEEPSEEK_V4_FUSION_GROUPS[0][index % 2] if fused else str(index)
            yield name, backing[:]

    async def consume():
        index = 0
        # A fused pair exceeds the bucket target and must remain indivisible.
        async for bucket in get_named_tensor_buckets(source(), 8, DEEPSEEK_V4_FUSION_GROUPS):
            assert len(bucket) == (2 if fused else 1)
            for _, tensor in bucket:
                torch.testing.assert_close(tensor, torch.full_like(tensor, index))
                index += 1
        assert index == 6

    asyncio.run(consume())
