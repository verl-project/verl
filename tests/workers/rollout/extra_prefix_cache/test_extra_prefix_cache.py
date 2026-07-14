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

import pytest

from verl.workers.rollout.extra_prefix_cache.config import ExtraPrefixCacheConfig, enabled
from verl.workers.rollout.extra_prefix_cache.controller import ExtraPrefixCacheController, compute_store_token_limit
from verl.workers.rollout.extra_prefix_cache.prefix_provider import ExplicitPrefixProvider
from verl.workers.rollout.extra_prefix_cache.protocol import (
    PrefixMetadata,
    build_cache_salt,
    build_request_id,
    parse_request_id,
)


def test_config_is_disabled_by_default() -> None:
    cfg = ExtraPrefixCacheConfig.from_any(None)
    assert not cfg.enable
    assert not enabled(None)


def test_explicit_provider_requires_len_and_fingerprint() -> None:
    provider = ExplicitPrefixProvider()
    assert provider.resolve([1, 2, 3], None) is None
    assert provider.resolve([1, 2, 3], {"stable_prefix_token_len": 2}) is None
    assert provider.resolve([1, 2, 3], {"stable_prefix_token_len": 4, "stable_prefix_fingerprint": "abc"}) is None

    metadata = provider.resolve(
        [1, 2, 3],
        {"stable_prefix_token_len": 2, "stable_prefix_fingerprint": "abc", "prefix_source": "unit"},
    )
    assert metadata is not None
    assert metadata.stable_prefix_token_len == 2
    assert metadata.stable_prefix_fingerprint == "abc"
    assert metadata.prefix_source == "unit"


def test_store_token_limit_aligns_down_to_chunk() -> None:
    assert compute_store_token_limit(130, 64) == 128
    assert compute_store_token_limit(63, 64) == 0
    assert compute_store_token_limit(63, 0) == 63


def test_request_id_policy_roundtrip() -> None:
    request_id = build_request_id(read=True, write=False, store_token_limit=128, base_request_id="abc")
    policy = parse_request_id(request_id)
    assert request_id == "vepc__r1__w0__l128__abc"
    assert policy.read
    assert not policy.write
    assert policy.store_token_limit == 128
    assert policy.tagged

    untagged = parse_request_id("plain", allow_untagged=False)
    assert not untagged.read
    assert not untagged.write
    assert not untagged.tagged


def test_salt_is_epoch_and_prefix_scoped() -> None:
    metadata = PrefixMetadata(stable_prefix_token_len=4, stable_prefix_fingerprint="sys-a")
    cfg = ExtraPrefixCacheConfig.from_any(
        {"enable": True, "namespace": "unit", "model_cache_epoch": "epoch0", "model_namespace": "model-a"}
    )
    salt = build_cache_salt(cfg, metadata)
    assert salt.startswith("unit:")
    assert ":epoch0:" in salt
    assert salt.endswith(":pfxsys-a")

    cfg_next = ExtraPrefixCacheConfig.from_any({**cfg.__dict__, "runtime_model_cache_epoch": "step-1"})
    assert build_cache_salt(cfg_next, metadata) != salt


@pytest.mark.asyncio
async def test_controller_builds_warmup_then_dedups() -> None:
    controller = ExtraPrefixCacheController(
        {
            "enable": True,
            "namespace": "unit",
            "model_cache_epoch": "epoch0",
            "model_namespace": "model-a",
            "chunk_size": 4,
        }
    )
    metadata = {"stable_prefix_token_len": 10, "stable_prefix_fingerprint": "abc"}
    prepared = await controller.prepare(prompt_ids=list(range(20)), metadata=metadata)
    assert prepared.enabled
    assert prepared.cache_salt is not None
    assert prepared.backend_request_id is not None
    assert prepared.store_token_limit == 8
    assert prepared.warmup_prompt_ids == list(range(8))
    assert prepared.warmup_backend_request_id is not None
    assert parse_request_id(prepared.backend_request_id).read
    assert not parse_request_id(prepared.backend_request_id).write
    assert not parse_request_id(prepared.warmup_backend_request_id).read
    assert parse_request_id(prepared.warmup_backend_request_id).write

    second = await controller.prepare(prompt_ids=list(range(20)), metadata=metadata)
    assert second.enabled
    assert second.cache_salt == prepared.cache_salt
    assert second.warmup_prompt_ids is None
    assert second.warmup_backend_request_id is None


@pytest.mark.asyncio
async def test_controller_disabled_or_missing_metadata_is_noop() -> None:
    disabled = ExtraPrefixCacheController({"enable": False})
    assert not (await disabled.prepare(prompt_ids=[1, 2], metadata=None)).enabled

    enabled_controller = ExtraPrefixCacheController({"enable": True})
    assert not (await enabled_controller.prepare(prompt_ids=[1, 2], metadata=None)).enabled
