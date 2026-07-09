# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _install_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_vllm_async_server():
    module_name = "_test_vllm_async_server"
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"

    fake_module_names = [
        "ray",
        "ray.actor",
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.cli.serve",
        "vllm.engine.arg_utils",
        "vllm.entrypoints.openai.api_server",
        "vllm.entrypoints.openai.parser",
        "vllm.entrypoints.openai.parser.harmony_utils",
        "vllm.inputs",
        "vllm.lora.request",
        "vllm.outputs",
        "vllm.usage.usage_lib",
        "vllm.v1",
        "vllm.v1.engine",
        "vllm.v1.engine.async_llm",
        "vllm.utils",
        "vllm.utils.argparse_utils",
        "verl.plugin.platform",
        "verl.utils.config",
        "verl.utils.device",
        "verl.utils.net_utils",
        "verl.utils.profiler",
        "verl.utils.tokenizer",
        "verl.utils.vllm.vllm_fp8_utils",
        "verl.workers.config",
        "verl.workers.rollout.replica",
        "verl.workers.rollout.utils",
        "verl.workers.rollout.vllm_rollout.utils",
    ]
    saved = {name: sys.modules.get(name) for name in fake_module_names + [module_name]}
    try:
        fake_ray_actor = _install_module("ray.actor", ActorHandle=object)
        fake_ray = _install_module("ray", actor=fake_ray_actor)
        fake_ray.remote = lambda obj=None, **_: obj if obj is not None else (lambda inner: inner)
        fake_ray.util = SimpleNamespace(scheduling_strategies=SimpleNamespace(NodeAffinitySchedulingStrategy=object))

        fake_vllm = _install_module("vllm", __version__="0.99.0")
        _install_module("vllm.entrypoints")
        fake_cli_serve = _install_module("vllm.entrypoints.cli.serve")
        fake_vllm.entrypoints = SimpleNamespace(cli=SimpleNamespace(serve=fake_cli_serve))
        fake_cli_serve.run_headless = lambda *_, **__: None
        _install_module("vllm.engine.arg_utils", AsyncEngineArgs=object)
        _install_module("vllm.entrypoints.openai")
        _install_module("vllm.entrypoints.openai.api_server", build_app=lambda *_, **__: None, init_app_state=None)
        _install_module("vllm.entrypoints.openai.parser")
        _install_module("vllm.entrypoints.openai.parser.harmony_utils", get_encoding=lambda: None)
        _install_module("vllm.inputs", TokensPrompt=dict)
        _install_module("vllm.lora.request", LoRARequest=object)
        _install_module("vllm.outputs", RequestOutput=object)
        _install_module("vllm.usage.usage_lib", UsageContext=SimpleNamespace(OPENAI_API_SERVER="openai"))
        _install_module("vllm.v1")
        _install_module("vllm.v1.engine")
        _install_module("vllm.v1.engine.async_llm", AsyncLLM=object)
        _install_module("vllm.utils")
        _install_module("vllm.utils.argparse_utils", FlexibleArgumentParser=object)
        fake_vllm.SamplingParams = object

        fake_platform = SimpleNamespace(
            ray_resource_name=lambda: "GPU",
            rollout_env_vars=lambda: {},
            ray_noset_envvars=lambda: [],
        )
        _install_module("verl.plugin.platform", get_platform=lambda: fake_platform)
        _install_module("verl.utils.config", omega_conf_to_dataclass=lambda value: value)
        _install_module(
            "verl.utils.device",
            get_resource_name=lambda: "GPU",
            get_visible_devices_keyword=lambda: "CUDA_VISIBLE_DEVICES",
            is_torch_npu_available=lambda check_device=True: False,
        )
        _install_module(
            "verl.utils.net_utils",
            get_free_port=lambda *_, **__: (12345, None),
            is_valid_ipv6_address=lambda _: False,
        )
        _install_module(
            "verl.utils.profiler",
            DistProfiler=SimpleNamespace(annotate=lambda *_, **__: (lambda fn: fn)),
            build_vllm_profiler_args=lambda *_, **__: [],
        )
        _install_module("verl.utils.tokenizer", normalize_token_ids=lambda token_ids: token_ids)
        _install_module("verl.utils.vllm.vllm_fp8_utils", apply_vllm_fp8_patches=lambda: None)
        _install_module("verl.workers.config", HFModelConfig=object, RolloutConfig=object)

        class _RolloutMode(Enum):
            HYBRID = "hybrid"
            COLOCATED = "colocated"
            STANDALONE = "standalone"

        class _RolloutReplica:
            def __init__(self, *_, **__):
                pass

        _install_module(
            "verl.workers.rollout.replica",
            RolloutMode=_RolloutMode,
            RolloutReplica=_RolloutReplica,
            TokenOutput=SimpleNamespace,
        )
        _install_module(
            "verl.workers.rollout.utils",
            get_max_position_embeddings=lambda *_: None,
            qwen2_5_vl_dedup_image_tokens=lambda *_, **__: None,
            run_uvicorn=lambda *_, **__: None,
        )
        _install_module(
            "verl.workers.rollout.vllm_rollout.utils",
            VLLM_LORA_INT_ID=1,
            VLLM_LORA_NAME="lora",
            VLLM_LORA_PATH="lora",
            SuppressSignalInThread=object,
            build_cli_args_from_config=lambda *_: [],
            build_mtp_speculative_config=lambda *_: None,
            extract_prompt_logprobs=lambda *_, **__: None,
            get_vllm_max_lora_rank=lambda rank: rank,
        )

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


vllm_async_server = _load_vllm_async_server()


class _FakeKvCacheEngine:
    def __init__(self):
        self.calls = []

    async def sleep(self, level: int):
        self.calls.append(("sleep", level))

    async def wake_up(self, tags=None):
        self.calls.append(("wake_up", tags))

    async def reset_encoder_cache(self):
        self.calls.append(("reset_encoder_cache", None))


class _FakeRemoteMethod:
    def __init__(self, func):
        self._func = func

    def remote(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class _FakeServerHandle:
    def __init__(self):
        self.calls = []
        self.wait_for_requests_to_drain = _FakeRemoteMethod(self._wait_for_requests_to_drain)
        self.release_kv_cache = _FakeRemoteMethod(self._release_kv_cache)
        self.resume_kv_cache = _FakeRemoteMethod(self._resume_kv_cache)

    async def _wait_for_requests_to_drain(self):
        self.calls.append("wait_for_requests_to_drain")

    async def _release_kv_cache(self):
        self.calls.append("release_kv_cache")

    async def _resume_kv_cache(self):
        self.calls.append("resume_kv_cache")


def test_release_kv_cache_keeps_vllm_weights_awake():
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.node_rank = 0
    server.config = SimpleNamespace(free_cache_engine=True)
    server.engine = _FakeKvCacheEngine()

    asyncio.run(server.release_kv_cache())

    assert server.engine.calls == [
        ("sleep", 1),
        ("wake_up", ["weights"]),
    ]


def test_release_kv_cache_skips_when_disabled():
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.node_rank = 0
    server.config = SimpleNamespace(free_cache_engine=False)
    server.engine = _FakeKvCacheEngine()

    asyncio.run(server.release_kv_cache())

    assert server.engine.calls == []


def test_resume_kv_cache_wakes_only_vllm_kv_cache():
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.node_rank = 0
    server.config = SimpleNamespace(free_cache_engine=True)
    server.engine = _FakeKvCacheEngine()

    asyncio.run(server.resume_kv_cache())

    assert server.engine.calls == [("wake_up", ["kv_cache"])]


def test_hybrid_lora_sleep_releases_only_vllm_kv_cache():
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(mtp=SimpleNamespace(enable=False, enable_rollout=False))
    server.model_config = SimpleNamespace(lora_rank=8, lora={"merge": False})
    server.engine = _FakeKvCacheEngine()

    asyncio.run(server._sleep_hybrid())

    assert server.engine.calls == [
        ("sleep", 1),
        ("wake_up", ["weights"]),
        ("reset_encoder_cache", None),
    ]


def test_vllm_replica_release_and_resume_kv_cache_forward_to_servers():
    first_server = _FakeServerHandle()
    second_server = _FakeServerHandle()
    replica = object.__new__(vllm_async_server.vLLMReplica)
    replica.servers = [first_server, second_server]

    asyncio.run(replica.release_kv_cache())
    asyncio.run(replica.resume_kv_cache())

    assert first_server.calls == [
        "wait_for_requests_to_drain",
        "release_kv_cache",
        "resume_kv_cache",
    ]
    assert second_server.calls == [
        "release_kv_cache",
        "resume_kv_cache",
    ]
