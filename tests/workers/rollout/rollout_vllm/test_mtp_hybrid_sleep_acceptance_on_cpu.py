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

import asyncio
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from packaging import version

pytest.importorskip("ray")

from verl.trainer.ppo.trainer_utils import compute_spec_decode_metrics

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_vllm_async_server():
    """Import vllm_async_server directly when vllm is installed, or with scoped stubs on CPU."""
    try:
        importlib.import_module("vllm")
    except ImportError:
        pass
    else:
        from verl.workers.rollout.vllm_rollout import vllm_async_server as real_mod

        return real_mod

    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    spec = importlib.util.spec_from_file_location("_verl_test_vllm_async_server", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    stub_names = [
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.cli",
        "vllm.entrypoints.cli.serve",
        "vllm.engine",
        "vllm.engine.arg_utils",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.api_server",
        "vllm.entrypoints.openai.parser",
        "vllm.entrypoints.openai.parser.harmony_utils",
        "vllm.inputs",
        "vllm.lora",
        "vllm.lora.request",
        "vllm.outputs",
        "vllm.usage",
        "vllm.usage.usage_lib",
        "vllm.utils",
        "vllm.utils.argparse_utils",
        "vllm.v1",
        "vllm.v1.engine",
        "vllm.v1.engine.async_llm",
        "vllm.v1.engine.exceptions",
        "vllm.v1.metrics",
        "vllm.v1.metrics.loggers",
        "verl.workers.rollout.vllm_rollout.utils",
    ]
    stubs = {name: types.ModuleType(name) for name in stub_names}
    stubs["vllm"].__version__ = "0.11.0"
    stubs["vllm"].SamplingParams = object
    stubs["vllm"].entrypoints = stubs["vllm.entrypoints"]
    stubs["vllm.entrypoints"].cli = stubs["vllm.entrypoints.cli"]
    stubs["vllm.entrypoints.cli"].serve = stubs["vllm.entrypoints.cli.serve"]
    stubs["vllm.entrypoints.cli.serve"].run_headless = None
    stubs["vllm.engine.arg_utils"].AsyncEngineArgs = object
    stubs["vllm.entrypoints.openai.api_server"].build_app = None
    stubs["vllm.entrypoints.openai.api_server"].init_app_state = None
    stubs["vllm.entrypoints.openai.parser.harmony_utils"].get_encoding = None
    stubs["vllm.inputs"].TokensPrompt = object
    stubs["vllm.lora.request"].LoRARequest = object
    stubs["vllm.outputs"].STREAM_FINISHED = object()
    stubs["vllm.outputs"].RequestOutput = object
    stubs["vllm.usage.usage_lib"].UsageContext = object
    stubs["vllm.utils.argparse_utils"].FlexibleArgumentParser = object
    stubs["vllm.v1.engine.async_llm"].AsyncLLM = object
    stubs["vllm.v1.engine.exceptions"].EngineDeadError = RuntimeError
    stubs["vllm.v1.metrics.loggers"].PrometheusStatLogger = object

    utils_stub = stubs["verl.workers.rollout.vllm_rollout.utils"]
    utils_stub.VLLM_LORA_INT_ID = 123
    utils_stub.VLLM_LORA_NAME = "123"
    utils_stub.VLLM_LORA_PATH = "simon_lora_path"
    utils_stub.SuppressSignalInThread = None
    utils_stub.build_cli_args_from_config = None
    utils_stub.build_mtp_speculative_config = None
    utils_stub.extract_prompt_logprobs = None
    utils_stub.get_vllm_max_lora_rank = None

    saved = {name: sys.modules.get(name) for name in stub_names}
    try:
        sys.modules.update(stubs)
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


vllm_async_server = _load_vllm_async_server()


class _FakeMtpEngine:
    """Minimal model of the drafter failure mode seen in hybrid sleep."""

    def __init__(self):
        self.mtp_drafter_available = True
        self.sleep_levels_that_discard_mtp_drafter = {2}

    async def sleep(self, level: int):
        if level in self.sleep_levels_that_discard_mtp_drafter:
            self.mtp_drafter_available = False

    async def reset_encoder_cache(self):
        pass

    def sync_actor_weights(self):
        pass

    def generate_spec_decode_stats(self):
        num_draft_tokens = 3
        num_accepted_tokens = num_draft_tokens if self.mtp_drafter_available else 0
        num_verify_steps = 1
        return num_draft_tokens, num_accepted_tokens, num_verify_steps


def test_mtp_hybrid_sleep_keeps_drafter_available_for_nonzero_acceptance(monkeypatch):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(
        mtp=SimpleNamespace(enable=True, enable_rollout=True),
        external_kv_cache_invalidation="reset",
    )
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server.engine = _FakeMtpEngine()

    asyncio.run(server._sleep_hybrid())
    server.engine.sync_actor_weights()
    drafts, accepts, verifies = server.engine.generate_spec_decode_stats()
    metrics = compute_spec_decode_metrics(
        spec_drafts=np.array([drafts]),
        spec_accepts=np.array([accepts]),
        spec_verifies=np.array([verifies]),
    )

    assert metrics["rollout/spec_accept_rate"] > 0.0
    assert metrics["rollout/spec_accept_length"] > 1.0


@pytest.mark.parametrize(
    "speculative_config",
    [
        {"model": "Qwen/Qwen3-0.6B-Draft", "num_speculative_tokens": 3},
        {"draft_model": "Qwen/Qwen3-0.6B-Draft", "num_speculative_tokens": 3},
        {"method": "eagle", "model": "Qwen/Qwen3-0.6B-Eagle", "num_speculative_tokens": 3},
        {"method": "eagle3", "num_speculative_tokens": 3},
        '{"model": "Qwen/Qwen3-0.6B-Draft", "num_speculative_tokens": 3}',
        "eagle3",
        SimpleNamespace(model="Qwen/Qwen3-0.6B-Draft", method=None),
    ],
)
def test_separate_draft_model_hybrid_sleep_selects_level_1(monkeypatch, speculative_config):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(
        mtp=SimpleNamespace(enable=False, enable_rollout=False),
        engine_kwargs={"vllm": {"speculative_config": speculative_config}},
        external_kv_cache_invalidation="reset",
    )
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server.engine = _FakeMtpEngine()

    assert server._resolve_sleep_level() == 1

    asyncio.run(server._sleep_hybrid())
    server.engine.sync_actor_weights()
    drafts, accepts, verifies = server.engine.generate_spec_decode_stats()
    metrics = compute_spec_decode_metrics(
        spec_drafts=np.array([drafts]),
        spec_accepts=np.array([accepts]),
        spec_verifies=np.array([verifies]),
    )

    assert metrics["rollout/spec_accept_rate"] > 0.0
    assert metrics["rollout/spec_accept_length"] > 1.0


@pytest.mark.parametrize(
    "engine_kwargs",
    [
        {},
        {"vllm": {}},
        {"vllm": {"speculative_config": None}},
        {"vllm": {"speculative_config": {}}},
        {"vllm": {"speculative_config": {"num_speculative_tokens": 3}}},
        {"vllm": {"speculative_config": {"method": "ngram", "num_speculative_tokens": 3}}},
        {"vllm": {"speculative_config": {"model": "[ngram]", "num_speculative_tokens": 3}}},
        {"vllm": {"speculative_config": {"method": "ngram_gpu", "num_speculative_tokens": 3}}},
        {"vllm": {"speculative_config": {"method": "suffix", "num_speculative_tokens": 3}}},
        {"vllm": {"speculative_config": '{"method": "ngram", "num_speculative_tokens": 3}'}},
        {"vllm": {"speculative_config": "ngram"}},
        {"vllm": {"speculative_config": "   "}},
    ],
)
def test_resolve_sleep_level_without_separate_draft_weights_returns_level_2(monkeypatch, engine_kwargs):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(
        mtp=SimpleNamespace(enable=False, enable_rollout=False),
        engine_kwargs=engine_kwargs,
    )
    server.model_config = SimpleNamespace(lora_rank=0, lora={})

    assert server._resolve_sleep_level() == 2


@pytest.mark.parametrize(
    "engine_kwargs",
    [
        # vLLM also accepts the dashed CLI spelling; a draft model set there must
        # not be discarded by a level-2 sleep.
        {"vllm": {"speculative-config": {"model": "Qwen/Qwen3-0.6B-Draft", "num_speculative_tokens": 3}}},
        {"vllm": {"speculative-config": '{"model": "Qwen/Qwen3-0.6B-Draft", "num_speculative_tokens": 3}'}},
        {"vllm": {"speculative-config": "eagle3"}},
        SimpleNamespace(vllm=SimpleNamespace(**{"speculative-config": {"model": "Qwen/Qwen3-0.6B-Draft"}})),
    ],
)
def test_hyphenated_speculative_config_selects_level_1(monkeypatch, engine_kwargs):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(
        mtp=SimpleNamespace(enable=False, enable_rollout=False),
        engine_kwargs=engine_kwargs,
    )
    server.model_config = SimpleNamespace(lora_rank=0, lora={})

    assert server._resolve_sleep_level() == 1


@pytest.mark.parametrize(
    "engine_kwargs, expected",
    [
        # The underscore key is the one verl sets, so it wins when both are present.
        (
            {
                "vllm": {
                    "speculative_config": {"method": "ngram"},
                    "speculative-config": {"model": "Qwen/Qwen3-0.6B-Draft"},
                }
            },
            2,
        ),
        (
            {
                "vllm": {
                    "speculative_config": {"model": "Qwen/Qwen3-0.6B-Draft"},
                    "speculative-config": {"method": "ngram"},
                }
            },
            1,
        ),
        (
            SimpleNamespace(
                vllm=SimpleNamespace(
                    speculative_config={"method": "ngram"},
                    **{"speculative-config": {"model": "Qwen/Qwen3-0.6B-Draft"}},
                )
            ),
            2,
        ),
    ],
)
def test_underscore_speculative_config_takes_precedence(monkeypatch, engine_kwargs, expected):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(
        mtp=SimpleNamespace(enable=False, enable_rollout=False),
        engine_kwargs=engine_kwargs,
    )
    server.model_config = SimpleNamespace(lora_rank=0, lora={})

    assert server._resolve_sleep_level() == expected


_SPEC_KEYS = ("speculative_config", "speculative-config")
_SPEC_DOTTED_PREFIXES = ("speculative_config.", "speculative-config.")


def _make_sleep_server(monkeypatch, engine_kwargs):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(
        mtp=SimpleNamespace(enable=False, enable_rollout=False),
        engine_kwargs=engine_kwargs,
    )
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    return server


_DRAFT = {"model": "Qwen/Qwen3-0.6B-Draft", "num_speculative_tokens": 3}
_NGRAM = {"method": "ngram"}


@pytest.mark.parametrize(
    "engine_kwargs, expected_level, expected_effective",
    [
        # both exact spellings must resolve to one value; underscore wins
        ({"vllm": {"speculative_config": _NGRAM, "speculative-config": _DRAFT}}, 2, _NGRAM),
        ({"vllm": {"speculative-config": _DRAFT, "speculative_config": _NGRAM}}, 2, _NGRAM),
        ({"vllm": {"speculative_config": _DRAFT, "speculative-config": _NGRAM}}, 1, _DRAFT),
        # dotted sub-fields are part of the effective configuration
        (
            {
                "vllm": {
                    "speculative_config.model": "Qwen/Qwen3-0.6B-Draft",
                    "speculative_config.num_speculative_tokens": 3,
                }
            },
            1,
            _DRAFT,
        ),
        ({"vllm": {"speculative-config.model": "Qwen/Qwen3-0.6B-Draft"}}, 1, {"model": "Qwen/Qwen3-0.6B-Draft"}),
        (
            {"vllm": {"speculative_config": {"method": "eagle"}, "speculative_config.model": "Qwen/Qwen3-0.6B-Draft"}},
            1,
            {"method": "eagle", "model": "Qwen/Qwen3-0.6B-Draft"},
        ),
        ({"vllm": {"speculative_config.method": "ngram"}}, 2, _NGRAM),
        # a plain weightless config keeps level 2
        ({"vllm": {"speculative_config": _NGRAM}}, 2, _NGRAM),
        # single dashed spelling is a valid configuration on its own
        ({"vllm": {"speculative-config": _DRAFT}}, 1, _DRAFT),
    ],
)
def test_normalized_speculative_config_drives_sleep_level(
    monkeypatch, engine_kwargs, expected_level, expected_effective
):
    """Launch prep and sleep resolution read the same normalized configuration."""
    server = _make_sleep_server(monkeypatch, engine_kwargs)
    launched_kwargs = dict(engine_kwargs["vllm"])

    server._apply_speculative_config_normalization(launched_kwargs)

    spec_entries = [key for key in launched_kwargs if key in _SPEC_KEYS or key.startswith(_SPEC_DOTTED_PREFIXES)]
    assert len(spec_entries) == 1
    assert launched_kwargs[spec_entries[0]] == expected_effective
    # The resolver reads the raw engine kwargs and must reach the same value.
    assert server._effective_speculative_config() == expected_effective
    assert server._resolve_sleep_level() == expected_level


def test_single_spelling_configuration_is_not_rewritten(monkeypatch):
    server = _make_sleep_server(monkeypatch, {"vllm": {"speculative-config": _DRAFT}})
    launched_kwargs = dict(server.config.engine_kwargs["vllm"])

    server._apply_speculative_config_normalization(launched_kwargs)

    assert launched_kwargs == {"speculative-config": _DRAFT}


def test_preprocess_engine_kwargs_collapses_both_spellings(monkeypatch):
    server = _make_sleep_server(monkeypatch, {"vllm": {"speculative_config": _NGRAM, "speculative-config": _DRAFT}})
    engine_kwargs = dict(server.config.engine_kwargs["vllm"])

    server._preprocess_engine_kwargs(engine_kwargs)

    # Other preprocessing steps (e.g. the old-vLLM multimodal workaround) may add
    # unrelated keys; only the speculative-config spellings must be collapsed.
    assert engine_kwargs["speculative_config"] == _NGRAM
    assert not any(
        key in _SPEC_KEYS or key.startswith(_SPEC_DOTTED_PREFIXES)
        for key in engine_kwargs
        if key != "speculative_config"
    )


def test_resolve_sleep_level_preserves_lora_and_npu_guards(monkeypatch):
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(
        mtp=SimpleNamespace(enable=False, enable_rollout=False),
        engine_kwargs={},
    )

    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)
    server.model_config = SimpleNamespace(lora_rank=8, lora={"merge": False})
    assert server._resolve_sleep_level() == 1

    server.model_config = SimpleNamespace(lora_rank=8, lora={"merge": True})
    assert server._resolve_sleep_level() == 2

    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: True)
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    # vllm-ascend needs level 1 below 0.30.0; the guard no longer applies at/above
    # 0.30.0. Pin the version explicitly so the test does not depend on whichever
    # vLLM happens to be installed in the test environment.
    monkeypatch.setattr(vllm_async_server, "_VLLM_VERSION", version.parse("0.29.0"))
    assert server._resolve_sleep_level() == 1
    monkeypatch.setattr(vllm_async_server, "_VLLM_VERSION", version.parse("0.30.0"))
    assert server._resolve_sleep_level() == 2
