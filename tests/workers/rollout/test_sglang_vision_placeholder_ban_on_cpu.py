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
"""SGLang counterpart of the vLLM vision-placeholder / OOV-tail mask: a static ban at the sampler."""

from __future__ import annotations

import asyncio
import functools
import pickle
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang")

from verl.utils.sglang import sampler_token_ban
from verl.utils.sglang.sampler_token_ban import install_sampler_token_ban, run_scheduler_process_with_token_ban

IMAGE_PAD = 151655
VIDEO_PAD = 151656
TOKENIZER_LEN = 151665  # Qwen2.5-VL: len(tokenizer) < config.vocab_size
LOGITS_WIDTH = 151680


def _no_mass(logits, columns) -> bool:
    # SGLang's sanitize_nan_logits may rewrite -inf to a large negative finite value, so assert on
    # probability rather than on the stored value.
    return bool((torch.softmax(logits.float(), dim=-1)[:, columns] == 0).all())


@pytest.fixture
def sampler_cls(monkeypatch):
    from sglang.srt.layers.sampler import Sampler

    monkeypatch.setattr(Sampler, "_preprocess_logits", Sampler._preprocess_logits)
    yield Sampler
    if hasattr(Sampler, sampler_token_ban._PATCH_ATTR):
        delattr(Sampler, sampler_token_ban._PATCH_ATTR)


def _preprocess(sampler_cls, logits):
    # Called the way Sampler.forward calls it, on the real SGLang class.
    sampling_info = SimpleNamespace(has_custom_logit_processor=False)
    return sampler_cls._preprocess_logits(object.__new__(sampler_cls), logits, sampling_info)


class TestSamplerTokenBan:
    def test_banned_ids_and_oov_tail_get_no_mass_and_the_rest_are_untouched(self, sampler_cls):
        install_sampler_token_ban([IMAGE_PAD, VIDEO_PAD], TOKENIZER_LEN)
        logits = torch.randn(4, LOGITS_WIDTH)
        logits[:, IMAGE_PAD] = 1e4  # the model's overwhelming preference is a banned id
        logits[:, LOGITS_WIDTH - 1] = 1e4
        original = logits.clone()

        out = _preprocess(sampler_cls, logits)

        banned = [IMAGE_PAD, VIDEO_PAD, *range(TOKENIZER_LEN, LOGITS_WIDTH)]
        assert _no_mass(out, banned)
        keep = torch.ones(LOGITS_WIDTH, dtype=torch.bool)
        keep[banned] = False
        torch.testing.assert_close(out[:, keep], original[:, keep], rtol=0, atol=0)
        assert not torch.isin(out.argmax(dim=-1), torch.tensor(banned)).any()

    def test_index_is_built_once_not_per_step(self, sampler_cls):
        install_sampler_token_ban([7, 9, 99], vocab_size=None)
        for _ in range(3):
            out = _preprocess(sampler_cls, torch.zeros(2, 32))
        cache = getattr(sampler_cls, sampler_token_ban._PATCH_ATTR)["index_cache"]
        assert len(cache) == 1
        (index,) = cache.values()
        assert index.tolist() == [7, 9]  # 99 has no column in a 32-wide logits row
        assert _no_mass(out, [7, 9])

    def test_reinstall_replaces_rather_than_stacks(self, sampler_cls):
        install_sampler_token_ban([7])
        install_sampler_token_ban([5])
        out = _preprocess(sampler_cls, torch.zeros(1, 16))
        assert _no_mass(out, [5]) and not _no_mass(out, [7])

    def test_missing_hook_fails_loudly(self, sampler_cls, monkeypatch):
        monkeypatch.delattr(sampler_cls, "_preprocess_logits")
        with pytest.raises(RuntimeError, match="_preprocess_logits not found"):
            install_sampler_token_ban([7])


class TestSchedulerEntry:
    def test_installs_the_ban_then_runs_sglang_and_survives_spawn_pickling(self, monkeypatch):
        import sglang.srt.entrypoints.engine as engine

        calls = []
        monkeypatch.setattr(sampler_token_ban, "install_sampler_token_ban", lambda *a: calls.append(("ban", a)))
        monkeypatch.setattr(engine, "run_scheduler_process", lambda *a, **k: calls.append(("run", a, k)))
        entry = functools.partial(
            run_scheduler_process_with_token_ban, verl_banned_token_ids=[IMAGE_PAD], verl_vocab_size=TOKENIZER_LEN
        )
        pickle.dumps(entry)  # SGLang starts the scheduler with multiprocessing spawn

        entry("server_args", "port_args", 0)

        assert calls == [("ban", ([IMAGE_PAD], TOKENIZER_LEN)), ("run", ("server_args", "port_args", 0), {})]


class _StopLaunch(Exception):
    pass


def _server(mod, **config):
    server = object.__new__(mod.SGLangHttpServer)
    server.config = SimpleNamespace(
        max_model_len=64,
        response_length=8,
        prompt_length=32,
        enable_rollout_routing_replay=False,
        skip_tokenizer_init=True,
        mtp=None,
        **config,
    )
    server.model_config = SimpleNamespace(lora_rank=0, lora={}, tokenizer=[0] * TOKENIZER_LEN)
    server._banned_token_ids = [IMAGE_PAD, VIDEO_PAD]
    server._disaggregation_role = "null"
    server._pd_decode_peers = []
    server.global_steps = 1
    return server


class TestSGLangServerWiring:
    def test_generate_request_carries_no_processor_and_keeps_prompt_placeholders(self, monkeypatch):
        mod = pytest.importorskip("verl.workers.rollout.sglang_rollout.async_sglang_server")
        monkeypatch.setattr(mod.ray, "get_runtime_context", lambda: SimpleNamespace(get_actor_name=lambda: "t"))
        captured = {}

        class _TokenizerManager:
            def generate_request(self, request, _):
                captured["request"] = request

                async def _out():
                    yield {"output_ids": [1], "meta_info": {"finish_reason": {"type": "stop"}}}

                return _out()

        server = _server(mod)
        server.tokenizer_manager = _TokenizerManager()
        prompt_ids = [10, IMAGE_PAD, 11]

        asyncio.run(server.generate(prompt_ids=prompt_ids, sampling_params={"max_tokens": 1}, request_id="r0"))

        request = captured["request"]
        assert not request.custom_logit_processor
        assert "custom_params" not in request.sampling_params
        assert request.input_ids == prompt_ids

    def test_launch_installs_the_ban_in_the_scheduler_without_the_processor_flag(self, monkeypatch):
        mod = pytest.importorskip("verl.workers.rollout.sglang_rollout.async_sglang_server")
        captured = {}

        def _capture_launch(**kwargs):
            captured.update(kwargs)
            raise _StopLaunch

        monkeypatch.setattr(mod, "ServerArgs", lambda **kwargs: captured.setdefault("server_args", kwargs))
        monkeypatch.setattr(mod.Engine, "_launch_subprocesses", staticmethod(_capture_launch))
        monkeypatch.setattr(mod.sglang.srt.entrypoints.engine, "_set_envs_and_config", None, raising=False)
        monkeypatch.setenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN", "0")

        server = _server(
            mod,
            engine_kwargs={"sglang": {}},
            checkpoint_engine={},
            quantization=None,
            tensor_model_parallel_size=1,
            data_parallel_size=1,
            expert_parallel_size=1,
            gpu_memory_utilization=0.5,
            enforce_eager=False,
            dtype="bfloat16",
            load_format="dummy",
            max_num_seqs=None,
            prometheus=SimpleNamespace(enable=False, served_model_name=None),
        )
        server.config.get = lambda key, default=None: getattr(server.config, key, default)
        server.model_config.local_path = "/nonexistent"
        server.model_config.trust_remote_code = False
        server.nnodes, server.node_rank, server.base_gpu_id = 1, 0, 0
        server.rollout_mode = mod.RolloutMode.HYBRID

        with pytest.raises(_StopLaunch):
            asyncio.run(server.launch_server())

        assert "enable_custom_logit_processor" not in captured["server_args"]
        entry = captured["run_scheduler_process_func"]
        assert entry.func is run_scheduler_process_with_token_ban
        assert entry.keywords == {"verl_banned_token_ids": [IMAGE_PAD, VIDEO_PAD], "verl_vocab_size": TOKENIZER_LEN}
        pickle.dumps(entry)
