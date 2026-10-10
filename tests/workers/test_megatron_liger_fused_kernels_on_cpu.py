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

import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.models.mcore import model_forward_fused as mff
from verl.utils.kernel import linear_cross_entropy
from verl.workers import engine_workers
from verl.workers.config import McoreEngineConfig
from verl.workers.engine.megatron import transformer_impl
from verl.workers.engine_workers import _plan_liger_flsce_capacity


@pytest.fixture
def engine_config():
    return SimpleNamespace(
        use_fused_kernels=True,
        use_remove_padding=True,
        max_token_len_per_gpu=4096,
        infer_max_token_len_per_gpu=2048,
        context_parallel_size=2,
    )


def _model(post_process, tied=False, dtype=torch.bfloat16):
    # All PP/VPP chunks retain GPTModel's global dimensions. Do not infer
    # configuration from an output weight that only the final stage owns.
    return SimpleNamespace(
        post_process=post_process,
        share_embeddings_and_output_weights=tied,
        vocab_size=124160,
        config=SimpleNamespace(hidden_size=5120, params_dtype=dtype),
        parameters=lambda: iter([torch.empty(1)]),
    )


@pytest.mark.parametrize("post_process", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_configure_every_pipeline_stage_from_model_dimensions(monkeypatch, engine_config, post_process, tied):
    model = _model(post_process, tied)
    process_group = object()
    calls = []
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: process_group)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(linear_cross_entropy, "configure_liger_flsce", lambda **kwargs: calls.append(kwargs) or True)

    assert mff._configure_liger_runtime(model, engine_config) is True
    assert calls == [
        {
            "max_tokens": 8192,
            "hidden_size": 5120,
            "local_vocab_size": 62080,
            "process_group": process_group,
            "device": torch.device("cpu"),
        }
    ]


def test_configuration_requires_token_capacity(engine_config):
    engine_config.max_token_len_per_gpu = None
    engine_config.infer_max_token_len_per_gpu = None
    with pytest.raises(RuntimeError, match="max-token limit"):
        mff._configure_liger_runtime(_model(False), engine_config)


@pytest.mark.parametrize("use_liger", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("post_process", [False, True])
def test_patch_owns_backend_selection_and_runtime_setup(monkeypatch, engine_config, use_liger, dtype, post_process):
    model = _model(post_process, dtype=dtype)
    calls = []
    monkeypatch.setattr(mff, "_get_patching_model", lambda value: value)
    monkeypatch.setattr(mff, "_resolve_fused_forward_mode", lambda value: mff._HOOK_MODE)
    monkeypatch.setattr(mff, "_configure_liger_runtime", lambda model, config: calls.append((model, config)) or True)
    mff.patch_fused_forward(model, SimpleNamespace(use_liger=use_liger), engine_config=engine_config)
    assert getattr(model, mff._FUSED_IMPL_BACKEND_ATTR) == ("liger" if use_liger else "triton")
    assert calls == ([(model, engine_config)] if use_liger and dtype == torch.bfloat16 else [])


def test_engine_delegates_all_virtual_chunks(monkeypatch, engine_config):
    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.engine_config = engine_config
    engine.model_config = SimpleNamespace(use_liger=True, mtp=SimpleNamespace(enable=False))
    engine.is_value_model = False
    engine.module = [_model(False), _model(True)]
    calls = []
    monkeypatch.setattr(
        mff, "patch_fused_forward", lambda model, config, **kwargs: calls.append((model, config, kwargs))
    )
    engine._maybe_enable_fused_kernels()
    assert calls == [(model, engine.model_config, {"engine_config": engine_config}) for model in engine.module]


def _colocated_config():
    return OmegaConf.create(
        {
            "model": {"use_liger": True, "use_fused_kernels": True},
            "actor": {
                "strategy": "megatron",
                "ppo_max_token_len_per_gpu": 8192,
                "megatron": {"context_parallel_size": 2, "tensor_model_parallel_size": 2},
            },
            "ref": {
                "strategy": "megatron",
                "log_prob_max_token_len_per_gpu": 2048,
                "megatron": {"context_parallel_size": 1, "tensor_model_parallel_size": 4},
            },
            "rollout": {"log_prob_max_token_len_per_gpu": 4096},
        }
    )


def test_ref_first_reserves_larger_actor_and_smaller_tp(monkeypatch):
    config = _colocated_config()
    reservation = _plan_liger_flsce_capacity(config, "actor_rollout_ref")
    assert reservation == (16384, 2)
    calls = []
    monkeypatch.setattr(linear_cross_entropy, "configure_liger_flsce", lambda **kwargs: calls.append(kwargs) or True)
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: object())
    for tp_size, cp_size, tokens in ((4, 1, 2048), (2, 2, 8192)):
        engine = McoreEngineConfig(tensor_model_parallel_size=tp_size, context_parallel_size=cp_size)
        engine._liger_flsce_capacity = reservation
        engine.infer_max_token_len_per_gpu = tokens
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group, size=tp_size: size)
        assert mff._configure_liger_runtime(_model(True), engine)
    assert [call["max_tokens"] for call in calls] == [16384, 16384]
    assert [call["local_vocab_size"] for call in calls] == [62080, 62080]


@pytest.mark.parametrize("flag", ["use_liger", "use_fused_kernels"])
def test_capacity_planning_requires_both_flags(flag):
    config = _colocated_config()
    config.model[flag] = False
    assert _plan_liger_flsce_capacity(config, "actor_rollout_ref") is None


def test_capacity_planning_is_role_and_backend_specific():
    config = _colocated_config()
    assert _plan_liger_flsce_capacity(config, "ref") == (2048, 4)
    assert _plan_liger_flsce_capacity(config, "actor_rollout") == (16384, 2)
    assert _plan_liger_flsce_capacity(config, "rollout") is None
    config.actor.strategy = config.ref.strategy = "fsdp"
    assert _plan_liger_flsce_capacity(config, "actor_rollout_ref") is None


def test_worker_reserves_both_consumers_before_ref_reset(monkeypatch):
    config = _colocated_config()
    config.actor.ppo_mini_batch_size = 8
    config.actor.ppo_micro_batch_size_per_gpu = 1
    config.actor.use_dynamic_bsz = False
    config.rollout.log_prob_use_dynamic_bsz = False
    config.rollout.log_prob_micro_batch_size_per_gpu = 1
    config.ref.log_prob_micro_batch_size_per_gpu = 1
    model = SimpleNamespace(get=lambda key, default=None: config.model.get(key, default))
    ref = SimpleNamespace(engine=McoreEngineConfig(tensor_model_parallel_size=4), optim=None, checkpoint=None)
    actor = SimpleNamespace(engine=McoreEngineConfig(tensor_model_parallel_size=2), optim=None, checkpoint=None)

    def convert(value):
        if value is config.model:
            return model
        return ref if value is config.ref else actor

    monkeypatch.setattr(engine_workers, "omega_conf_to_dataclass", convert)
    monkeypatch.setattr(engine_workers, "TrainingWorkerConfig", SimpleNamespace)
    calls = []

    class ActorReached(Exception):
        pass

    class Reference:
        def __init__(self, config):
            self.config = config

        def reset(self):
            calls.append(("ref", self.config.engine_config._liger_flsce_capacity))
            assert self.config.engine_config.infer_max_token_len_per_gpu == 2048

        def get_dispatch_collect(self):
            return {}

    class Actor(Reference):
        def reset(self):
            calls.append(("actor", self.config.engine_config._liger_flsce_capacity))
            assert self.config.engine_config.max_token_len_per_gpu == 8192
            assert self.config.engine_config.infer_max_token_len_per_gpu == 4096
            raise ActorReached

    worker = SimpleNamespace(
        config=config,
        role="actor_rollout_ref",
        ref_worker_cls=Reference,
        actor_worker_cls=Actor,
        set_dispatch_collect=lambda **kwargs: None,
        _omega_profiler_config={},
        distillation_enabled=False,
    )
    with pytest.raises(ActorReached):
        engine_workers.ActorRolloutRefWorker.init_model(worker)
    assert calls == [("ref", (16384, 2)), ("actor", (16384, 2))]


def test_optional_liger_fallback_does_not_restrict_deepep(monkeypatch, engine_config):
    monkeypatch.setattr(linear_cross_entropy, "configure_liger_flsce", lambda **kwargs: False)
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: object())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(mff, "_validate_liger_moe_runtime", lambda model: pytest.fail("No native runtime to protect"))
    assert mff._configure_liger_runtime(_model(True), engine_config) is False


@pytest.mark.parametrize("rdma_bytes", [0, 4096])
@pytest.mark.parametrize("legacy_flag", [False, True])
def test_liger_rejects_only_nvshmem_deepep_buffers(monkeypatch, rdma_bytes, legacy_flag):
    model = _model(True)
    model.config.num_moe_experts = 8
    model.config.moe_enable_deepep = legacy_flag
    model.config.moe_token_dispatcher_type = "alltoall" if legacy_flag else "flex"
    model.config.moe_flex_dispatcher_backend = "deepep"
    seen = []

    def hint(hidden_bytes, size):
        seen.append((hidden_bytes, size))
        return rdma_bytes

    options = SimpleNamespace(get_rdma_buffer_size_hint=hint)
    buffer = SimpleNamespace(get_dispatch_config=lambda size: options, get_combine_config=lambda size: options)
    monkeypatch.setitem(sys.modules, "deep_ep", SimpleNamespace(Buffer=buffer))
    monkeypatch.setattr(mff.parallel_state, "get_expert_tensor_and_model_parallel_group", lambda: object())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 16)
    if rdma_bytes:
        with pytest.raises(RuntimeError, match="DeepEP V1 RDMA"):
            mff._validate_liger_moe_runtime(model)
    else:
        mff._validate_liger_moe_runtime(model)
    assert seen and all(value == (10240, 16) for value in seen)


@pytest.mark.parametrize("backend", ["alltoall", "hybridep"])
def test_other_moe_dispatchers_do_not_load_deepep_v1(monkeypatch, backend):
    model = _model(True)
    model.config.num_moe_experts = 8
    model.config.moe_token_dispatcher_type = "alltoall" if backend == "alltoall" else "flex"
    model.config.moe_flex_dispatcher_backend = backend
    monkeypatch.setitem(sys.modules, "deep_ep", None)
    mff._validate_liger_moe_runtime(model)
