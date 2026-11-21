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
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("freeze", [False, True])
@pytest.mark.parametrize("load_format", ["dummy", "safetensors"])
def test_frozen_ple_config_allows_full_sync_initialization(freeze, load_format):
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "model_engine=veomni",
                f"actor_rollout_ref.actor.veomni.freeze_ple_embeddings={freeze}",
                f"actor_rollout_ref.rollout.load_format={load_format}",
            ],
        )
    assert config.actor_rollout_ref.actor.veomni.freeze_ple_embeddings is freeze
    assert config.actor_rollout_ref.rollout.load_format == load_format
    assert "freeze_ple_embeddings" not in config.actor_rollout_ref.rollout


@pytest.mark.parametrize(
    "is_npu,model_type,vllm_version,lora,mtp,expected",
    [
        (True, "qwen4_exp", "0.32.0", False, False, 1),
        (False, "qwen4_exp", "0.32.0", False, False, 2),
        (True, "qwen3", "0.32.0", False, False, 2),
        (True, "qwen3", "0.30.0", False, False, 2),
        (True, "qwen3", "0.29.0", False, False, 1),
        (False, "qwen3", "0.29.0", False, False, 2),
        (False, "qwen3", "0.32.0", True, False, 1),
        (True, "qwen3", "0.32.0", False, True, 1),
    ],
)
def test_sleep_level_preserves_other_model_and_backend_behavior(
    monkeypatch, is_npu, model_type, vllm_version, lora, mtp, expected
):
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout import vllm_async_server

    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: is_npu)
    monkeypatch.setattr(vllm_async_server, "_VLLM_VERSION", vllm_async_server.version.parse(vllm_version))
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(mtp=SimpleNamespace(enable=mtp, enable_rollout=mtp))
    server.model_config = SimpleNamespace(
        lora_rank=8 if lora else 0, lora={}, hf_config=SimpleNamespace(model_type=model_type)
    )
    assert server._resolve_sleep_level() == expected


@pytest.mark.parametrize("sleep_method", ["_sleep_hybrid", "release_kv_cache"])
@pytest.mark.parametrize("load_format", ["dummy", "safetensors"])
@pytest.mark.parametrize("freeze", [False, True])
def test_ple_full_sync_preserves_runner_state_with_level1(monkeypatch, sleep_method, load_format, freeze, tmp_path):
    pytest.importorskip("vllm")
    pytest.importorskip("veomni")
    import torch

    from verl.utils.veomni.frozen_ple import freeze_ple_tables
    from verl.workers.config import RolloutConfig
    from verl.workers.engine.veomni import transformer_impl as veomni
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout import vllm_async_server

    # Modern Ascend otherwise selects level 2; this model needs level 1 regardless of freezing.
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: True)
    monkeypatch.setattr(vllm_async_server, "_VLLM_VERSION", vllm_async_server.version.parse("0.32.0"))

    class Engine:
        def __init__(self):
            self.table = torch.zeros(2, 3)
            self.offsets = torch.tensor([-2, -1])
            self.sleep_levels = []
            self.backup = None

        async def sleep(self, level):
            self.sleep_levels.append(level)
            self.backup = (self.table.clone(), self.offsets.clone()) if level == 1 else None
            self.table.zero_()
            self.offsets.zero_()

        async def wake_up(self, tags):
            if self.backup is not None:
                self.table.copy_(self.backup[0])
                self.offsets.copy_(self.backup[1])
                self.backup = None

        async def reset_encoder_cache(self):
            pass

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = RolloutConfig(name="vllm", load_format=load_format)
    server.model_config = SimpleNamespace(lora_rank=0, lora={}, hf_config=SimpleNamespace(model_type="qwen4_exp"))
    server.rollout_mode = RolloutMode.HYBRID
    server.node_rank = 0
    server.engine = Engine()
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen4_exp")
    model.ple = torch.nn.Module()
    model.ple.ple_embedding = torch.nn.Module()
    embedding = model.ple.ple_embedding
    embedding.rows_per_checkpoint_shard = 2
    embedding.padded_rows_per_shard = 4
    embedding.split_ngram_parts = 1
    embedding.ngram_embedding = torch.nn.ModuleDict({"shard_0": torch.nn.Embedding(4, 3)})
    if freeze:
        freeze_ple_tables(model)
    monkeypatch.setattr(veomni, "get_checkpoint_tensor_converter", lambda model: None)
    monkeypatch.setattr(veomni, "convert_weight_keys", lambda params, model: params)
    monkeypatch.setattr(veomni, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(veomni.parallel_state, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False))
    actor = object.__new__(veomni.VeOmniEngineWithLMHead)
    actor.module = model
    actor.engine_config = SimpleNamespace(freeze_ple_embeddings=freeze)
    actor._is_offload_param = False
    actor._uses_fsdp2_cpu_offload_policy = True
    actor._rollout_weight_quantizer = None
    name = "ple.ple_embedding.ngram_embedding.shard_0.weight"
    expected = model.state_dict()[name][:2].clone()
    model.projection = torch.nn.Linear(3, 3, bias=False)
    projection = model.projection.weight.detach().clone()
    if load_format == "safetensors":
        from safetensors.torch import load_file, save_file

        checkpoint = tmp_path / "model.safetensors"
        save_file({name: expected}, checkpoint)
        server.engine.table.copy_(load_file(checkpoint)[name])

    def sync_weights():
        # Use the real actor exporter, including checkpoint padding removal.
        weights = dict(actor.get_per_tensor_param()[0])
        assert name in weights  # Frozen parameters must still be exported on every sync.
        torch.testing.assert_close(weights["projection.weight"], model.projection.weight)
        projection.copy_(weights["projection.weight"])
        server.engine.table.copy_(weights[name])

    for _ in range(3):  # Includes the sleep/wake before the first actor sync.
        asyncio.run(getattr(server, sleep_method)())
        asyncio.run(server.engine.wake_up(tags=["weights"]))
        torch.testing.assert_close(server.engine.offsets, torch.tensor([-2, -1]))
        with torch.no_grad():
            model.projection.weight.add_(1)
        # A stale receiver table makes omission from later syncs observable too.
        server.engine.table.fill_(-100)
        sync_weights()
        torch.testing.assert_close(server.engine.table, expected)
    torch.testing.assert_close(projection, model.projection.weight)
    assert server.engine.sleep_levels == [1] * 3
