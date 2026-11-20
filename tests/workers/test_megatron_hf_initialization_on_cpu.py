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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from verl.trainer.config import CheckpointConfig
from verl.workers.config import McoreEngineConfig, McoreOptimizerConfig
from verl.workers.engine.megatron import transformer_impl


class _ModelConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


@pytest.mark.parametrize("is_value_model", [False, True], ids=["actor", "critic"])
@pytest.mark.parametrize(
    ("native_format", "initial_path"),
    [(False, None), (False, "/native/model"), (True, None), (True, "/native/model"), (True, "")],
)
def test_initial_weights_loader_is_independent_of_checkpoint_save_format(
    native_format, initial_path, is_value_model, monkeypatch
):
    engine_cls = (
        transformer_impl.MegatronEngineWithValueHead if is_value_model else transformer_impl.MegatronEngineWithLMHead
    )
    engine = object.__new__(engine_cls)
    engine.engine_config = McoreEngineConfig(use_dist_checkpointing=native_format, dist_checkpointing_path=initial_path)
    original_config = engine.engine_config
    engine.optimizer_config = McoreOptimizerConfig()
    engine.checkpoint_config = CheckpointConfig()
    engine.model_config = _ModelConfig(
        hf_config=object(),
        local_path="/models/hf",
        architectures=["TestForCausalLM"],
        mtp=SimpleNamespace(enable=False),
        get_processor=lambda: None,
    )
    engine.is_value_model = is_value_model
    engine.bridge = SimpleNamespace(load_hf_weights=Mock())
    engine.provider = object()
    engine.peft_cls = None
    engine.enable_routing_replay = False
    engine._real_nvfp4_enabled = False
    engine._qat_enabled = False
    engine._is_offload_param = False
    engine._is_offload_optimizer = False
    engine.param_dtype = torch.float32
    engine.share_embeddings_and_output_weights = False
    engine.tf_config = object()
    engine._build_tf_config = Mock()
    engine._resolve_override_ddp_config = lambda: {}
    engine._maybe_enable_fused_kernels = Mock()
    engine._build_optimizer = Mock()
    engine._build_lr_scheduler = Mock()
    engine.to = Mock()
    modules = [object()]
    native_loader = Mock()
    manager = Mock()
    monkeypatch.setattr("verl.utils.megatron_utils.make_megatron_module", lambda **kwargs: (modules, engine.tf_config))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(transformer_impl, "load_mcore_dist_weights", native_loader)
    monkeypatch.setattr(transformer_impl, "MegatronCheckpointManager", manager)
    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)

    engine.initialize()

    if native_format and initial_path is not None:
        native_loader.assert_called_once_with(modules, initial_path, is_value_model=is_value_model)
        engine.bridge.load_hf_weights.assert_not_called()
    else:
        native_loader.assert_not_called()
        engine.bridge.load_hf_weights.assert_called_once_with(
            modules,
            "/models/hf",
            allowed_mismatched_params=["output_layer.weight"] if is_value_model else [],
        )
    assert engine.module is modules
    assert engine.engine_config is original_config
    assert engine.engine_config.use_dist_checkpointing is native_format
    assert engine.engine_config.dist_checkpointing_path == initial_path
    manager.assert_called_once()
    assert manager.call_args.kwargs["use_dist_checkpointing"] is native_format
    assert manager.call_args.kwargs["role"] == ("critic" if is_value_model else "actor")
    assert engine.checkpoint_mananager is manager.return_value
