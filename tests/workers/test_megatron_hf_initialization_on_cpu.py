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

from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from verl.workers.config import McoreEngineConfig, McoreOptimizerConfig
from verl.workers.engine.megatron import transformer_impl


@pytest.mark.parametrize("is_value_model", [False, True], ids=["actor", "critic"])
@pytest.mark.parametrize(
    ("native_format", "initial_path", "use_native_loader"),
    [(False, None, False), (False, "/native/model", False), (True, None, False), (True, "/native/model", True)],
)
def test_initial_weights_loader_is_independent_of_checkpoint_save_format(
    native_format, initial_path, use_native_loader, is_value_model, monkeypatch
):
    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.engine_config = McoreEngineConfig(use_dist_checkpointing=native_format, dist_checkpointing_path=initial_path)
    engine.optimizer_config = McoreOptimizerConfig()
    engine.model_config = OmegaConf.create({"hf_config": {}, "local_path": "/models/hf"})
    engine.is_value_model = is_value_model
    engine.bridge = Mock()
    engine.provider = None
    engine.peft_cls = None
    engine.enable_routing_replay = False

    modules = [object()]
    native_loader = Mock()
    monkeypatch.setattr("verl.utils.megatron_utils.make_megatron_module", Mock(return_value=(modules, object())))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(transformer_impl, "load_mcore_dist_weights", native_loader)

    assert engine._build_megatron_module() is modules

    if use_native_loader:
        native_loader.assert_called_once_with(modules, initial_path, is_value_model=is_value_model)
        engine.bridge.load_hf_weights.assert_not_called()
    else:
        native_loader.assert_not_called()
        engine.bridge.load_hf_weights.assert_called_once_with(
            modules,
            "/models/hf",
            allowed_mismatched_params=["output_layer.weight"] if is_value_model else [],
        )
    assert engine.engine_config.use_dist_checkpointing is native_format
