# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import types
from importlib import util
from pathlib import Path


def _install_fake_megatron(monkeypatch, include_mla_config=True):
    class TransformerConfig:
        num_layers = None
        hidden_size = None
        num_attention_heads = None

        def __init__(self, num_layers, hidden_size, num_attention_heads):
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads

    class MLATransformerConfig(TransformerConfig):
        def __init__(self, num_layers, hidden_size, num_attention_heads):
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads

    class TopKRouter:
        def __init__(self, config=None):
            self.config = config

        def routing(self, *args, **kwargs):
            raise NotImplementedError

    class MoEAlltoAllTokenDispatcher:
        def preprocess(self, routing_map):
            return routing_map

    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    transformer = types.ModuleType("megatron.core.transformer")
    transformer_config = types.ModuleType("megatron.core.transformer.transformer_config")
    moe = types.ModuleType("megatron.core.transformer.moe")
    moe_utils = types.ModuleType("megatron.core.transformer.moe.moe_utils")
    router = types.ModuleType("megatron.core.transformer.moe.router")
    token_dispatcher = types.ModuleType("megatron.core.transformer.moe.token_dispatcher")

    transformer.TransformerConfig = TransformerConfig
    if include_mla_config:
        transformer.MLATransformerConfig = MLATransformerConfig
    transformer_config.TransformerConfig = TransformerConfig
    router.TopKRouter = TopKRouter
    token_dispatcher.MoEAlltoAllTokenDispatcher = MoEAlltoAllTokenDispatcher

    moe_utils.apply_router_token_dropping = lambda *args, **kwargs: None
    moe_utils.compute_routing_scores_for_aux_loss = lambda *args, **kwargs: None
    moe_utils.group_limited_topk = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer", transformer)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.transformer_config", transformer_config)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe", moe)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.moe_utils", moe_utils)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.router", router)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.token_dispatcher", token_dispatcher)

    monkeypatch.delitem(sys.modules, "verl.utils.megatron.router_replay_patch", raising=False)

    return TransformerConfig, MLATransformerConfig


def _construct_after_config_converter_filter(config_cls):
    config = {
        "num_layers": 1,
        "hidden_size": 128,
        "num_attention_heads": 1,
        "enable_routing_replay": True,
    }
    filtered_config = {key: value for key, value in config.items() if hasattr(config_cls, key)}
    return config_cls(**filtered_config)


def _load_router_replay_patch():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "verl/utils/megatron/router_replay_patch.py"
    spec = util.spec_from_file_location("router_replay_patch_under_test", module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_router_replay_patch_exposes_config_flag_for_converter_filter(monkeypatch):
    config_classes = _install_fake_megatron(monkeypatch)
    router_replay_patch = _load_router_replay_patch()

    router_replay_patch.apply_router_replay_patch()

    for config_cls in config_classes:
        config = _construct_after_config_converter_filter(config_cls)
        assert config.enable_routing_replay is True


def test_router_replay_patch_supports_megatron_without_mla_config(monkeypatch):
    transformer_config, _ = _install_fake_megatron(monkeypatch, include_mla_config=False)
    router_replay_patch = _load_router_replay_patch()

    router_replay_patch.apply_router_replay_patch()

    config = _construct_after_config_converter_filter(transformer_config)
    assert config.enable_routing_replay is True
