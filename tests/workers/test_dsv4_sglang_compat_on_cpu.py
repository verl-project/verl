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

import base64
import sys
import types

import numpy as np


def test_sglang_deepseek_v4_config_aliases_and_preserves_metadata(monkeypatch):
    from verl.workers.config.model import _get_sglang_deepseek_v4_config_class

    class FakeDeepSeekV4Config:
        def __init__(self, v_head_dim=None, window_size=None, n_hash_layers=None):
            self.v_head_dim = v_head_dim
            self.window_size = window_size
            self.n_hash_layers = n_hash_layers

    fake_module = types.ModuleType("sglang.srt.configs.deepseek_v4")
    fake_module.DeepSeekV4Config = FakeDeepSeekV4Config
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.deepseek_v4", fake_module)

    config_cls = _get_sglang_deepseek_v4_config_class()
    config = config_cls(
        head_dim=192,
        sliding_window=4096,
        num_hash_layers=3,
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        future_checkpoint_field="kept",
    )

    assert config.v_head_dim == 192
    assert config.window_size == 4096
    assert config.n_hash_layers == 3
    assert config.future_checkpoint_field == "kept"


def test_sglang_0512_base64_routed_experts_payload():
    from verl.workers.rollout.sglang_rollout.async_sglang_server import _decode_routed_experts_payload

    expected = np.arange(12, dtype=np.int32)
    encoded = base64.b64encode(expected.tobytes()).decode()
    actual = _decode_routed_experts_payload(encoded)

    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, expected)
