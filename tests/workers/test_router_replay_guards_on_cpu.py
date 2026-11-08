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

"""CPU tests for the router replay startup guards.

The guards turn unsupported router replay setups into explicit failures
instead of silent route corruption:

* ``_validate_router_replay_setup`` rejects ``moe_permute_fusion`` /
  ``moe_router_fusion`` / ``moe_router_topk == 1`` and warns for
  dispatchers other than ``alltoall``.
* ``apply_router_replay_patch`` raises a clear error when Megatron-Core's
  MoE helpers failed to import, instead of leaving helper symbols unbound.
"""

from types import SimpleNamespace

import pytest

from verl.utils.megatron import router_replay_patch as replay_patch
from verl.workers.engine.megatron import transformer_impl


def _configs(mode="R3", **overrides):
    tf_values = {
        "moe_permute_fusion": False,
        "moe_router_fusion": False,
        "moe_token_dispatcher_type": "alltoall",
        "moe_router_topk": 8,
        "num_layers": 4,
    }
    tf_values.update(overrides)
    engine_config = SimpleNamespace(router_replay=SimpleNamespace(mode=mode))
    return engine_config, SimpleNamespace(**tf_values)


def test_disabled_mode_skips_all_checks():
    engine_config, tf_config = _configs(
        mode="disabled", moe_permute_fusion=True, moe_router_fusion=True, moe_router_topk=1
    )

    transformer_impl._validate_router_replay_setup(engine_config, tf_config)


@pytest.mark.parametrize(
    "override",
    [
        {"moe_permute_fusion": True},
        {"moe_router_fusion": True},
        {"moe_router_topk": 1},
    ],
)
def test_rejects_unsupported_setup(override):
    engine_config, tf_config = _configs(**override)

    with pytest.raises(ValueError, match="router_replay"):
        transformer_impl._validate_router_replay_setup(engine_config, tf_config)


def test_warns_for_non_alltoall_dispatcher(monkeypatch):
    engine_config, tf_config = _configs(moe_token_dispatcher_type="flex")
    messages = []

    monkeypatch.setattr(transformer_impl.logger, "warning", lambda msg, *args: messages.append(msg % args))

    transformer_impl._validate_router_replay_setup(engine_config, tf_config)

    assert any("alltoall" in message for message in messages)


def test_patch_rejects_degraded_import(monkeypatch):
    monkeypatch.setattr(replay_patch, "_REPLAY_IMPORT_ERROR", ImportError("missing moe_utils"))

    with pytest.raises(RuntimeError, match="MoE helpers are unavailable"):
        replay_patch.apply_router_replay_patch()
