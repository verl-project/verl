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

"""Exercise the production weight-sync method without GPU or Ray dependencies."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest


@pytest.fixture
def update_weights():
    """Load the actual method body, omitting only its infrastructure decorators."""
    path = Path(__file__).resolve().parents[2] / "verl/workers/engine_workers.py"
    tree = ast.parse(path.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(
        node for node in worker.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "update_weights"
    )
    method.decorator_list = []
    namespace = {
        "asyncio": asyncio,
        "torch": MagicMock(),
        "get_torch_device": MagicMock(),
        "set_expandable_segments": MagicMock(),
        "log_gpu_memory_usage": MagicMock(),
        "logger": MagicMock(),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["update_weights"]


def _worker(*, layered_summon=True, resync_base=True, base_sync_done=True, peft_merge=False, peft_config="adapter"):
    config = MagicMock()
    config.rollout.checkpoint_engine.backend = "naive"
    config.rollout.free_cache_engine = False
    config.rollout.get.return_value = "vllm"
    engine = MagicMock(is_param_offload_enabled=False)
    engine.get_per_tensor_param.side_effect = lambda *, base_sync_done, **kwargs: (
        "adapter_params" if base_sync_done else "base_params",
        peft_config,
    )
    return SimpleNamespace(
        config=config,
        actor=SimpleNamespace(engine=engine),
        rollout=AsyncMock(),
        peft_merge=peft_merge,
        layered_summon=layered_summon,
        lora_resync_base=resync_base,
        base_sync_done=base_sync_done,
    )


@pytest.mark.parametrize("base_sync_done,resync_base", [(True, True), (False, True), (False, False)])
def test_layered_summon_rejects_base_export(update_weights, base_sync_done, resync_base):
    """Reject base exports before entering the unsupported engine collection path."""
    worker = _worker(base_sync_done=base_sync_done, resync_base=resync_base)

    with pytest.raises(ValueError, match=r"layered_summon.*base.*resync_base"):
        asyncio.run(update_weights(worker))

    worker.actor.engine.get_per_tensor_param.assert_called_once_with(layered_summon=True, base_sync_done=True)
    worker.rollout.update_weights.assert_not_awaited()
    assert worker.base_sync_done is base_sync_done


@pytest.mark.parametrize(
    "overrides,expected_base_sync",
    [
        ({"resync_base": False}, False),
        ({"layered_summon": False}, True),
        ({"layered_summon": False, "base_sync_done": False, "resync_base": False}, True),
        ({"peft_merge": True, "peft_config": None}, False),
        ({"peft_config": None}, False),
    ],
)
def test_supported_sync_modes(update_weights, overrides, expected_base_sync):
    """Preserve adapter-only, base resync, initial base sync, merged and non-LoRA modes."""
    worker = _worker(**overrides)
    asyncio.run(update_weights(worker, global_steps=7))

    expected_exports = [call(layered_summon=worker.layered_summon, base_sync_done=True)]
    expected_updates = []
    peft_config = overrides.get("peft_config", "adapter")
    if expected_base_sync:
        expected_exports.append(call(layered_summon=worker.layered_summon, base_sync_done=False))
        expected_updates.append(call("base_params", peft_config=peft_config, base_sync_done=False, global_steps=7))
    expected_updates.append(call("adapter_params", peft_config=peft_config, base_sync_done=True, global_steps=7))
    assert worker.actor.engine.get_per_tensor_param.call_args_list == expected_exports
    assert worker.rollout.update_weights.await_args_list == expected_updates
    assert worker.base_sync_done is True
