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

"""Unit tests for LoRA merge-vs-adapter detection in the SGLang rollout.

``lora_served_as_adapter`` drives every LoRA branch of ``SGLangHttpServer``:
``enable_lora`` in ``launch_server``, ``lora_path`` on each ``generate`` request, and the
release tags in ``sleep``. With ``model.lora.merge=True`` the trainer merges the adapter
into the base weights and pushes a full weight update (``peft_config=None``), so SGLang
must stay LoRA-free -- otherwise requests reference an adapter that is never loaded.

``lora_base_kept_on_sleep`` narrows the ``sleep`` case: with ``model.lora.resync_base=True``
the trainer re-syncs the base on every update, so sleep releases the weights in adapter mode
too.

Note the two config blocks of ``HFModelConfig``, which are never synced: megatron runs set
``model.lora.rank``, fsdp runs set the flat ``model.lora_rank``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verl.workers.rollout.sglang_rollout.utils import lora_base_kept_on_sleep, lora_served_as_adapter


@dataclass
class _StubModelConfig:
    """Minimal stand-in exposing the HFModelConfig fields the helper reads."""

    lora_rank: int = 0
    lora: dict[str, Any] = field(default_factory=dict)


class TestLoraServedAsAdapter:
    def test_no_lora(self):
        assert lora_served_as_adapter(_StubModelConfig()) is False

    def test_megatron_merge(self):
        """megatron + merge: keep SGLang LoRA-free even though lora.rank > 0."""
        assert lora_served_as_adapter(_StubModelConfig(lora={"rank": 16, "merge": True})) is False

    def test_megatron_adapter(self):
        assert lora_served_as_adapter(_StubModelConfig(lora={"rank": 16})) is True

    def test_fsdp_merge(self):
        assert lora_served_as_adapter(_StubModelConfig(lora_rank=8, lora={"merge": True})) is False

    def test_fsdp_adapter(self):
        assert lora_served_as_adapter(_StubModelConfig(lora_rank=8)) is True

    def test_merge_absent_defaults_to_adapter(self):
        assert lora_served_as_adapter(_StubModelConfig(lora_rank=8, lora={})) is True

    def test_merge_without_lora(self):
        """merge=True on a run without LoRA is still 'no adapter'."""
        assert lora_served_as_adapter(_StubModelConfig(lora={"rank": 0, "merge": True})) is False

    def test_resync_base_still_serves_adapter(self):
        """resync_base only changes what sleep releases; SGLang still serves the adapter."""
        assert lora_served_as_adapter(_StubModelConfig(lora={"rank": 16, "resync_base": True})) is True


class TestSleepTags:
    """``SGLangHttpServer.sleep`` keeps the weights only when they are the base of an adapter
    that gets hot-swapped in place and the base is not re-synced on every update."""

    @staticmethod
    def _sleep_tags(model_config) -> list[str]:
        return ["kv_cache"] if lora_base_kept_on_sleep(model_config) else ["kv_cache", "weights"]

    def test_merge_mode_releases_weights(self):
        assert self._sleep_tags(_StubModelConfig(lora={"rank": 16, "merge": True})) == ["kv_cache", "weights"]

    def test_adapter_mode_keeps_weights(self):
        assert self._sleep_tags(_StubModelConfig(lora={"rank": 16})) == ["kv_cache"]

    def test_adapter_mode_resync_base_false_keeps_weights(self):
        assert self._sleep_tags(_StubModelConfig(lora={"rank": 16, "resync_base": False})) == ["kv_cache"]

    def test_megatron_adapter_mode_resync_base_releases_weights(self):
        model_config = _StubModelConfig(lora={"rank": 16, "resync_base": True})
        assert self._sleep_tags(model_config) == ["kv_cache", "weights"]

    def test_fsdp_adapter_mode_resync_base_releases_weights(self):
        model_config = _StubModelConfig(lora_rank=8, lora={"resync_base": True})
        assert self._sleep_tags(model_config) == ["kv_cache", "weights"]

    def test_resync_base_without_lora_releases_weights(self):
        assert self._sleep_tags(_StubModelConfig(lora={"rank": 0, "resync_base": True})) == ["kv_cache", "weights"]
