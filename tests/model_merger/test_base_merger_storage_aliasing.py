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

"""Regression tests for the defensive duplicate-storage guard added for
https://github.com/verl-project/verl/issues/6259.

``BaseModelMerger.save_hf_model_and_tokenizer`` is the single choke point (shared by
both the FSDP and Megatron mergers) right before ``save_pretrained`` writes the merged
state_dict out as safetensors shards. If two *different* keys in that state_dict
happen to alias the same tensor storage, ``save_pretrained`` silently keeps only one of
them per shard -- exactly the "duplicate keys removed" corruption reported in the
issue. ``assert_no_aliased_state_dict_tensors`` turns that silent corruption into a
loud, actionable error naming the offending keys.

This test loads the real ``verl/model_merger/base_model_merger.py`` source directly via
``importlib``, stubbing out ``accelerate``/``transformers``/``verl.utils`` (not
installed, or version-incompatible, in every test environment) so the check itself is
exercised against production code without requiring those heavy optional dependencies.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_MERGER_DIR = REPO_ROOT / "verl" / "model_merger"


def _install_fake_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


@pytest.fixture()
def base_merger_module():
    """Import the real base_model_merger.py with accelerate/transformers/verl.utils stubbed."""

    if "verl" not in sys.modules:
        _install_fake_module("verl", __path__=[str(REPO_ROOT / "verl")])
    if "verl.model_merger" not in sys.modules:
        _install_fake_module("verl.model_merger", __path__=[str(MODEL_MERGER_DIR)])

    _install_fake_module("accelerate", init_empty_weights=lambda: _NullContext())

    class _FakeHFClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise NotImplementedError("not used by these tests")

        @classmethod
        def from_config(cls, *args, **kwargs):
            raise NotImplementedError("not used by these tests")

    _install_fake_module(
        "transformers",
        AutoConfig=_FakeHFClass,
        AutoModelForCausalLM=_FakeHFClass,
        AutoModelForTokenClassification=_FakeHFClass,
        GenerationConfig=_FakeHFClass,
    )

    _install_fake_module("verl.utils", hf_processor=lambda *a, **k: None, hf_tokenizer=lambda *a, **k: None)
    _install_fake_module(
        "verl.utils.transformers_compat",
        drop_tied_target_keys=lambda *a, **k: None,
        get_auto_model_for_vision2seq=lambda: _FakeHFClass,
    )

    spec = importlib.util.spec_from_file_location(
        "verl.model_merger.base_model_merger", MODEL_MERGER_DIR / "base_model_merger.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["verl.model_merger.base_model_merger"] = module
    spec.loader.exec_module(module)

    yield module

    for name in (
        "accelerate",
        "transformers",
        "verl.utils",
        "verl.utils.transformers_compat",
        "verl.model_merger.base_model_merger",
    ):
        sys.modules.pop(name, None)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_assert_no_aliased_state_dict_tensors_passes_for_independent_tensors(base_merger_module):
    state_dict = {
        "model.a.weight": torch.zeros(4),
        "model.b.weight": torch.ones(4),
    }
    # Should not raise.
    base_merger_module.assert_no_aliased_state_dict_tensors(state_dict)


def test_assert_no_aliased_state_dict_tensors_ignores_empty_tensors(base_merger_module):
    state_dict = {
        "model.a.weight": torch.empty(0),
        "model.b.weight": torch.empty(0),
    }
    # Zero-element tensors can spuriously share a null data pointer; must not false-positive.
    base_merger_module.assert_no_aliased_state_dict_tensors(state_dict)


def test_assert_no_aliased_state_dict_tensors_detects_collision(base_merger_module):
    shared_buffer = torch.arange(8, dtype=torch.float32)
    state_dict = {
        "model.small_a.weight": shared_buffer[0:4],
        "model.small_b.weight": shared_buffer[4:8],
        "model.independent.weight": torch.ones(4),
    }

    with pytest.raises(RuntimeError, match="6259") as exc_info:
        base_merger_module.assert_no_aliased_state_dict_tensors(state_dict)

    message = str(exc_info.value)
    assert "model.small_a.weight" in message
    assert "model.small_b.weight" in message
    assert "model.independent.weight" not in message
