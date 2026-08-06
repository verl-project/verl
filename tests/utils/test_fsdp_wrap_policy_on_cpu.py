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

"""CPU-only tests for ``get_fsdp_wrap_policy``.

These tests cover the lenient resolution of ``_no_split_modules``: HF
``transformers`` sometimes ships forward-compat names alongside concrete
classes (e.g. Qwen3.5 lists ``Qwen3_5TextDecoderLayer`` next to the real
``Qwen3_5DecoderLayer``). The wrap policy should succeed as long as at least
one name resolves and only fail when none do.
"""

import pytest
import torch.nn as nn

from verl.utils.fsdp_utils import get_fsdp_wrap_policy


class _RealLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)


class _PartiallyResolvableModel(nn.Module):
    """Mirrors the Qwen3.5 case: one valid layer name + one unknown name."""

    _no_split_modules = ["_RealLayer", "_NonExistentLayer"]

    def __init__(self):
        super().__init__()
        self.layer = _RealLayer()


class _AllUnresolvableModel(nn.Module):
    _no_split_modules = ["_GhostLayerA", "_GhostLayerB"]

    def __init__(self):
        super().__init__()
        self.layer = _RealLayer()


def test_wrap_policy_skips_missing_layer_class_names(caplog):
    """A single missing name in ``_no_split_modules`` must not break wrapping."""
    import logging

    caplog.set_level(logging.WARNING, logger="verl.utils.fsdp_utils")

    policy = get_fsdp_wrap_policy(_PartiallyResolvableModel())
    assert policy is not None, "wrap policy should be built from the resolvable name"

    # The user should be told which names were skipped.
    assert any("_NonExistentLayer" in record.message for record in caplog.records), (
        "Expected a warning listing the skipped layer class names"
    )


def test_wrap_policy_raises_when_no_layer_classes_resolve():
    """If no name resolves, raise -- otherwise FSDP would silently no-op."""
    with pytest.raises(Exception, match="Could not find any of the transformer layer classes"):
        get_fsdp_wrap_policy(_AllUnresolvableModel())


def test_wrap_policy_explicit_config_overrides_no_split_modules():
    """Explicit ``transformer_layer_cls_to_wrap`` config takes precedence."""
    model = _PartiallyResolvableModel()
    config = {"transformer_layer_cls_to_wrap": ["_RealLayer"]}
    policy = get_fsdp_wrap_policy(model, config=config)
    assert policy is not None
