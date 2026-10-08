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
"""Unit tests for the platform-plugin attention dispatch in verl.utils.attention_utils."""

import sys
import types
from unittest import mock

import pytest

import verl.plugin.platform.platform_manager as pm
from verl.plugin.platform import set_platform
from verl.utils import attention_utils

FAKE_MODULE_NAME = "tests.utils._fake_plugin_attention_utils"


def _install_fake_attention_module():
    """Register a fake dotted module exposing the four flash-attn-equivalent functions."""
    mod = types.ModuleType(FAKE_MODULE_NAME)
    mod.index_first_axis = mock.Mock(name="index_first_axis", return_value="plugin_index_first_axis")
    mod.pad_input = mock.Mock(name="pad_input", return_value="plugin_pad_input")
    mod.rearrange = mock.Mock(name="rearrange", return_value="plugin_rearrange")
    mod.unpad_input = mock.Mock(name="unpad_input", return_value="plugin_unpad_input")
    sys.modules[FAKE_MODULE_NAME] = mod
    return mod


@pytest.fixture
def reset_platform():
    pm._current_platform = None
    yield
    pm._current_platform = None
    sys.modules.pop(FAKE_MODULE_NAME, None)


def test_plugin_module_dispatch(reset_platform):
    """A platform that returns a module path should have all four functions dispatched through it."""
    fake_mod = _install_fake_attention_module()

    platform = mock.Mock()
    platform.attention_utils_module.return_value = FAKE_MODULE_NAME
    set_platform(platform)

    with mock.patch("verl.utils.device.is_torch_npu_available", return_value=False):
        assert attention_utils.index_first_axis() == "plugin_index_first_axis"
        assert attention_utils.pad_input() == "plugin_pad_input"
        assert attention_utils.rearrange() == "plugin_rearrange"
        assert attention_utils.unpad_input() == "plugin_unpad_input"

    fake_mod.index_first_axis.assert_called_once()
    fake_mod.pad_input.assert_called_once()
    fake_mod.rearrange.assert_called_once()
    fake_mod.unpad_input.assert_called_once()


def test_npu_takes_precedence_over_plugin_module(reset_platform):
    """The NPU path must win even when the current platform also offers a plugin module."""
    _install_fake_attention_module()

    platform = mock.Mock()
    platform.attention_utils_module.return_value = FAKE_MODULE_NAME
    set_platform(platform)

    with mock.patch("verl.utils.device.is_torch_npu_available", return_value=True):
        func, *_ = attention_utils._get_attention_functions()

    from verl.utils.npu_flash_attn_utils import index_first_axis as npu_index_first_axis

    assert func is npu_index_first_axis


def test_no_plugin_module_falls_back(reset_platform):
    """A platform that returns None must not be treated as providing a plugin module."""
    platform = mock.Mock()
    platform.attention_utils_module.return_value = None
    set_platform(platform)

    with mock.patch("verl.utils.device.is_torch_npu_available", return_value=False):
        index_first_axis, pad_input, rearrange, unpad_input = attention_utils._get_attention_functions()

    assert index_first_axis is not None
    assert pad_input is not None
    assert rearrange is not None
    assert unpad_input is not None
