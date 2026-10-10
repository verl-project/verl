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

from typing import Callable

_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Dynamically import attention functions based on available hardware."""

    from verl.utils.device import is_torch_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    # flash-attn only ships CUDA wheels, so every other backend -- NPU, and any
    # device without a flash-attn build -- as well as CPU-only installs (unit
    # tests, dev boxes) use the pure-torch port in attention_padding_utils.
    use_flash_attn = not is_torch_npu_available(check_device=False)
    if use_flash_attn:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:
            use_flash_attn = False

    if not use_flash_attn:
        from verl.utils.attention_padding_utils import index_first_axis, pad_input, rearrange, unpad_input

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args, **kwargs):
    """
    Unified entry point for `index_first_axis` across backends.

    Dynamically dispatches to the appropriate implementation:
      - With flash-attn installed: `flash_attn.bert_padding.index_first_axis`
      - Otherwise: `verl.utils.attention_padding_utils.index_first_axis`

    Users can call this function directly without worrying about the underlying device.
    """
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    """
    Unified entry point for `pad_input` across backends.

    Dynamically dispatches to the appropriate implementation:
      - With flash-attn installed: `flash_attn.bert_padding.pad_input`
      - Otherwise: `verl.utils.attention_padding_utils.pad_input`

    Users can call this function directly without worrying about the underlying device.
    """
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    """
    Unified entry point for `rearrange` across backends.

    Dynamically dispatches to the appropriate implementation:
      - With flash-attn installed: `flash_attn.bert_padding.rearrange`
      - Otherwise: `einops.rearrange`, re-exported from
        `verl.utils.attention_padding_utils`

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    """
    Unified entry point for `unpad_input` across backends.

    Dynamically dispatches to the appropriate implementation:
      - With flash-attn installed: `flash_attn.bert_padding.unpad_input`
      - Otherwise: `verl.utils.attention_padding_utils.unpad_input`

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
