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

import ast
import inspect
import textwrap

import pytest
import torch

from verl.models.mcore import util


def test_build_vlm_attn_mask_thd_keeps_valid_token_mask():
    """VLM bridges need the 2D mask to prepare MRoPE and packed sequences."""
    input_ids = torch.nested.nested_tensor(
        [torch.tensor([11, 12, 13]), torch.tensor([21, 22])],
        layout=torch.jagged,
    )

    padded_input_ids, attention_mask = util.build_vlm_attn_mask_thd(input_ids, pad_token_id=0)

    torch.testing.assert_close(padded_input_ids, torch.tensor([[11, 12, 13], [21, 22, 0]]))
    torch.testing.assert_close(
        attention_mask,
        torch.tensor([[True, True, True], [True, True, False]]),
    )


def test_qwen3_vl_bridge_consumes_then_clears_thd_mask():
    """The pinned mbridge must not pass the 2D padding mask to THD attention."""
    pytest.importorskip("mbridge")
    from mbridge.models.qwen3_vl.model import Qwen3VLModel

    tree = ast.parse(textwrap.dedent(inspect.getsource(Qwen3VLModel.forward)))
    rope_call_line = None
    clear_line = None
    language_model_call_line = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "get_rope_index":
            if any(keyword.arg == "attention_mask" for keyword in node.keywords):
                rope_call_line = node.lineno
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "attention_mask" for target in node.targets):
                if isinstance(node.value, ast.Constant) and node.value.value is None:
                    clear_line = node.lineno
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                if node.func.attr == "language_model":
                    language_model_call_line = node.lineno

    assert rope_call_line is not None
    assert clear_line is not None
    assert language_model_call_line is not None
    assert rope_call_line < clear_line < language_model_call_line
