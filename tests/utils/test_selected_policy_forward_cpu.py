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

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


def test_qwen_fused_forward_returns_selected_probabilities_without_forwarding_candidate_ids(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    operator_path = root / "verl/utils/experimental/selected_log_probs.py"
    name = "verl.utils.experimental.selected_log_probs"
    spec = importlib.util.spec_from_file_location(name, operator_path)
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
    monkeypatch.setitem(sys.modules, name, operator)

    class FusedLinear:
        def __init__(self, **kwargs):
            pass

        def forward(self, hidden_states, vocab_weights, input_ids, temperature):
            lp = (hidden_states @ vocab_weights.T / temperature).log_softmax(-1)
            return lp.gather(-1, input_ids[..., None]).squeeze(-1), -(lp.exp() * lp).sum(-1)

    fused = ModuleType("verl.utils.experimental.torch_functional")
    fused.FusedLinearForPPO = FusedLinear
    monkeypatch.setitem(sys.modules, fused.__name__, fused)
    path = root / "verl/models/transformers/qwen3_5.py"
    tree = ast.parse(path.read_text())
    fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "forward_with_torch_backend"
    )
    preamble = ast.parse("from __future__ import annotations").body
    namespace = {
        "torch": torch,
        "DTensor": type("UnusedDTensor", (), {}),
        "get_ulysses_sequence_parallel_world_size": lambda: 1,
        "Qwen3_5CausalLMOutputForPPO": SimpleNamespace,
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=preamble + [fn], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    hidden = torch.randn(1, 5, 7, requires_grad=True)
    head = torch.nn.Linear(7, 13, bias=False)

    class BackboneOutput:
        hidden_states = None

        def __getitem__(self, index):
            return hidden

    def backbone(input_ids, **kwargs):
        assert "selected_token_ids" not in kwargs
        return BackboneOutput()

    model = SimpleNamespace(model=backbone, lm_head=head)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    candidates = torch.tensor([[[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]])
    output = namespace["forward_with_torch_backend"](
        model, input_ids=ids, selected_token_ids=candidates, temperature=0.8
    )
    expected = (hidden @ head.weight.T / 0.8).log_softmax(-1).gather(-1, candidates)
    torch.testing.assert_close(output.candidate_log_probs, expected)
    output.candidate_log_probs.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(head.weight.grad).all()
