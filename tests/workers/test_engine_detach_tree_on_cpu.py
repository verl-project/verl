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
"""Regression guard for verl#6698 on the non-FSDP engines.

verl#6699 fixed ``FSDPEngineWithLMHead.forward_step`` (see
tests/workers/test_engine_forward_step_detach_on_cpu.py, which also covers the
VeOmni engines since they inherit that ``forward_step``). The Megatron,
TorchTitan and AutoModel engines collect their per-micro-batch outputs the same
way, so they need the same guard.

Their engine modules cannot be imported on a CPU-only runner (torchtitan /
nemo_automodel / mcore are not installed there), so the behaviour is tested on
the shared ``detach_tree`` helper and the call sites are checked on the source.
"""

import ast
import gc
import weakref
from pathlib import Path

import pytest
import torch

import verl
from verl.workers.engine.utils import detach_tree

ENGINE_DIR = Path(verl.__file__).parent / "workers" / "engine"

# Every place a backend publishes model_output into the dict that outlives the
# micro-batch: forward_step's output_lst entry, or Megatron's forward_data_store.
MODEL_OUTPUT_CALL_SITES = [
    "fsdp/transformer_impl.py",
    "megatron/transformer_impl.py",
    "torchtitan/transformer_impl.py",
    "automodel/transformer_impl.py",
]


class _TinyCheckpointedLM(torch.nn.Module):
    """Frozen embedding + checkpointed trainable block, mimicking a PEFT/LoRA
    base model with gradient checkpointing and enable_input_require_grads."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.embed.weight.requires_grad_(False)  # frozen base
        self.proj = torch.nn.Linear(8, 8, bias=False)  # the trainable part
        self.saved_block_input = None  # weakref to the checkpoint-saved input

    def forward(self, input_ids):
        x = self.embed(input_ids)
        # PEFT enable_input_require_grads: embedding output requires grad so
        # gradients can flow into trainable params under checkpointing.
        x.requires_grad_(True)
        hidden = torch.utils.checkpoint.checkpoint(self.proj, x, use_reentrant=False)
        self.saved_block_input = weakref.ref(x)
        return hidden


def test_detach_tree_releases_the_checkpoint_saved_activation():
    module = _TinyCheckpointedLM()
    hidden = module(torch.randint(0, 16, (1, 6)))

    model_output = detach_tree({"log_probs": hidden.sum(-1), "entropy": hidden.mean(-1)})
    assert all(value.grad_fn is None for value in model_output.values())

    loss = hidden.sum()
    loss.backward()
    assert module.proj.weight.grad is not None

    saved_input = module.saved_block_input
    assert saved_input() is not None  # sanity: alive while the live graph exists
    del loss, hidden
    gc.collect()
    # `model_output` is intentionally still held, the way a backend's output_lst
    # (or Megatron's forward_data_store) holds it across the rest of the batch.
    assert saved_input() is None, (
        "checkpoint-saved block input (embedding output) survived backward: "
        "the per-micro-batch output is retaining the autograd graph"
    )


def test_detach_tree_preserves_values_and_untouched_objects():
    attached = torch.ones(4, requires_grad=True) * 2
    plain = torch.ones(4)
    metric = object()

    result = detach_tree({"a": attached, "b": [plain, (attached, metric)], "c": 1.0})

    assert result["a"].grad_fn is None and torch.equal(result["a"], attached.detach())
    assert result["b"][0] is plain  # no-grad tensors are passed through untouched
    assert isinstance(result["b"], list) and isinstance(result["b"][1], tuple)
    assert result["b"][1][0].grad_fn is None
    assert result["b"][1][1] is metric
    assert result["c"] == 1.0


def test_detach_tree_keeps_nested_tensors_intact():
    """Backends publish per-token outputs as jagged nested tensors."""
    values = torch.arange(6, dtype=torch.float32, requires_grad=True) * 1.0
    nested = torch.nested.nested_tensor_from_jagged(values, torch.tensor([0, 2, 6]))

    detached = detach_tree(nested)

    assert detached.grad_fn is None
    assert torch.equal(detached.values(), values.detach())


@pytest.mark.parametrize("relpath", MODEL_OUTPUT_CALL_SITES)
def test_backend_publishes_model_output_through_detach_tree(relpath):
    tree = ast.parse((ENGINE_DIR / relpath).read_text())

    published = []
    for node in ast.walk(tree):
        # output["model_output"] = <expr>
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "model_output"
                ):
                    published.append(node.value)
        # {"model_output": <expr>, ...}
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "model_output":
                    published.append(value)

    assert published, f"{relpath}: no model_output publication found; did the engine move?"
    for value in published:
        assert isinstance(value, ast.Call) and getattr(value.func, "id", None) == "detach_tree", (
            f"{relpath}:{value.lineno}: model_output must be published through detach_tree, "
            "otherwise every micro-batch's autograd graph stays alive for the whole batch"
        )
