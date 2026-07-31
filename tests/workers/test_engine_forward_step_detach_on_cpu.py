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
"""Regression guard for verl#6698.

FSDPEngineWithLMHead.forward_step used to return ``model_output`` tensors
(log_probs/entropy) still attached to the autograd graph. forward_backward_batch
collects these per-micro-batch outputs in ``output_lst`` until the WHOLE batch
finishes, so the retained graph pinned, per training micro-batch, whatever the
graph still referenced after backward — most notably the activation-checkpoint
frame's saved block input (the embedding output, which requires grad under
PEFT's ``enable_input_require_grads``) and its gradient buffer. On long-sequence
LoRA runs this leaked ~0.27 GiB per micro-batch and OOM'd the actor update.

This test reproduces the retention mechanism in miniature: a frozen embedding
whose output is forced to require grad (the PEFT situation), a checkpointed
block, and a weakref on the checkpoint-saved block input. After backward, with
``output`` still held (as output_lst does), the saved input must be collectible
and the returned model_output must carry no grad_fn.
"""

import gc
import weakref
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from verl.utils.device import get_device_id
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

VOCAB, HIDDEN, SEQ = 16, 8, 6


class _TinyCheckpointedLM(torch.nn.Module):
    """Frozen embedding + checkpointed trainable block, mimicking a PEFT/LoRA
    base model with gradient checkpointing and enable_input_require_grads."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.embed.weight.requires_grad_(False)  # frozen base
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)  # the trainable part
        self.saved_block_input = None  # weakref to the checkpoint-saved input

    def forward(self, input_ids=None, use_cache=False):
        x = self.embed(input_ids)
        # PEFT enable_input_require_grads: embedding output requires grad so
        # gradients can flow into trainable params under checkpointing.
        x.requires_grad_(True)
        hidden = torch.utils.checkpoint.checkpoint(self.proj, x, use_reentrant=False)
        self.saved_block_input = weakref.ref(x)
        return SimpleNamespace(hidden=hidden)


def _make_engine_stub():
    """Bypass __init__; provide only what forward_step touches."""
    eng = object.__new__(FSDPEngineWithLMHead)
    eng._autocast_dtype = torch.float32  # skip autocast
    eng.module = _TinyCheckpointedLM().to(get_device_id())
    eng.prepare_model_inputs = lambda micro_batch: ({"input_ids": micro_batch["input_ids"]}, {})
    eng.prepare_model_outputs = lambda output, output_args, micro_batch, logits_processor_func: {
        "log_probs": output.hidden.sum(-1)
    }
    eng.get_data_parallel_group = lambda: None
    return eng


def _loss_fn(model_output, data, dp_group=None):
    return model_output["log_probs"].sum(), {"dummy_metric": 1.0}


def test_forward_step_output_carries_no_grad_fn_and_releases_graph():
    eng = _make_engine_stub()
    micro_batch = TensorDict(
        {"input_ids": torch.randint(0, VOCAB, (1, SEQ))},
        batch_size=[1],
    )

    loss, output = FSDPEngineWithLMHead.forward_step(eng, micro_batch, _loss_fn, forward_only=False)

    # The live loss must still drive backward into the trainable params.
    assert loss.grad_fn is not None
    loss.backward()
    assert eng.module.proj.weight.grad is not None

    # Contract: nothing in the collected per-micro-batch output may keep the
    # autograd graph alive once the loss reference is dropped.
    for key, value in output["model_output"].items():
        assert not (torch.is_tensor(value) and value.grad_fn is not None), (
            f"model_output[{key!r}] still attached to the autograd graph"
        )

    saved_input = eng.module.saved_block_input
    assert saved_input() is not None  # sanity: alive while loss exists
    del loss
    gc.collect()
    # `output` is intentionally still held, like forward_backward_batch's
    # output_lst holds it across the remaining micro-batches of the batch.
    assert saved_input() is None, (
        "checkpoint-saved block input (embedding output) survived backward: "
        "the per-micro-batch output is retaining the autograd graph"
    )
    assert output["loss"] is not None
