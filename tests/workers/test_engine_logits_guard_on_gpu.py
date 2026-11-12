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
"""
``FSDPEngineWithLMHead.prepare_model_outputs`` must not let the log-prob kernel overwrite fp32
logits that the top-k distillation processor keeps for its own backward. This drives the real
method (both padding branches) with the real log-prob dispatch and compares the student gradient
against a plain-PyTorch reference, so reverting the engine's ``inplace_backward`` guard fails here.
"""

import pytest
import torch
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.trainer.distillation.fsdp.losses import _chunked_topk_log_probs
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_name
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

pytestmark = pytest.mark.skipif(
    not verl_F.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE or get_device_name() != "cuda",
    reason="needs flash-attn cross entropy on CUDA",
)

V, K = 32000, 8
SEQ_LENGTHS = [96, 64]


def _engine_stub():
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.use_ulysses_sp = False
    eng.engine_config = type("Cfg", (), {"entropy_checkpointing": False, "batch_invariant": False})()
    return eng


def _student_grad(use_remove_padding, dtype, kernel):
    """Gradient w.r.t. the lm_head logits of: policy log-prob loss + top-k distillation loss."""
    n = sum(SEQ_LENGTHS)
    gen = torch.Generator().manual_seed(0)
    base = (torch.randn(n, V, generator=gen) * 3).to("cuda", dtype)
    labels = torch.randint(0, V, (n,), generator=gen).cuda()
    topk_ids = torch.randint(0, V, (1, n, K), generator=gen).cuda()
    w_logprob = torch.randn(n, generator=gen).cuda()
    w_distill = torch.randn(n, generator=gen).cuda()

    leaf = base.clone().requires_grad_(True)
    logits = leaf * 1.0  # non-leaf, like an lm_head output
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQ_LENGTHS).cumsum(0).tolist()], dtype=torch.int64, device="cuda")

    def processor(student_logits, data):
        distill = _chunked_topk_log_probs(student_logits, topk_ids).float().sum(-1)
        return {"distill": distill}

    class Output:
        pass

    output = Output()
    if use_remove_padding:
        output.logits = logits.unsqueeze(0)
        output_args = {
            "input_ids_rmpad_rolled": labels,
            "temperature_rmpad": torch.ones(n, device="cuda"),
            "pad_size": 0,
        }
    else:
        parts = logits.split(SEQ_LENGTHS)
        output.logits = torch.stack([torch.nn.functional.pad(p, (0, 0, 0, max(SEQ_LENGTHS) - len(p))) for p in parts])
        output_args = {"input_ids_rmpad_rolled": labels, "temperature": torch.ones(len(SEQ_LENGTHS), device="cuda")}

    input_ids = torch.nested.nested_tensor_from_jagged(
        torch.zeros(n, dtype=torch.int64, device="cuda"), offsets=cu_seqlens
    )
    micro_batch = TensorDict({"input_ids": input_ids}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        distillation_use_topk=True,
        distillation_only=False,
        max_response_length=max(SEQ_LENGTHS),
    )

    if kernel == "engine":
        out = FSDPEngineWithLMHead.prepare_model_outputs(_engine_stub(), output, output_args, micro_batch, processor)
        log_probs, distill = out["log_probs"].values().float(), out["distill"].values().float()
    else:
        log_probs = verl_F.logprobs_from_logits_naive(logits, labels).float()
        distill = processor(logits.unsqueeze(0), None)["distill"].squeeze(0)
    (-(log_probs * w_logprob).sum() - (distill * w_distill).sum()).backward()
    return leaf.grad.float()


def _rel_err(use_remove_padding, dtype, monkeypatch):
    monkeypatch.setenv("VERL_DISABLE_FLASH_ATTN_CE", "0")
    grad = _student_grad(use_remove_padding, dtype, "engine")
    ref = _student_grad(use_remove_padding, dtype, "reference")
    return ((grad - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_engine_keeps_fp32_logits_for_topk_distillation(use_remove_padding, monkeypatch):
    assert _rel_err(use_remove_padding, torch.float32, monkeypatch) < 1e-5


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_engine_bf16_gradient_matches_reference(use_remove_padding, monkeypatch):
    # bf16 logits are copied by `.float()`, so the in-place kernel stays safe (and is kept)
    assert _rel_err(use_remove_padding, torch.bfloat16, monkeypatch) < 1e-2
