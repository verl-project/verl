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

import os
import sys
import types
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine, FSDPEngineWithLMHead
from verl.workers.engine.utils import enable_batch_invariance


def test_batch_invariant_is_off_by_default():
    assert FSDPEngineConfig().batch_invariant is False
    assert FSDPEngine._batch_invariant is False


def test_batch_invariant_rejects_fused_kernels(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(transformer_impl.FSDPEngine, "_init_device_mesh", lambda self: None)
    monkeypatch.setattr("verl.workers.engine.fsdp.utils.apply_npu_fsdp_patches", lambda model_config: None)
    model_config = types.SimpleNamespace(use_remove_padding=True, use_fused_kernels=True)

    with pytest.raises(ValueError, match="use_fused_kernels"):
        FSDPEngine(model_config, FSDPEngineConfig(batch_invariant=True), None, None)


def _fake_vllm_batch_invariant(monkeypatch, install):
    """Stand-in for vLLM's batch_invariant module; ``install`` runs inside init_batch_invariance."""
    fake = types.ModuleType("vllm.model_executor.determinism.batch_invariant")
    fake._batch_invariant_MODE = False

    def init_batch_invariance():
        fake._batch_invariant_MODE = install() is not False

    fake.init_batch_invariance = init_batch_invariance
    monkeypatch.setitem(sys.modules, "vllm.model_executor.determinism.batch_invariant", fake)


def test_enable_batch_invariance_installs_vllm_overrides_and_reverts_its_nccl_profile(monkeypatch):
    calls = []

    def install():
        calls.append(os.environ["VLLM_BATCH_INVARIANT"])
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["NCCL_MAX_NCHANNELS"] = "1"
        os.environ["NCCL_PROTO"] = "Simple"

    _fake_vllm_batch_invariant(monkeypatch, install)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("NCCL_MAX_NCHANNELS", raising=False)
    monkeypatch.setenv("NCCL_PROTO", "LL128")

    enable_batch_invariance()

    assert calls == ["1"]
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert "NCCL_MAX_NCHANNELS" not in os.environ
    assert os.environ["NCCL_PROTO"] == "LL128"


def test_enable_batch_invariance_keeps_autograd_torch_bmm(monkeypatch):
    # vLLM 0.29-0.31 also assign torch.bmm to a Triton launch without autograd; the aten::bmm override stays.
    original_bmm = torch.bmm
    monkeypatch.setattr(torch, "bmm", original_bmm)

    def install():
        torch.bmm = lambda a, b: a

    _fake_vllm_batch_invariant(monkeypatch, install)
    enable_batch_invariance()

    assert torch.bmm is original_bmm


def test_enable_batch_invariance_rejects_a_skipped_install(monkeypatch):
    # vLLM skips the install when its cached envs read VLLM_BATCH_INVARIANT as unset.
    _fake_vllm_batch_invariant(monkeypatch, lambda: False)

    with pytest.raises(RuntimeError, match="VLLM_BATCH_INVARIANT"):
        enable_batch_invariance()


def test_enable_batch_invariance_keeps_the_exported_cublas_workspace(monkeypatch):
    # On SM80 vLLM leaves CUBLAS_WORKSPACE_CONFIG at the :4096:8 its env overrides set.
    def install():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    _fake_vllm_batch_invariant(monkeypatch, install)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    enable_batch_invariance()

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


@pytest.mark.parametrize("use_remove_padding", [True, False])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_batch_invariant_log_probs_divide_fp32_logits_by_temperature(use_remove_padding, temperature):
    # The vLLM sampler casts the logits to fp32 before dividing by the fp32 temperature.
    torch.manual_seed(0)
    seq_lengths = torch.tensor([3, 2])
    total_nnz, vocab_size = int(seq_lengths.sum()), 64
    cu_seqlens = torch.cat([torch.tensor([0]), seq_lengths.cumsum(0)])
    input_ids = torch.nested.nested_tensor_from_jagged(torch.randint(0, vocab_size, (total_nnz,)), offsets=cu_seqlens)
    labels = torch.randint(0, vocab_size, (total_nnz,))
    logits_rmpad = torch.randn(total_nnz, vocab_size, dtype=torch.bfloat16) * 8
    output = types.SimpleNamespace()
    output_args = {"input_ids_rmpad_rolled": labels, "temperature_is_one": temperature == 1.0}
    if use_remove_padding:
        output.logits = logits_rmpad.unsqueeze(0)
        output_args.update(temperature_rmpad=torch.full((total_nnz,), temperature), pad_size=0)
    else:
        output.logits = torch.zeros(2, 3, vocab_size, dtype=torch.bfloat16)
        output.logits[0, :3], output.logits[1, :2] = logits_rmpad[:3], logits_rmpad[3:]
        output_args["temperature"] = torch.full((2,), temperature)
    micro_batch = TensorDict({"input_ids": input_ids}, batch_size=[])
    tu.assign_non_tensor(micro_batch, use_remove_padding=use_remove_padding, pad_mode=DatasetPadMode.NO_PADDING)
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    engine._batch_invariant = True

    narrow = torch.nested.narrow
    with patch.object(torch.nested, "narrow", wraps=narrow) as narrow_calls:
        model_output = FSDPEngineWithLMHead.prepare_model_outputs(engine, output, output_args, micro_batch, None)

    ref = torch.log_softmax(logits_rmpad.float() / temperature, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(model_output["log_probs"].values(), ref)
    if not use_remove_padding:
        # at T=1 the unscaled logits are the scaled ones, so they are packed once
        assert narrow_calls.call_count == (1 if temperature == 1.0 else 2)
