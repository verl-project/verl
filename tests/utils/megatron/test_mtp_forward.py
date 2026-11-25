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

from inspect import signature
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("megatron.core")

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec, get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.transformer_config import TransformerConfig

from verl.utils.megatron_utils import patch_engine_mtp
from verl.workers.config.model import MtpConfig


@pytest.fixture
def model_parallel(tmp_path):
    dist.init_process_group(backend="nccl", init_method=(tmp_path / "rdzv").as_uri(), rank=0, world_size=1)
    try:
        parallel_state.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        yield
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron forward requires CUDA")
@pytest.mark.skipif("mtp_input_mask" not in signature(GPTModel.forward).parameters, reason="Core lacks MTP input masks")
@pytest.mark.parametrize("detach_encoder", [False, True])
def test_mtp_forward(model_parallel, monkeypatch, detach_encoder):
    monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
    config = TransformerConfig(num_layers=1, mtp_num_layers=1, hidden_size=32, num_attention_heads=2)
    layer_spec = get_gpt_layer_local_spec()
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        mtp_block_spec=get_gpt_mtp_block_spec(config, layer_spec, use_transformer_engine=False),
        vocab_size=32,
        max_sequence_length=4,
        share_embeddings_and_output_weights=False,
    ).cuda()
    patch_engine_mtp(
        model, SimpleNamespace(mtp=MtpConfig(enable=True, enable_train=True, detach_encoder=detach_encoder))
    )

    mtp_outputs = []

    def capture_mtp_output(_layer, _args, output):
        output[0].retain_grad()
        mtp_outputs.append(output)

    model.mtp.layers[0].register_forward_hook(capture_mtp_output)
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    logits = model(
        input_ids=input_ids,
        position_ids=torch.arange(4, device="cuda").unsqueeze(0),
        attention_mask=None,
        labels=torch.tensor([[2, 3, 4, 5]], device="cuda"),
        mtp_input_mask=torch.tensor([[True, False, True, True]], device="cuda"),
    )
    # Zero main-logit gradients isolate the auxiliary loss attached by Core.
    assert logits.shape == (1, 4, 32)
    logits.backward(torch.zeros_like(logits))

    (mtp_output,) = mtp_outputs
    hidden_states, _, _, _, rolled_mask = mtp_output
    expected_mask = torch.tensor([[False, True, True, False]], device="cuda")
    torch.testing.assert_close(rolled_mask, expected_mask)
    torch.testing.assert_close(hidden_states.grad.abs().sum(dim=-1).T > 0, expected_mask)
    assert torch.isfinite(hidden_states.grad).all()
    for module in (model.embedding, model.decoder):
        has_grad = any(p.grad is not None and p.grad.count_nonzero().item() > 0 for p in module.parameters())
        assert has_grad == (not detach_encoder)
