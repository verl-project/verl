# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Weight-only ``qat.mode=mxfp4`` (``qat.activation=none``) on a real Megatron-Core MoE, single GPU.

The experts are TE grouped-GEMM linears, which modelopt quantizes with one quantizer per expert
(``...linear_fc1.weight_quantizer.{i}``). This is the layout the DeepSeek-V4 recipe sees; the toy
nn.Linear tests cannot reach it.
"""

import re
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformer_engine")
pytest.importorskip("modelopt.torch.quantization")

import verl.models.mcore  # noqa: F401, E402  # isort: skip
import torch.distributed as dist  # noqa: E402
from megatron.core import parallel_state as mpu  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec  # noqa: E402
from megatron.core.models.gpt.gpt_model import GPTModel  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402
from modelopt.torch.quantization.nn import TensorQuantizer  # noqa: E402

from verl.utils.qat.modelopt import apply_qat_to_modules  # noqa: E402

# Routed-expert weight quantizers: one per expert of a TE grouped linear (``weight_quantizer.{i}``).
_EXPERT_WEIGHT_QUANTIZER = re.compile(r"(^|\.)mlp\.experts\..*weight_quantizer(\.\d+)?$")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_LAYERS, NUM_EXPERTS, HIDDEN, SEQ, VOCAB = 2, 4, 256, 64, 128
_QAT_CFG = SimpleNamespace(
    enable=True,
    mode="mxfp4",
    activation="none",
    group_size=16,
    ignore_patterns=["lm_head", "embed_tokens", "re:.*mlp.gate$"],
)


@pytest.fixture
def moe_model(tmp_path):
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1
    )
    mpu.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    try:
        config = TransformerConfig(
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN,
            num_attention_heads=4,
            ffn_hidden_size=512,
            num_moe_experts=NUM_EXPERTS,
            moe_ffn_hidden_size=128,
            moe_router_topk=2,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
        )
        spec = get_gpt_layer_with_transformer_engine_spec(num_experts=NUM_EXPERTS, moe_grouped_gemm=True)
        model = (
            GPTModel(config=config, transformer_layer_spec=spec, vocab_size=VOCAB, max_sequence_length=SEQ)
            .cuda()
            .bfloat16()
        )
        yield model
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def test_grouped_gemm_experts_get_one_mxfp4_quantizer_per_expert(moe_model):
    (model,) = apply_qat_to_modules([moe_model], _QAT_CFG)

    enabled = [n for n, q in model.named_modules() if isinstance(q, TensorQuantizer) and q.is_enabled]
    assert enabled, "no quantizer enabled"
    assert all(_EXPERT_WEIGHT_QUANTIZER.search(n) for n in enabled), enabled[:5]
    # one per expert, per expert linear (fc1, fc2), per layer
    assert len(enabled) == NUM_LAYERS * 2 * NUM_EXPERTS, (len(enabled), enabled[:5])

    fc1 = model.decoder.layers[0].mlp.experts.linear_fc1
    w = fc1.weight0.detach()
    fq = fc1.weight_quantizer[0](w)
    assert not torch.equal(fq, w), "random bf16 weights are not on the MXFP4 grid"
    assert torch.equal(fc1.weight_quantizer[0](fq), fq), "fake-quant must be idempotent on-grid"
    from verl.utils.qat.fp4 import fp4_quant

    assert torch.equal(fq, fp4_quant(w, dequant=True)), "per-expert QDQ must be verl's TileLang MXFP4 kernel"

    # Forward + backward through the fake-quantized grouped GEMM reach every expert weight (STE).
    tokens = torch.randint(0, VOCAB, (2, SEQ), device="cuda")
    pos = torch.arange(SEQ, device="cuda").unsqueeze(0).expand(2, -1)
    loss = model(tokens, pos, attention_mask=None, labels=tokens).float().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert fc1.weight0.grad is not None and torch.isfinite(fc1.weight0.grad).all()
