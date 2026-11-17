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

"""Numerical integration check for the real DP=2, CP=2 Megatron training path.

Run on four GPUs with the matching Megatron-Core and Transformer Engine:

    torchrun --standalone --nproc-per-node=4 \
        tests/workers/engine/megatron/test_cp_dp_gradient_normalization.py

The reference uses the same tiny GPT weights without context partitioning and
computes the global supervised-token mean directly. The distributed side calls
the production engine, pipeline schedule, DDP, and gradient finalizer. There are
two microbatches per replica, unequal sequence lengths with a mix of even and odd
lengths, varying supervision counts, and a case where one DP replica has no
supervised tokens. No model download is needed. Ordinary single-process pytest
runs skip this GPU integration test.
"""

import argparse
import copy
import gc
import json
import os
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

_LENGTHS = (8, 11, 7, 10)
_VOCAB_SIZE = 64


def _samples(supervision_counts, device):
    samples = []
    for index, (length, supervised) in enumerate(zip(_LENGTHS, supervision_counts, strict=True)):
        ids = (torch.arange(length, device=device) * 7 + index * 11 + 3) % _VOCAB_SIZE
        mask = torch.zeros(length, device=device, dtype=torch.bool)
        if supervised:
            mask[-supervised:] = True
        samples.append((ids, mask))
    return samples


def _initialize_parallel(cp_size):
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        create_gloo_process_groups=False,
    )


def _make_model(cp_size, per_token):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.manual_seed(1729)
    model_parallel_cuda_manual_seed(1729)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        context_parallel_size=cp_size,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.flash,
        calculate_per_token_loss=per_token,
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        cp_comm_type="p2p",
    )
    raw_model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=32,
        position_embedding_type="rope",
        share_embeddings_and_output_weights=False,
        parallel_output=True,
    ).cuda()
    model = Float16Module(config, raw_model)
    model.train()
    return config, raw_model, model


def _reference(supervision_counts):
    config, raw_model, model = _make_model(cp_size=1, per_token=False)
    state = {
        name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in raw_model.state_dict().items()
    }
    total_supervised = sum(supervision_counts)
    loss = torch.zeros((), device="cuda", dtype=torch.float32)
    for ids, mask in _samples(supervision_counts, "cuda"):
        logits = model(
            input_ids=ids.unsqueeze(0),
            position_ids=torch.arange(ids.numel(), device="cuda").unsqueeze(0),
            attention_mask=None,
        )
        token_losses = F.cross_entropy(logits[0, :-1].float(), ids[1:], reduction="none")
        loss = loss + (token_losses * mask[1:]).sum() / total_supervised
    loss.backward()
    gradients = {
        name: parameter.grad.detach().float().cpu().clone() for name, parameter in raw_model.named_parameters()
    }
    value = float(loss.detach())
    del loss, logits, token_losses, model, raw_model, config
    gc.collect()
    torch.cuda.empty_cache()
    return state, gradients, value


def _distributed(state, supervision_counts, per_token):
    from megatron.core import parallel_state as mpu
    from megatron.core.distributed import DistributedDataParallel, finalize_model_grads
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
    from verl.workers.utils.losses import sft_loss

    config, raw_model, model = _make_model(cp_size=2, per_token=per_token)
    raw_model.load_state_dict(state)
    ddp = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
            average_in_collective=False,
        ),
        module=model,
    )
    config.finalize_model_grads_func = finalize_model_grads
    ddp.zero_grad_buffer()

    # Build the actual engine around the small model without checkpoint loading,
    # tokenizer/model downloads, Ray actors, or an optimizer step.
    engine = object.__new__(MegatronEngineWithLMHead)
    engine.module = [ddp]
    engine.tf_config = config
    engine.engine_config = SimpleNamespace(
        context_parallel_size=2,
        dynamic_context_parallel=False,
        use_remove_padding=True,
        use_fused_kernels=False,
        pad_bshd_to_minibatch_max=False,
        pad_to_length=False,
        pad_to_length_bucket=None,
        entropy_from_logits_with_chunking=False,
    )
    engine.model_config = SimpleNamespace(
        mtp=SimpleNamespace(enable=False, enable_train=False),
        hf_config=SimpleNamespace(architectures=["Qwen2ForCausalLM"]),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    engine.enable_routing_replay = False
    first = 2 * mpu.get_data_parallel_rank()
    rows = _samples(supervision_counts, "cuda")[first : first + 2]
    data = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor([ids for ids, _ in rows], layout=torch.jagged),
            "loss_mask": torch.nested.nested_tensor([mask for _, mask in rows], layout=torch.jagged),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        gradient_accumulation_steps=2,
        pad_mode=DatasetPadMode.NO_PADDING,
        temperature=1.0,
    )
    output = engine.forward_backward_batch(data, loss_function=partial(sft_loss, None))
    assert data["batch_num_tokens"] == sum(supervision_counts)
    assert data["dp_size"] == 2
    assert len(output["loss"]) == 2
    metadata = {
        "global_supervised_tokens": int(data["batch_num_tokens"]),
        "global_routed_tokens": int(data["routed_num_tokens"]) if per_token else None,
        "dp_size": int(data["dp_size"]),
        "cp_size": 2,
        "microbatches_per_replica": len(output["loss"]),
        "sequence_lengths": list(_LENGTHS),
        "supervised_tokens_by_replica": [sum(supervision_counts[:2]), sum(supervision_counts[2:])],
    }
    if per_token:
        assert metadata["global_routed_tokens"] == sum(_LENGTHS)
    gradients = {
        name: parameter.main_grad.detach().float().cpu().clone() for name, parameter in raw_model.named_parameters()
    }
    loss = torch.tensor(sum(output["loss"]), dtype=torch.float64, device="cuda")
    dist.all_reduce(loss)
    loss /= dist.get_world_size()
    value = float(loss)
    del output, data, engine, ddp, model, raw_model, config
    gc.collect()
    torch.cuda.empty_cache()
    return gradients, value, metadata


def _run(output_path=None):
    from megatron.core import parallel_state as mpu

    assert int(os.environ.get("WORLD_SIZE", "1")) == 4, "Run this integration check with four torchrun ranks"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=4))
    results = []
    try:
        for name, counts in (("unequal_supervision", (2, 7, 1, 4)), ("zero_supervision_replica", (2, 7, 0, 0))):
            _initialize_parallel(cp_size=1)
            state, expected, expected_loss = _reference(counts)
            mpu.destroy_model_parallel()
            dist.barrier()
            token_mean_gradients = None
            token_mean_loss = None
            for per_token in (False, True):
                _initialize_parallel(cp_size=2)
                actual, actual_loss, metadata = _distributed(state, counts, per_token)
                assert actual.keys() == expected.keys()
                squared_error = sum((actual[key] - expected[key]).square().sum() for key in actual)
                squared_reference = sum(expected[key].square().sum() for key in actual)
                relative_l2 = float((squared_error / squared_reference).sqrt())
                max_error = max(float((actual[key] - expected[key]).abs().max()) for key in actual)
                # BF16 attention and microbatch accumulation have different rounding
                # orders under CP. These bounds remain well below the 10%+ errors
                # caused by replica-local supervised-token denominators.
                assert relative_l2 < 0.025, (name, per_token, relative_l2, max_error)
                assert abs(actual_loss - expected_loss) < 0.005, (name, per_token, actual_loss, expected_loss)
                record = {
                    "case": name,
                    "calculate_per_token_loss": per_token,
                    "relative_gradient_l2": relative_l2,
                    "max_gradient_abs_error": max_error,
                    "loss": actual_loss,
                    "reference_loss": expected_loss,
                    "gradient_norm_ratio_to_reference": float(
                        (sum(actual[key].square().sum() for key in actual) / squared_reference).sqrt()
                    ),
                    "metadata": metadata,
                }
                if per_token:
                    regime_squared_error = sum(
                        (actual[key] - token_mean_gradients[key]).square().sum() for key in actual
                    )
                    regime_squared_reference = sum(token_mean_gradients[key].square().sum() for key in actual)
                    regime_relative_l2 = float((regime_squared_error / regime_squared_reference).sqrt())
                    regime_norm_ratio = float(
                        (sum(actual[key].square().sum() for key in actual) / regime_squared_reference).sqrt()
                    )
                    assert regime_relative_l2 < 0.01, (name, regime_relative_l2)
                    assert abs(regime_norm_ratio - 1.0) < 0.01, (name, regime_norm_ratio)
                    assert abs(actual_loss - token_mean_loss) < 1e-8
                    record["relative_gradient_l2_to_token_mean_regime"] = regime_relative_l2
                    record["gradient_norm_ratio_to_token_mean_regime"] = regime_norm_ratio
                    record["loss_difference_between_regimes"] = actual_loss - token_mean_loss
                else:
                    token_mean_gradients = actual
                    token_mean_loss = actual_loss
                if dist.get_rank() == 0:
                    print(json.dumps(record), flush=True)
                results.append(record)
                mpu.destroy_model_parallel()
                dist.barrier()
    finally:
        if mpu.model_parallel_is_initialized():
            mpu.destroy_model_parallel()
        dist.destroy_process_group()
    if int(os.environ["RANK"]) == 0 and output_path is not None:
        Path(output_path).write_text(json.dumps(results, indent=2) + "\n")


def test_cp_dp_gradient_normalization():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4 or not torch.cuda.is_available():
        pytest.skip("Requires four GPUs and torchrun; see the module docstring")
    _run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    _run(parser.parse_args().output)
