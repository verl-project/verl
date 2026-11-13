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
"""Opt-in numerical test through the real CSA2 model, verl engine and Megatron DDP.

Run on four GPUs with the paired DeepSeek-V4.1 Megatron/Bridge stack:

    DSV41_RUN_CHUNK_TRAINING=1 torchrun --standalone --nproc-per-node=4 \
        -m pytest -q -s tests/models/test_dsv41_chunk_training.py

The reduced model keeps SWA, compressed Full/Reuse/Reindex attention, mHC and MoE.
Two micro-batches and unequal DP supervision exercise accumulation and token normalization.
Real lengths 96 and 98 tile CP2 exactly; 98 still needs 98/100/102 physical
rows at K1/K2/K3, so chunk-dependent padding remains covered.
No checkpoint, tokenizer download or large model is needed.
"""

import json
import os
from copy import deepcopy
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    os.environ.get("DSV41_RUN_CHUNK_TRAINING") != "1",
    reason="opt in with DSV41_RUN_CHUNK_TRAINING=1 and four torchrun workers",
)


def _tiny_provider(per_token, moe_z_loss_coeff=0.0):
    from megatron.bridge.models.deepseek.deepseek_v41_provider import DeepSeekV41ModelProvider
    from megatron.core.transformer.enums import AttnBackend

    return DeepSeekV41ModelProvider(
        vocab_size=128,
        should_pad_vocab=False,
        seq_length=128,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        sequence_parallel=False,
        params_dtype=torch.float32,
        bf16=False,
        fp16=False,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        transformer_impl="transformer_engine",
        attention_backend=AttnBackend.local,
        experimental_attention_variant="dsv4_hybrid",
        dsv4_version="v4.1",
        dsa_kernel_backend="none",
        num_layers=12,
        hidden_size=32,
        num_attention_heads=4,
        q_lora_rank=16,
        v_head_dim=16,
        qk_pos_emb_head_dim=8,
        output_projection_groups=2,
        output_projection_lora_rank=8,
        qk_layernorm=True,
        layernorm_epsilon=1e-20,
        normalization="RMSNorm",
        rotary_base=10000,
        csa_compress_rotary_base=160000,
        original_max_position_embeddings=65536,
        rotary_scaling_factor=16,
        beta_fast=32,
        beta_slow=1,
        mscale=0,
        mscale_all_dim=0,
        csa_window_size=4,
        csa_compress_ratios=[ratio for r in (0, 2, 2, 1, 1, 1) for ratio in (r, 0)],
        csa2_kv_source_layers=[2, 6],
        csa2_index_source_layers=[2, 6, 8],
        csa2_candidate_source_layer=6,
        csa2_candidate_topk_blocks=2,
        csa2_candidate_block_size=2,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_loss_coeff=0,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=48,
        moe_shared_expert_intermediate_size=48,
        moe_token_dispatcher_type="alltoall",
        moe_router_score_function="sqrtsoftplus",
        moe_router_dtype="fp32",
        moe_router_topk_scaling_factor=1.5,
        moe_router_enable_expert_bias=False,
        moe_aux_loss_coeff=0,
        moe_z_loss_coeff=moe_z_loss_coeff,
        activation_func=F.silu,
        gated_linear_unit=True,
        activation_func_clamp_value=10,
        add_bias_linear=False,
        hidden_dropout=0,
        attention_dropout=0,
        calculate_per_token_loss=per_token,
    )


@pytest.fixture(scope="module")
def groups():
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "0")) != 4:
        pytest.skip("requires four CUDA torchrun workers (CP2/DP2)")
    from megatron.core import parallel_state as mpu
    from megatron.core.process_groups_config import ProcessGroupCollection

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=2)
    yield ProcessGroupCollection.use_mpu_process_groups()
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


def _batch(final_chunk_only, length):
    from megatron.core import parallel_state as mpu
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    dp_rank = mpu.get_data_parallel_rank()
    generator = torch.Generator().manual_seed(734 + dp_rank)
    ids = [torch.randint(1, 128, (length,), generator=generator, device="cpu").cuda() for _ in range(2)]
    masks = []
    for micro in range(2):
        mask = torch.zeros(length, device="cuda")
        positions = [length - 15, length - 7, length - 1] if final_chunk_only else [1, 31, 33, 63, 65, 81, length - 1]
        mask[positions[: 1 + micro + dp_rank]] = 1
        masks.append(mask)
    data = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor(ids, layout=torch.jagged),
            "loss_mask": torch.nested.nested_tensor(masks, layout=torch.jagged),
            "temperature": torch.ones(2, device="cuda"),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=1, gradient_accumulation_steps=2)
    return data


def _engine(model, provider):
    from verl.workers.config import McoreEngineConfig
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    # Supply a real initialized model while bypassing checkpoint loading and optimizer creation.
    engine = MegatronEngineWithLMHead.__new__(MegatronEngineWithLMHead)
    engine.engine_config = McoreEngineConfig(
        context_parallel_size=2, use_fused_kernels=False, use_remove_padding=True, dynamic_context_parallel=False
    )
    engine.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"], model_type="deepseek_v41"),
        tokenizer=SimpleNamespace(pad_token_id=0),
        mtp=SimpleNamespace(enable=False, enable_train=False),
    )
    engine.tf_config = provider
    engine.module = [model]
    engine.enable_routing_replay = False
    return engine


def _build_model_and_engine(groups, monkeypatch, per_token, moe_z_loss_coeff=0.0):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    for key in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(key, raising=False)
    provider = _tiny_provider(per_token, moe_z_loss_coeff)
    provider._pg_collection = groups
    provider.finalize()
    # Bridge enables per-token normalization under CP by default; exercise both supported regimes.
    provider.calculate_per_token_loss = per_token
    model = provider.provide().cuda().train()
    ddp = DistributedDataParallel(
        provider,
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=False
        ),
        model,
        disable_bucketing=True,
        pg_collection=groups,
    )
    provider.no_sync_func = ddp.no_sync
    provider.finalize_model_grads_func = finalize_model_grads
    return model, ddp, _engine(ddp, provider)


def _record_result(row):
    row = {"rank": dist.get_rank(), **row}
    directory = os.environ.get("DSV41_CHUNK_EVIDENCE_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"rank{dist.get_rank()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    if dist.get_rank() == 0:
        print(json.dumps({key: value for key, value in row.items() if key != "parameter_errors"}), flush=True)


def _check_training(groups, monkeypatch, per_token, final_chunk_only, length, moe_z_loss_coeff=0.0, policy=False):
    from verl.workers.utils.losses import sft_loss

    loss_function = partial(sft_loss, None)
    if policy:
        from math import log

        from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
        from verl.workers.config import ActorConfig

        actor = ActorConfig(strategy="megatron", rollout_n=1, ppo_micro_batch_size_per_gpu=1, clip_ratio=0.2)

        def loss_function(model_output, data, dp_group=None):
            values = model_output["log_probs"].values().reshape(1, length)
            mask = torch.ones_like(values)
            return compute_policy_loss_gspo(
                torch.full_like(values, -log(128)), values, mask, mask, loss_agg_mode="token-mean", config=actor
            )

    model, ddp, engine = _build_model_and_engine(groups, monkeypatch, per_token, moe_z_loss_coeff)
    initial = deepcopy(model.state_dict())
    reference = None

    for chunks in (1, 2, 3):
        monkeypatch.setenv("DSV41_SEQ_CHUNKS", str(chunks))
        model.load_state_dict(initial)
        ddp.zero_grad_buffer()
        result = engine.forward_backward_batch(_batch(final_chunk_only, length), loss_function)
        parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        grads = {name: parameter.main_grad.detach().clone() for name, parameter in parameters.items()}
        assert all(torch.isfinite(value).all() for value in grads.values())
        assert any(torch.count_nonzero(value) for value in grads.values())
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.add_(grads[name], alpha=-1e-3)
        updates = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        loss = sum(result["loss"])
        metrics = {"loss_max_abs": 0.0, "grad_max_abs": 0.0, "grad_relative_l2": 0.0, "update_max_abs": 0.0}
        is_reference = reference is None
        errors = {}
        if is_reference:
            reference = loss, grads, updates
        else:
            ref_loss, ref_grads, ref_updates = reference
            errors = {
                name: (
                    (grads[name] - ref_grads[name]).abs().max().item(),
                    ((grads[name] - ref_grads[name]).norm() / ref_grads[name].norm().clamp_min(1e-12)).item(),
                )
                for name in grads
            }
            metrics = {
                "loss_max_abs": abs(loss - ref_loss),
                "grad_max_abs": max(error[0] for error in errors.values()),
                "grad_relative_l2": (
                    sum((grads[name] - ref_grads[name]).square().sum() for name in grads).sqrt()
                    / sum(value.square().sum() for value in ref_grads.values()).sqrt().clamp_min(1e-12)
                ).item(),
                "update_max_abs": max((updates[name] - ref_updates[name]).abs().max().item() for name in updates),
            }
            if dist.get_rank() == 0:
                print(
                    json.dumps(
                        {
                            "chunks": chunks,
                            "worst_gradient_errors": sorted(errors.items(), key=lambda item: item[1][0], reverse=True)[
                                :5
                            ],
                        }
                    ),
                    flush=True,
                )
            directory = os.environ.get("DSV41_CHUNK_EVIDENCE_DIR")
            if directory and moe_z_loss_coeff and os.environ.get("DSV41_CHUNK_SAVE_GRADS") == "1":
                torch.save(
                    {
                        "reference": {name: value.cpu() for name, value in ref_grads.items()},
                        "actual": {name: value.cpu() for name, value in grads.items()},
                    },
                    Path(directory) / f"gradients-length{length}-rank{dist.get_rank()}-K{chunks}.pt",
                )
        _record_result(
            {
                "case": "gspo_training" if policy else ("aux_training" if moe_z_loss_coeff else "training"),
                "chunks": chunks,
                "per_token": per_token,
                "final_chunk_only": final_chunk_only,
                "length": length,
                "moe_z_loss_coeff": moe_z_loss_coeff,
                "loss": loss,
                "parameter_errors": errors,
                **metrics,
            }
        )
        if not is_reference:
            assert loss == pytest.approx(ref_loss, rel=3e-4, abs=3e-5)
            for name in grads:
                torch.testing.assert_close(grads[name], ref_grads[name], rtol=3e-3, atol=3e-5, msg=name)
                torch.testing.assert_close(updates[name], ref_updates[name], rtol=3e-4, atol=3e-7, msg=name)
        dist.barrier()


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("final_chunk_only", [False, True])
@pytest.mark.parametrize("length", [96, 98])
def test_outer_chunk_training_matches_single_pass(groups, monkeypatch, per_token, final_chunk_only, length):
    _check_training(groups, monkeypatch, per_token, final_chunk_only, length)


@pytest.mark.parametrize("length", [96, 98])
def test_outer_chunk_training_preserves_nonzero_moe_z_loss(groups, monkeypatch, length):
    _check_training(groups, monkeypatch, per_token=True, final_chunk_only=False, length=length, moe_z_loss_coeff=1e-3)


@pytest.mark.parametrize("per_token", [False, True])
def test_outer_chunk_gspo_matches_complete_sequence_gradient(groups, monkeypatch, per_token):
    _check_training(groups, monkeypatch, per_token, final_chunk_only=False, length=98, policy=True)


def test_outer_chunk_infer_batch_preserves_outputs_and_existing_gradients(groups, monkeypatch):
    from verl.workers.utils.losses import sft_loss

    model, ddp, engine = _build_model_and_engine(groups, monkeypatch, per_token=True)
    monkeypatch.setenv("DSV41_SEQ_CHUNKS", "1")
    ddp.zero_grad_buffer()
    engine.forward_backward_batch(_batch(False, 98), partial(sft_loss, None))
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    existing_grads = {name: parameter.main_grad.detach().clone() for name, parameter in parameters.items()}
    assert any(torch.count_nonzero(value) for value in existing_grads.values())
    model.eval()
    reference = None
    for chunks in (1, 2, 3):
        monkeypatch.setenv("DSV41_SEQ_CHUNKS", str(chunks))
        result = engine.infer_batch(_batch(False, 98), partial(sft_loss, None))
        log_probs = result["model_output"]["log_probs"].values()
        assert log_probs.numel() == 196
        assert not log_probs.requires_grad
        for name, parameter in parameters.items():
            assert torch.equal(parameter.main_grad, existing_grads[name]), name
        loss = sum(result["loss"])
        metrics = {"loss_max_abs": 0.0, "log_probs_max_abs": 0.0, "log_probs_relative_l2": 0.0}
        if reference is None:
            reference = loss, log_probs.detach().clone()
        else:
            ref_loss, ref_log_probs = reference
            delta = log_probs - ref_log_probs
            metrics = {
                "loss_max_abs": abs(loss - ref_loss),
                "log_probs_max_abs": delta.abs().max().item(),
                "log_probs_relative_l2": (delta.norm() / ref_log_probs.norm().clamp_min(1e-12)).item(),
            }
            assert loss == pytest.approx(ref_loss, rel=3e-4, abs=3e-5)
            torch.testing.assert_close(log_probs, ref_log_probs, rtol=3e-4, atol=3e-5)
        _record_result(
            {"case": "inference", "chunks": chunks, "length": 98, "loss": loss, "grad_max_abs": 0.0, **metrics}
        )
        dist.barrier()
