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
"""Opt-in CP2/DP2 image chunk equivalence through the real VLM and verl engine.

Run after other jobs have released four GPUs, with the paired Core/Bridge:

    DSV41_RUN_CHUNK_TRAINING=1 torchrun --standalone --nproc-per-node=4 \
        -m pytest -q -s tests/models/test_dsv41_chunk_vision.py

Four microbatches each contain one THD sequence, with logical lengths 96 or 98.
Images cross physical CP ranks and temporal chunk boundaries; one microbatch is
text-only. DP replicas have unequal supervision. Multiple sequences in a single
chunked engine microbatch remain unsupported and are not exercised here.
No model checkpoint or tokenizer is downloaded.
"""

import math
from dataclasses import asdict, replace
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.models import test_dsv41_chunk_training as chunk_training

pytestmark = chunk_training.pytestmark
groups = chunk_training.groups


def _vision_batch():
    from megatron.core import parallel_state as mpu
    from megatron.core.models.deepseek_v41.image_processing import image_token_types

    from verl.utils import tensordict_utils as tu
    from verl.utils.model import extract_multi_modal_inputs

    dp_rank = mpu.get_data_parallel_rank()
    generator = torch.Generator().manual_seed(947 + dp_rank)
    lengths = (96, 98, 98, 96)
    image_starts = ((19, 43, 68), (5,), (), (5,))
    image_types = image_token_types(3, 3)  # 9x9 ViT patches -> 3x3 projected grid.
    assert image_types.numel() == 14
    ids, masks, multimodal = [], [], []
    for row, (length, starts) in enumerate(zip(lengths, image_starts, strict=True)):
        tokens = torch.randint(1, 127, (length,), generator=generator)
        types = torch.full((1, length), -1, dtype=torch.long)
        patches = []
        for start in starts:
            end = start + image_types.numel()
            assert end <= length
            tokens[start:end] = 127
            types[0, start:end] = image_types
            patches.append(torch.randn(81, 3, 2, 2, generator=generator))
        multimodal.append(
            {
                "pixel_values": torch.cat(patches),
                "image_grid_hws": torch.tensor([[9, 9]] * len(starts)),
                "vision_token_types": types,
            }
            if starts
            else {}
        )
        mask = torch.zeros(length)
        # All supervised targets follow every image. DP0 has 8 targets, DP1 12.
        supervised = [length - 11, length - 8, length - 5, length - 2]
        count = 1 + row // 2 + row % 2 + dp_rank
        assert not starts or supervised[0] >= starts[-1] + image_types.numel()
        mask[supervised[:count]] = 1
        ids.append(tokens)
        masks.append(mask)
    # Match AgentLoopWorkerTQ and the CPU TransferQueue storage backend: fields
    # stay on CPU until the engine moves each microbatch to its CUDA device.
    # Processor dictionaries must remain batched metadata during that split.
    batch = tu.list_of_dict_to_tensordict(
        [
            {
                "input_ids": tokens,
                "loss_mask": mask,
                "temperature": torch.ones(()),
                "multi_modal_inputs": inputs,
            }
            for tokens, mask, inputs in zip(ids, masks, multimodal, strict=True)
        ]
    )
    tu.assign_non_tensor(
        batch,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        gradient_accumulation_steps=4,
    )
    assert int(batch["loss_mask"].values().sum()) == 8 + 4 * dp_rank
    for length, micro in zip(lengths, tu.chunk_tensordict(batch, 4), strict=True):
        inputs = extract_multi_modal_inputs(micro.get("multi_modal_inputs"), input_ids=micro["input_ids"])
        assert micro["input_ids"].offsets().diff().tolist() == [length]
        if inputs:
            assert inputs["vision_token_types"].shape == (1, length)
    return batch


def _build_vision_model(groups, monkeypatch, *, per_token, lora=False, recompute=False, engram=False):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.models.deepseek_v41.config import VisionConfig
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    for key in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(key, raising=False)
    provider = chunk_training._tiny_provider(per_token)
    provider.seq_length = 256
    provider.vision_config = VisionConfig(
        num_hidden_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=48,
        patch_size=2,
        rope_theta=10000,
        downsample_ratio=3,
        max_image_tokens=32,
        min_pixels=36,
    )
    if engram:
        from megatron.core.models.deepseek_v41.config import EngramConfig
        from megatron.core.models.deepseek_v41.engram_hash import EngramLayout

        options = EngramConfig(
            layer_ids=[1],
            num_embeddings=[0],
            max_ngram_size=3,
            vocab_size=101,
            n_heads=2,
            head_dim=8,
            pad_token_id=0,
            compressed_vocab_size=128,
        )
        layout = EngramLayout.from_args(
            SimpleNamespace(**{"engram_" + key: value for key, value in asdict(options).items()})
        )
        provider.engram_config = replace(
            options,
            num_embeddings=[sum(sum(row) for row in layer) for layer in layout.primes],
        )
        # Avoid tokenizer loading: every tiny-vocabulary token maps to itself.
        provider.engram_token_map = torch.arange(128)
    if recompute:
        provider.recompute_granularity = "full"
        provider.recompute_method = "uniform"
        provider.recompute_num_layers = 1
    provider._pg_collection = groups
    provider.finalize()
    provider.calculate_per_token_loss = per_token
    model = provider.provide().cuda().train()
    assert getattr(model, "supports_cp_chunk_images", False), "requires chunk-image-aware paired Megatron-Core"
    if lora:
        from megatron.bridge.peft.lora import LoRA

        model = LoRA(
            target_modules=[
                "vision.patch_embed.proj",
                "vision.blocks.0.attn.wqkv",
                "vision.blocks.0.attn.wo",
                "aligner.w1",
                "aligner.w2",
                "*.layers.2.*.linear_q_down_proj",
                "*.layers.2.*.linear_kv_proj",
                "*.layers.2.*.linear_proj",
            ],
            dim=4,
            alpha=8,
            dropout=0,
            lora_dtype=torch.float32,
        )(model).cuda()
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        assert trainable and all(".adapter." in name for name in trainable), trainable
        assert all(parameter.is_cuda for parameter in model.parameters() if parameter.requires_grad)
    ddp = DistributedDataParallel(
        provider,
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
        ),
        model,
        disable_bucketing=True,
        pg_collection=groups,
    )
    provider.no_sync_func = ddp.no_sync
    provider.finalize_model_grads_func = finalize_model_grads
    return model, ddp, chunk_training._engine(ddp, provider)


def _forward_preserving_inputs(engine, batch, loss_function, *, forward_only=False):
    offsets = batch["input_ids"].offsets().clone()
    values = batch["input_ids"].values().clone()
    if forward_only:
        result = engine.infer_batch(batch, loss_function)
    else:
        result = engine.forward_backward_batch(batch, loss_function)
    # DP's routed-token count must not all-reduce a view of the last offset.
    assert torch.equal(batch["input_ids"].offsets(), offsets)
    assert torch.equal(batch["input_ids"].values(), values)
    return result


def _assert_nonzero_components(grads, *, lora, engram=False):
    prefixes = ("vision.", "aligner.", "decoder.")
    if engram:
        prefixes += ("decoder.layers.2.engram.embed.",)
    for prefix in prefixes:
        selected = {
            name: grad
            for name, grad in grads.items()
            if name.startswith(prefix) and (not lora or name.endswith(".adapter.linear_out.weight"))
        }
        assert selected, (prefix, list(grads))
        assert any(torch.count_nonzero(grad) for grad in selected.values()), prefix


def _component_gradient_errors(grads, reference):
    errors = {}
    prefixes = ("vision.", "aligner.", "decoder.")
    if any(name.startswith("decoder.layers.2.engram.embed.") for name in grads):
        prefixes += ("decoder.layers.2.engram.embed.",)
    for prefix in prefixes:
        names = [name for name in grads if name.startswith(prefix)]
        expected_norm = sum(reference[name].double().square().sum() for name in names).sqrt()
        actual_norm = sum(grads[name].double().square().sum() for name in names).sqrt()
        assert expected_norm > 0 and actual_norm > 0, prefix
        relative_l2 = (
            sum((grads[name].double() - reference[name].double()).square().sum() for name in names).sqrt()
            / expected_norm
        ).item()
        errors[prefix[:-1]] = {
            "relative_l2": relative_l2,
            "norm_ratio": (actual_norm / expected_norm).item(),
            "reference_norm": expected_norm.item(),
        }
        # Absolute tolerances alone can hide the loss of small vision gradients.
        assert relative_l2 < 3e-3, (prefix, errors[prefix[:-1]])
    return errors


def _layout_evidence():
    """Pin each microbatch's CP/chunk cuts and K-dependent physical padding."""
    layouts = []
    for length, starts in ((96, (19, 43, 68)), (98, (5,)), (98, ()), (96, (5,))):
        physical = {str(k): math.ceil(length / (2 * k)) * (2 * k) for k in (1, 2, 8)}
        local_rows = physical["8"] // 16
        assert local_rows in (6, 7)
        spans = [[start, start + 14] for start in starts]
        if spans:
            assert any(begin // local_rows != (end - 1) // local_rows for begin, end in spans)
            assert any(begin // (2 * local_rows) != (end - 1) // (2 * local_rows) for begin, end in spans)
        layouts.append(
            {"logical": length, "physical_by_k": physical, "local_rows_k8": local_rows, "image_spans": spans}
        )
    # The middle image crosses the ordinary CP2 cut and K2 temporal cut at row 48.
    assert any(begin < 48 < end for begin, end in layouts[0]["image_spans"])
    return layouts


@pytest.mark.parametrize(
    "per_token,lora,recompute,engram",
    [(False, False, False, False), (True, False, False, False), (True, True, True, False), (True, False, False, True)],
    ids=["full-legacy-normalization", "full-per-token", "lora-per-token-recompute", "full-per-token-engram"],
)
def test_image_chunk_training_matches_single_pass(groups, monkeypatch, per_token, lora, recompute, engram):
    from verl.workers.utils.losses import sft_loss

    layouts = _layout_evidence()
    model, ddp, engine = _build_vision_model(
        groups, monkeypatch, per_token=per_token, lora=lora, recompute=recompute, engram=engram
    )
    # Bridge intentionally exports flattened adapter/base keys from state_dict;
    # restore the live parameters directly rather than loading that export schema.
    all_parameters = dict(model.named_parameters())
    initial = {name: parameter.detach().clone() for name, parameter in all_parameters.items()}
    buffers = dict(model.named_buffers())
    initial_buffers = {name: buffer.detach().clone() for name, buffer in buffers.items()}
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    reference = None
    for chunks in (1, 2, 8):
        monkeypatch.setenv("DSV41_SEQ_CHUNKS", str(chunks))
        with torch.no_grad():
            for name, parameter in all_parameters.items():
                parameter.copy_(initial[name])
            for name, buffer in buffers.items():
                buffer.copy_(initial_buffers[name])
        ddp.zero_grad_buffer()
        result = _forward_preserving_inputs(engine, _vision_batch(), partial(sft_loss, None))
        grads = {name: parameter.main_grad.detach().clone() for name, parameter in parameters.items()}
        assert all(torch.isfinite(grad).all() for grad in grads.values())
        _assert_nonzero_components(grads, lora=lora, engram=engram)
        loss = sum(result["loss"])
        log_probs = result["model_output"]["log_probs"].values().detach().clone()
        assert log_probs.numel() == 388 and torch.isfinite(log_probs).all()
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.add_(grads[name], alpha=-1e-3)
        updates = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        assert any(not torch.equal(updates[name], initial[name]) for name in updates)
        metrics = {"loss_max_abs": 0.0, "log_probs_max_abs": 0.0, "grad_max_abs": 0.0, "grad_relative_l2": 0.0}
        component_errors = {}
        if reference is None:
            reference = loss, log_probs, grads, updates
        else:
            ref_loss, ref_log_probs, ref_grads, ref_updates = reference
            component_errors = _component_gradient_errors(grads, ref_grads)
            metrics = {
                "loss_max_abs": abs(loss - ref_loss),
                "log_probs_max_abs": (log_probs - ref_log_probs).abs().max().item(),
                "grad_max_abs": max((grads[name] - ref_grads[name]).abs().max().item() for name in grads),
                "grad_relative_l2": (
                    sum((grads[name] - ref_grads[name]).square().sum() for name in grads).sqrt()
                    / sum(grad.square().sum() for grad in ref_grads.values()).sqrt().clamp_min(1e-12)
                ).item(),
                "grad_norm_ratio": (
                    sum(grad.square().sum() for grad in grads.values()).sqrt()
                    / sum(grad.square().sum() for grad in ref_grads.values()).sqrt().clamp_min(1e-12)
                ).item(),
                "update_max_abs": max((updates[name] - ref_updates[name]).abs().max().item() for name in updates),
            }
            assert loss == pytest.approx(ref_loss, rel=3e-4, abs=3e-5)
            torch.testing.assert_close(log_probs, ref_log_probs, rtol=3e-4, atol=3e-5)
            for name in grads:
                torch.testing.assert_close(grads[name], ref_grads[name], rtol=3e-3, atol=3e-5, msg=name)
                torch.testing.assert_close(updates[name], ref_updates[name], rtol=3e-4, atol=3e-7, msg=name)
        chunk_training._record_result(
            {
                "case": "image_engram_training" if engram else ("image_lora_training" if lora else "image_training"),
                "chunks": chunks,
                "per_token": per_token,
                "recompute": recompute,
                "engram": engram,
                "loss": loss,
                "layouts": layouts,
                "supervised_tokens_global": 20,
                "nonzero_components": ["vision", "aligner", "decoder"] + (["engram_embed"] if engram else []),
                "component_gradient_errors": component_errors,
                **metrics,
            }
        )
        dist.barrier()


@pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
def test_image_chunk_inference_preserves_outputs_and_existing_gradients(groups, monkeypatch, lora):
    from verl.workers.utils.losses import sft_loss

    model, ddp, engine = _build_vision_model(groups, monkeypatch, per_token=True, lora=lora)
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    ddp.zero_grad_buffer()
    # Known nonzero buffers let this case check inference purity directly.
    # The training cases separately verify genuine vision and decoder gradients.
    with torch.no_grad():
        for index, parameter in enumerate(parameters.values()):
            parameter.main_grad.fill_(0.125 * (1 + index % 7))
    existing = {name: parameter.main_grad.detach().clone() for name, parameter in parameters.items()}
    _assert_nonzero_components(existing, lora=lora)
    model.eval()
    reference = None
    for chunks in (1, 2, 8):
        monkeypatch.setenv("DSV41_SEQ_CHUNKS", str(chunks))
        result = _forward_preserving_inputs(engine, _vision_batch(), partial(sft_loss, None), forward_only=True)
        log_probs = result["model_output"]["log_probs"].values()
        assert log_probs.numel() == 388 and not log_probs.requires_grad
        for name, parameter in parameters.items():
            assert torch.equal(parameter.main_grad, existing[name]), name
        loss = sum(result["loss"])
        if reference is None:
            reference = loss, log_probs.detach().clone()
        else:
            ref_loss, ref_log_probs = reference
            assert loss == pytest.approx(ref_loss, rel=3e-4, abs=3e-5)
            torch.testing.assert_close(log_probs, ref_log_probs, rtol=3e-4, atol=3e-5)
        chunk_training._record_result(
            {
                "case": "image_lora_inference" if lora else "image_inference",
                "existing_grad_origin": "sentinel",
                "chunks": chunks,
                "loss": loss,
                "log_probs_max_abs": (log_probs - reference[1]).abs().max().item(),
                "grad_max_abs": 0.0,
            }
        )
        dist.barrier()
