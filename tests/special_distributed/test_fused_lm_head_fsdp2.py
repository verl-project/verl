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

"""Run with torchrun --standalone --nproc-per-node=2 -m pytest -sv this_file.

Exercise the real model, fused kernels and independently sharded FSDP2 head.
References use the original HF logits forward and the same fused kernel with
the head owned by the root FSDP unit, both under the same mixed precision policy.
No pretrained weights or dataset downloads are needed.
"""

import os
from datetime import timedelta
from types import MethodType

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
from transformers import (
    Glm4vConfig,
    Glm4vForConditionalGeneration,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from verl.models.transformers.monkey_patch import patch_forward_with_backends
from verl.utils.fsdp_utils import apply_fsdp2


@pytest.fixture(scope="module", autouse=True)
def distributed():
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("requires torchrun with at least two CUDA devices")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    yield
    dist.destroy_process_group()


@pytest.mark.parametrize("reference_kind", ["logits", "fused"])
@pytest.mark.parametrize("model_type", ["qwen3", "llama", "qwen2_vl", "qwen3_vl", "qwen3_5", "glm4v"])
@pytest.mark.parametrize("backend", ["torch", "triton", "liger"])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_fused_head_matches_logits(model_type, backend, offload, tied, reference_kind, monkeypatch):
    if backend == "liger":
        pytest.importorskip("liger_kernel")
    config_cls, model_cls = {
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen2_vl": (Qwen2VLConfig, Qwen2VLForConditionalGeneration),
        "qwen3_vl": (Qwen3VLConfig, Qwen3VLForConditionalGeneration),
        "qwen3_5": (Qwen3_5Config, Qwen3_5ForConditionalGeneration),
        "glm4v": (Glm4vConfig, Glm4vForConditionalGeneration),
    }[model_type]
    text_config = dict(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        tie_word_embeddings=tied,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    multimodal = model_type not in ("qwen3", "llama")
    if multimodal:
        text_config.update(
            layer_types=["full_attention"],
            rope_parameters=dict(
                rope_type="default", rope_theta=10000.0, mrope_section=[4, 6, 6], partial_rotary_factor=1.0
            ),
        )
        vision_config = dict(
            depth=1,
            embed_dim=32,
            hidden_size=128 if model_type == "qwen2_vl" else 32,
            intermediate_size=64,
            out_hidden_size=128,
            num_heads=4,
            deepstack_visual_indexes=[],
            patch_size=2,
            image_size=8,
            num_position_embeddings=16,
        )
        config = config_cls(
            text_config=text_config, vision_config=vision_config, tie_word_embeddings=tied, image_token_id=200
        )
    else:
        config = config_cls(**text_config)
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = model_cls(config).cuda()
    reference = model_cls(config).cuda()
    reference.load_state_dict(model.state_dict())
    # Match actor setups that freeze the vision tower but train the language model.
    for candidate in (model, reference):
        for name, param in candidate.named_parameters():
            if "visual" in name:
                param.requires_grad_(False)
    original_forward = model_cls.forward
    # The backend patch changes the HF class. Keep the reference's original
    # entry point and restore the class between parameterized cases.
    monkeypatch.setattr(model_cls, "forward", original_forward)
    patch_forward_with_backends(model, True, backend)
    if reference_kind == "logits":
        reference.forward = MethodType(original_forward, reference)
    else:
        reference._verl_fused_kernels_backend = backend
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    kwargs = dict(
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        reshard_after_forward=True,
    )
    if offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()

    def shard_model(candidate):
        if multimodal:
            # Exercise each adapter's real forward with an independent head.
            # Keep vision in the root: some HF vision embeddings access their
            # weight directly and cannot be independently sharded by FSDP2.
            if not tied:
                fully_shard(candidate.lm_head, **kwargs)
            fully_shard(candidate, **kwargs)
        else:
            apply_fsdp2(candidate, kwargs, {})

    shard_model(model)
    if reference_kind == "logits":
        shard_model(reference)
    else:
        # Identical fused arithmetic, with the head owned by the root FSDP
        # unit. This isolates head lifecycle from kernel rounding differences.
        fully_shard(reference, **kwargs)
    assert isinstance(model.lm_head, FSDPModule) is (not tied)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
    for step in range(2):
        optim.zero_grad(set_to_none=True)
        ref_optim.zero_grad(set_to_none=True)
        for microbatch in range(2):
            sync = microbatch == 1
            model.set_requires_gradient_sync(sync)
            reference.set_requires_gradient_sync(sync)
            inputs = (torch.arange(64, device="cuda").view(2, 32) + 17 * dist.get_rank() + microbatch + step) % 256
            if multimodal:
                inputs[:, 0] = config.image_token_id
            labels = torch.roll(inputs, -1, -1)
            position_ids = torch.arange(32, device="cuda").expand(2, -1)
            if multimodal:
                position_ids = position_ids.unsqueeze(0).expand(4, -1, -1)
            forward_kwargs = dict(input_ids=inputs, position_ids=position_ids, return_dict=True)
            if multimodal:
                # One merged image token per sample. Calling the vision tower
                # also avoids unused FSDP groups during gradient accumulation.
                forward_kwargs.update(
                    pixel_values=torch.zeros(8, 24, device="cuda"),
                    image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]], device="cuda"),
                )
            temperature = 0.7
            output = model(**forward_kwargs, shift_labels=labels, temperature=temperature)
            if reference_kind == "logits":
                logits = (reference(**forward_kwargs).logits / temperature).float()
                ref_log_probs = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                ref_entropy = logits.logsumexp(-1) - (logits.softmax(-1) * logits).sum(-1)
            else:
                ref_output = reference(**forward_kwargs, shift_labels=labels, temperature=temperature)
                ref_log_probs, ref_entropy = ref_output.log_probs, ref_output.entropy
            # Triton's existing contract flattens token dimensions.
            torch.testing.assert_close(output.log_probs.reshape_as(ref_log_probs), ref_log_probs, atol=2e-2, rtol=2e-3)
            torch.testing.assert_close(
                output.entropy.reshape_as(ref_entropy).float(), ref_entropy.float(), atol=2e-2, rtol=2e-3
            )
            loss = (-output.log_probs.mean() + 0.01 * output.entropy.mean()) / 2
            ref_loss = (-ref_log_probs.mean() + 0.01 * ref_entropy.mean()) / 2
            loss.backward()
            ref_loss.backward()
        for (name, param), (ref_name, ref_param) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == ref_name
            if not param.requires_grad:
                assert param.grad is None and ref_param.grad is None, name
                continue
            assert param.grad is not None, name
            grad, ref_grad = param.grad.to_local(), ref_param.grad.to_local()
            if reference_kind == "logits":
                # Triton accumulates/scales logits in FP32; the ordinary HF
                # path rounds the projection and temperature division to BF16.
                torch.testing.assert_close(
                    grad, ref_grad, atol=1e-3, rtol=3e-2, msg=lambda message, name=name: f"{name}: {message}"
                )
                relative_error = (grad - ref_grad).norm() / ref_grad.norm().clamp_min(1e-8)
                assert relative_error < 0.03, (name, relative_error)
            else:
                torch.testing.assert_close(
                    grad, ref_grad, atol=1e-6, rtol=1e-5, msg=lambda message, name=name: f"{name}: {message}"
                )
        optim.step()
        ref_optim.step()
        for (name, param), ref_param in zip(model.named_parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(
                param.to_local(),
                ref_param.to_local(),
                # The logits reference can accumulate the accepted BF16
                # gradient error over SGD steps (lr=0.1).
                atol=(step + 1) * 1e-4 if reference_kind == "logits" else 1e-6,
                rtol=3e-2 if reference_kind == "logits" else 1e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )
    model.eval()
    with torch.no_grad():
        output = model(**forward_kwargs, temperature=temperature)
        assert torch.isfinite(output.log_probs).all()
        assert torch.isfinite(output.entropy).all()
