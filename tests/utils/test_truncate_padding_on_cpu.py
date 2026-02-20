# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Tests for the truncate_padding feature."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import FSDPActorConfig

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_forward_micro_batch_truncate_padding():
    """truncate_padding produces identical results while feeding shorter sequences to the model.

    Default batch: prompt_lengths=[10,8,12,6], response_lengths=[15,20,10,18],
    max_prompt=12, padded_response=20 => max_seq_len=32.
    left_padding per seq = [2,4,0,6], total_real = [25,28,22,24].
    After rolling, common right padding = 32 - max(25,28,22,24) = 4 => trimmed seqlen = 28.
    """
    _init_dist()
    model, model_config = _make_model()
    micro_batch = _make_micro_batch(vocab_size=model_config.vocab_size)

    captured_seqlens = {}

    class _CaptureSeqlen(nn.Module):
        def __init__(self, wrapped, label):
            super().__init__()
            self.wrapped = wrapped
            self.label = label

        def forward(self, **kwargs):
            captured_seqlens[self.label] = kwargs["input_ids"].shape[1]
            return self.wrapped(**kwargs)

    standard_actor = _make_actor(_CaptureSeqlen(model, "standard"), truncate_padding=False)
    truncate_actor = _make_actor(_CaptureSeqlen(model, "truncate"), truncate_padding=True)

    with torch.no_grad():
        standard_result = standard_actor._forward_micro_batch(
            micro_batch, temperature=1.0, calculate_entropy=True
        )
        truncate_result = truncate_actor._forward_micro_batch(
            micro_batch, temperature=1.0, calculate_entropy=True
        )

    # Verify the model received shorter sequences (32 -> 28 after trimming 4 common padding columns)
    assert captured_seqlens["standard"] == 32
    assert captured_seqlens["truncate"] == 28

    # Verify correctness on real response positions
    mask = micro_batch["response_mask"].bool()
    torch.testing.assert_close(truncate_result["log_probs"][mask], standard_result["log_probs"][mask])
    torch.testing.assert_close(truncate_result["entropys"][mask], standard_result["entropys"][mask])


def test_forward_micro_batch_with_sum_pi_squared():
    """_forward_micro_batch computes sum_pi_squared identically for both paths."""
    # Setup
    _init_dist()
    model, model_config = _make_model()
    micro_batch = _make_micro_batch(vocab_size=model_config.vocab_size)
    standard_actor = _make_actor(model, truncate_padding=False, calculate_sum_pi_squared=True)
    truncate_padding_actor = _make_actor(model, truncate_padding=True, calculate_sum_pi_squared=True)

    # Run
    with torch.no_grad():
        standard_result = standard_actor._forward_micro_batch(micro_batch, temperature=1.0)
        truncate_padding_result = truncate_padding_actor._forward_micro_batch(micro_batch, temperature=1.0)

    # Assert
    mask = micro_batch["response_mask"].bool()
    torch.testing.assert_close(
        truncate_padding_result["sum_pi_squared"][mask],
        standard_result["sum_pi_squared"][mask],
        msg="truncate_padding sum_pi_squared must match standard path on real response positions",
    )


def test_forward_micro_batch_fused_kernels():
    """_forward_micro_batch with use_fused_kernels produces identical results
    for both truncate_padding=True and truncate_padding=False."""
    # Setup
    _init_dist()
    model, model_config = _make_model()
    micro_batch = _make_micro_batch(vocab_size=model_config.vocab_size)

    fused_model = _FusedModelWrapper(model)
    standard_actor = _make_actor(fused_model, truncate_padding=False, use_fused_kernels=True)
    truncate_padding_actor = _make_actor(fused_model, truncate_padding=True, use_fused_kernels=True)

    # Run
    with torch.no_grad():
        standard_result = standard_actor._forward_micro_batch(micro_batch, temperature=1.0, calculate_entropy=True)
        truncate_padding_result = truncate_padding_actor._forward_micro_batch(
            micro_batch, temperature=1.0, calculate_entropy=True
        )

    # Assert
    mask = micro_batch["response_mask"].bool()
    torch.testing.assert_close(
        truncate_padding_result["log_probs"][mask],
        standard_result["log_probs"][mask],
        msg="fused + truncate_padding log_probs must match fused standard path",
    )
    torch.testing.assert_close(
        truncate_padding_result["entropys"][mask],
        standard_result["entropys"][mask],
        msg="fused + truncate_padding entropy must match fused standard path",
    )


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _init_dist():
    """Initialize a single-process gloo group if not already initialized."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29599")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _make_model() -> tuple[nn.Module, GPT2Config]:
    """Create a tiny GPT-2 model for fast CPU testing."""
    config = GPT2Config(vocab_size=100, n_embd=64, n_head=2, n_layer=1, n_positions=64)
    return GPT2LMHeadModel(config).eval(), config


def _make_micro_batch(
    vocab_size: int,
    max_prompt_length: int = 12,
    padded_response_length: int = 20,
    actual_prompt_lengths: list[int] = None,
    response_lengths: list[int] = None,
) -> dict[str, torch.Tensor]:
    """Create a micro-batch dict matching dp_actor's expected layout."""
    if actual_prompt_lengths is None:
        actual_prompt_lengths = [10, 8, 12, 6]
    if response_lengths is None:
        response_lengths = [15, 20, 10, 18]

    batch_size = len(actual_prompt_lengths)
    max_seq_len = max_prompt_length + padded_response_length

    input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    position_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    responses = torch.zeros(batch_size, padded_response_length, dtype=torch.long)
    response_mask = torch.zeros(batch_size, padded_response_length)

    for i in range(batch_size):
        left_pad = max_prompt_length - actual_prompt_lengths[i]
        total_real = actual_prompt_lengths[i] + response_lengths[i]
        tokens = torch.randint(1, vocab_size, (total_real,))
        input_ids[i, left_pad : left_pad + total_real] = tokens
        attention_mask[i, left_pad : max_prompt_length + response_lengths[i]] = 1
        position_ids[i, left_pad : max_prompt_length + response_lengths[i]] = torch.arange(total_real)
        responses[i, : response_lengths[i]] = tokens[actual_prompt_lengths[i] :]
        response_mask[i, : response_lengths[i]] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
        "response_mask": response_mask,
    }


class _FusedModelWrapper(nn.Module):
    """Wraps a standard model to simulate fused kernel output on CPU.

    Fused kernels return pre-computed per-token log_probs and entropy instead
    of raw logits. This wrapper computes those from the standard logits so we
    can exercise the fused code path without GPU-specific kernels.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        temperature = kwargs.pop("temperature", 1.0)
        kwargs.pop("return_dict", None)
        output = self.model(**kwargs)
        logits = output.logits / temperature
        log_probs_dist = F.log_softmax(logits, dim=-1)

        input_ids = kwargs["input_ids"]
        shifted_labels = input_ids[:, 1:]
        per_token_log_probs = log_probs_dist[:, :-1].gather(2, shifted_labels.unsqueeze(-1)).squeeze(-1)
        log_probs = F.pad(per_token_log_probs, (0, 1), value=0)

        entropy = -(log_probs_dist.exp() * log_probs_dist).sum(dim=-1)

        return SimpleNamespace(log_probs=log_probs, entropy=entropy)


def _make_actor(
    model: nn.Module,
    truncate_padding: bool = False,
    calculate_sum_pi_squared: bool = False,
    use_fused_kernels: bool = False,
) -> DataParallelPPOActor:
    """Create a DataParallelPPOActor with minimal config."""
    config = FSDPActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=4,
        use_dynamic_bsz=False,
        use_torch_compile=False,
        truncate_padding=truncate_padding,
        calculate_sum_pi_squared=calculate_sum_pi_squared,
        use_fused_kernels=use_fused_kernels,
    )
    return DataParallelPPOActor(config=config, actor_module=model)
