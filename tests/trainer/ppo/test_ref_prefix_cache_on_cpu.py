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
"""Tests for the ref-model cross-step prefix KV cache (native past_key_values).

``forward_ref_with_prefix_cache`` prefills a prompt once with ``use_cache=True``
and reuses the cached ``past_key_values`` for every response sharing that prompt
(and across steps, since the ref model is frozen). These tests assert the
resulting per-response log-probs match an independent (prompt + response)
forward, that cache hit == cache miss, and that ``forward_step`` routes to the
cache path only on a dedicated ref engine.
"""

from unittest import mock

import numpy as np
import pytest
import torch

# Some CI/local images ship torch_npu on a CPU host, where
# npu_cross_entropy_loss is unusable. Force the standard torch log-probs path so
# the test is device-agnostic; production code keeps the real dispatch.
import verl.utils.torch_functional as _verl_F

_verl_F.NPU_CROSS_ENTROPY_LOSS_AVAILABLE = False
_verl_F.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False

from transformers import AutoModelForCausalLM  # noqa: E402
from transformers.models.qwen2 import Qwen2Config  # noqa: E402

from verl.trainer.ppo.ref_prefix_cache import (  # noqa: E402
    RefPrefixKVCache,
    forward_ref_with_prefix_cache,
)
from verl.utils.torch_functional import logprobs_from_logits  # noqa: E402
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead  # noqa: E402

PAD_ID = 0
TOL = 1e-5


def _make_tiny_model():
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=32000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    return AutoModelForCausalLM.from_config(cfg).eval()


def _gt_log_probs(model, prompts, responses, response_mask, temperature=1.0):
    """Ground-truth log-probs from an independent (prompt + response) forward."""
    out = []
    plen = prompts.size(1)
    with torch.no_grad():
        for i in range(responses.size(0)):
            rlen = int(response_mask[i].sum())
            seq = torch.cat([prompts[i], responses[i, :rlen]]).unsqueeze(0)
            logits = model(input_ids=seq, attention_mask=torch.ones_like(seq), use_cache=False).logits
            if temperature != 1.0:
                logits = logits / temperature
            lp = logprobs_from_logits(logits[0, :-1], labels=seq[0, 1:])
            out.append(lp[plen - 1 : plen - 1 + rlen])
    return out


def _max_diff(shared_jagged, gt_list):
    return max((a - b).abs().max().item() for a, b in zip(shared_jagged.unbind(), gt_list, strict=False))


def _make_shared_prefix_micro_batch(prompts, responses, response_mask, uids, temperature=1.0):
    from tensordict import TensorDict

    import verl.utils.tensordict_utils as tu

    n = responses.size(0)
    td = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "temperature": torch.full((n,), float(temperature)),
        },
        batch_size=[n],
    )
    tu.assign_non_tensor(td, uid=np.array(uids, dtype=object))
    return td


def _make_engine_stub(model, ref_engine=True):
    """Bypass __init__; provide only what forward_step + the cache branch touch."""
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.module = model
    eng._autocast_dtype = torch.float32  # skip autocast
    eng.use_remove_padding = False
    model_config = type("mc", (), {})()
    model_config.use_ref_prefix_cache = True
    model_config.hf_config = type("hc", (), {"pad_token_id": PAD_ID})()
    eng.model_config = model_config
    eng.engine_config = type("ec", (), {"forward_only": ref_engine})()
    return eng


# --------------------------------------------------------------------------- #
# forward_ref_with_prefix_cache (standalone)
# --------------------------------------------------------------------------- #


def test_prefix_cache_correctness_miss_and_hit():
    model = _make_tiny_model()
    prompt = torch.tensor([[10, 11, 12, 13, 14]])
    prompts = prompt.repeat(2, 1)
    responses = torch.zeros(2, 4, dtype=torch.long)
    responses[0] = torch.tensor([20, 21, 22, 23])
    responses[1] = torch.tensor([30, 31, 32, 33])
    response_mask = responses.ne(PAD_ID)
    uids = ["a", "a"]
    gt = _gt_log_probs(model, prompts, responses, response_mask)

    cache = RefPrefixKVCache(max_entries=8)
    lp_miss = forward_ref_with_prefix_cache(model, prompts, responses, response_mask, uids, PAD_ID, cache)
    assert cache.stats()["miss_count"] == 1 and cache.stats()["hit_count"] == 0
    lp_hit = forward_ref_with_prefix_cache(model, prompts, responses, response_mask, uids, PAD_ID, cache)
    assert cache.stats()["hit_count"] == 1  # cross-step reuse

    assert _max_diff(lp_miss, gt) < TOL
    assert _max_diff(lp_hit, gt) < TOL
    assert _max_diff(lp_miss, lp_hit) < TOL  # hit == miss


def test_prefix_cache_multiple_groups_and_variable_length():
    model = _make_tiny_model()
    prompt_a = torch.tensor([[10, 11, 12, 13, 14]])
    prompt_b = torch.tensor([[100, 101, 102, 103, 104]])
    prompts = torch.cat([prompt_a.repeat(2, 1), prompt_b], 0)
    responses = torch.zeros(3, 4, dtype=torch.long)
    responses[0] = torch.tensor([20, 21, 22, 23])
    responses[1, :3] = torch.tensor([30, 31, 32])  # variable length
    responses[2] = torch.tensor([50, 51, 52, 53])
    response_mask = responses.ne(PAD_ID)
    uids = ["a", "a", "b"]
    gt = _gt_log_probs(model, prompts, responses, response_mask)

    cache = RefPrefixKVCache()
    lp = forward_ref_with_prefix_cache(model, prompts, responses, response_mask, uids, PAD_ID, cache)
    assert cache.stats()["miss_count"] == 2  # two distinct prompts
    # second call: both hit
    forward_ref_with_prefix_cache(model, prompts, responses, response_mask, uids, PAD_ID, cache)
    assert cache.stats()["hit_count"] == 2
    assert _max_diff(lp, gt) < TOL


def test_prefix_cache_with_temperature():
    model = _make_tiny_model()
    prompt = torch.tensor([[10, 11, 12, 13, 14]])
    prompts = prompt.repeat(2, 1)
    responses = torch.zeros(2, 4, dtype=torch.long)
    responses[0] = torch.tensor([20, 21, 22, 23])
    responses[1] = torch.tensor([30, 31, 32, 33])
    response_mask = responses.ne(PAD_ID)
    uids = ["a", "a"]
    temperature = 0.7
    gt = _gt_log_probs(model, prompts, responses, response_mask, temperature=temperature)

    cache = RefPrefixKVCache()
    lp = forward_ref_with_prefix_cache(
        model, prompts, responses, response_mask, uids, PAD_ID, cache, temperature=temperature
    )
    assert _max_diff(lp, gt) < TOL


# --------------------------------------------------------------------------- #
# forward_step branch selection
# --------------------------------------------------------------------------- #


def test_forward_step_takes_prefix_cache_path():
    model = _make_tiny_model()
    eng = _make_engine_stub(model)
    prompt = torch.tensor([[10, 11, 12, 13, 14]])
    prompts = prompt.repeat(2, 1)
    responses = torch.zeros(2, 4, dtype=torch.long)
    responses[0] = torch.tensor([20, 21, 22, 23])
    responses[1] = torch.tensor([30, 31, 32, 33])
    response_mask = responses.ne(PAD_ID)
    mb = _make_shared_prefix_micro_batch(prompts, responses, response_mask, ["a", "a"])

    with (
        mock.patch("verl.workers.engine.fsdp.transformer_impl.get_device_id", return_value="cpu"),
        mock.patch("verl.workers.engine.fsdp.transformer_impl.get_device_name", return_value="cpu"),
    ):
        loss, output = FSDPEngineWithLMHead.forward_step(eng, mb, loss_function=None, forward_only=True)

    lp = output["model_output"]["log_probs"]
    assert lp.is_nested
    gt = _gt_log_probs(model, prompts, responses, response_mask)
    assert _max_diff(lp, gt) < TOL
    # cache persists on the engine across forward_step calls
    assert eng._ref_prefix_kv_cache is not None


def test_forward_step_skips_prefix_path_on_actor_engine():
    """An actor engine (engine_config.forward_only=False) must NOT take the ref
    cache branch, even on a forward_only call."""
    model = _make_tiny_model()
    eng = _make_engine_stub(model, ref_engine=False)
    prompt = torch.tensor([[10, 11, 12, 13, 14]])
    prompts = prompt.repeat(2, 1)
    responses = torch.zeros(2, 4, dtype=torch.long)
    responses[0] = torch.tensor([20, 21, 22, 23])
    responses[1] = torch.tensor([30, 31, 32, 33])
    response_mask = responses.ne(PAD_ID)
    mb = _make_shared_prefix_micro_batch(prompts, responses, response_mask, ["a", "a"])

    # Spy: if the branch is wrongly entered on an actor engine, this fires.
    spy = mock.Mock(return_value=None)
    eng._forward_ref_prefix_cache = spy
    eng.prepare_model_inputs = lambda micro_batch: ({"input_ids": prompts[:, :1]}, {})
    eng.prepare_model_outputs = lambda **kw: {"log_probs": torch.zeros(1)}
    eng.get_data_parallel_group = lambda: None

    with (
        mock.patch("verl.workers.engine.fsdp.transformer_impl.get_device_id", return_value="cpu"),
        mock.patch("verl.workers.engine.fsdp.transformer_impl.get_device_name", return_value="cpu"),
    ):
        FSDPEngineWithLMHead.forward_step(eng, mb, loss_function=None, forward_only=True)

    assert not spy.called, "actor engine must not take the ref prefix-cache branch"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
