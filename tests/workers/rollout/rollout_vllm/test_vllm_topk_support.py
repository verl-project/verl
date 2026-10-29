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
"""Check provider evidence for complete and incomplete vLLM sampler heads.

Environment overrides:
  VLLM_TOPK_SUPPORT_MODEL_PATH - model (default Qwen2.5-0.5B-Instruct)
"""

import os

import numpy as np
import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.logprobs import FlatLogprobs, append_logprobs_for_next_position
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

from verl.workers.rollout.utils import extract_response_topk_logprobs_with_coverage

MODEL_PATH = os.path.expanduser(os.getenv("VLLM_TOPK_SUPPORT_MODEL_PATH", "~/models/Qwen/Qwen2.5-0.5B-Instruct"))


@pytest.fixture(scope="module")
def llm():
    k = 64
    return LLM(
        MODEL_PATH,
        logprobs_mode="processed_logprobs",
        max_logprobs=k,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        seed=0,
    )


@pytest.mark.parametrize("top_k,top_p", [(16, 1.0), (64, 0.9), (-1, 0.9), (-1, 1.0)])
def test_processed_topk_support_evidence(llm, top_k, top_p):
    k = 64
    params = SamplingParams(temperature=1.0, top_p=top_p, top_k=top_k, max_tokens=32, logprobs=k, flat_logprobs=True)
    outputs = llm.generate(["Write one short sentence about sampling.", "List three prime numbers."], params)
    complete_rows = 0
    for output in outputs:
        completion = output.outputs[0]
        sampled, ids, log_probs, complete = extract_response_topk_logprobs_with_coverage(
            completion.logprobs, k, vocab_size=llm.llm_engine.model_config.get_vocab_size()
        )
        in_support = np.isfinite(log_probs)
        # A partial head with all finite entries is unknown, even if its mass rounds to one.
        np.testing.assert_array_equal(complete, in_support.sum(-1) < k)
        complete_rows += int(complete.sum())
        np.testing.assert_allclose(np.exp(log_probs[complete]).sum(-1), 1.0, atol=1e-3)
        for t, token_id in enumerate(completion.token_ids):
            slots = np.nonzero((ids[t] == token_id) & in_support[t])[0]
            if complete[t]:
                assert slots.size == 1, f"sampled token {token_id} is outside a proven complete head at position {t}"
            if slots.size:
                np.testing.assert_allclose(sampled[t], log_probs[t, slots[0]], atol=1e-5)
    if top_k == 16:
        assert complete_rows > 0
    if top_k == -1 and top_p == 1.0:
        assert complete_rows == 0


@pytest.mark.parametrize("k,expected_complete", [(2, False), (4, True)])
def test_small_decode_batch_provider_support_coverage(k, expected_complete):
    # vLLM 0.29 retains both cutoff ties, while 0.32 keeps exactly top_k entries.
    # Either way, an all-finite partial head does not prove complete support.
    logits = torch.tensor([[2.0, 1.0, 1.0, 0.0]], device="cuda")
    logits = apply_top_k_top_p(logits, torch.tensor([2], device="cuda"), None)
    log_probs = logits.log_softmax(-1)
    support_ids = log_probs[0].isfinite().nonzero().flatten().tolist()
    assert len(support_ids) in (2, 3)
    head, ids = log_probs.topk(k, -1)
    # When 0.29 retains the cutoff tie, sample the omitted support member. Otherwise
    # sample from the recorded head; duplicated sampled/top-k entries are valid.
    omitted_support_ids = [token_id for token_id in support_ids if token_id not in ids[0].tolist()]
    sampled_id = omitted_support_ids[0] if omitted_support_ids else ids[0, 0].item()
    flat = FlatLogprobs()
    append_logprobs_for_next_position(
        flat,
        [sampled_id, *ids[0].tolist()],
        [log_probs[0, sampled_id].item(), *head[0].tolist()],
        [None] * (k + 1),
        2,
        k,
    )
    _, head_ids, _, complete = extract_response_topk_logprobs_with_coverage(flat, k, vocab_size=4)
    assert complete.tolist() == [expected_complete]
    if omitted_support_ids:
        assert not (head_ids == sampled_id).any()


@pytest.mark.parametrize("rule,threshold", [(None, 2.0), ("token", 2.0), ("token", "0.5_5.0")])
def test_pure_top_p_overflow_scores_sample_outside_recorded_head(rule, threshold):
    from verl.trainer.ppo.score_centering import score_centering_auto_correction, score_centering_weight_fn

    # Pure top-p has no cardinality cap: this real provider keeps all four tokens,
    # while the recording budget is two. Sample an actual nonzero-probability tail token.
    sampler_logits = apply_top_k_top_p(torch.zeros(1, 4, device="cuda"), None, torch.tensor([0.9], device="cuda"))
    sampler_lp = sampler_logits.log_softmax(-1)
    assert int(sampler_lp.isfinite().sum()) == 4
    q_head, ids = sampler_lp.topk(2, -1)
    sampled_id = next(i for i in range(4) if i not in ids[0].tolist())
    flat = FlatLogprobs()
    append_logprobs_for_next_position(
        flat,
        [sampled_id, *ids[0].tolist()],
        [sampler_lp[0, sampled_id].item(), *q_head[0].tolist()],
        [None] * 3,
        4,
        2,
    )
    q_sample, _, recorded, complete = extract_response_topk_logprobs_with_coverage(flat, 2, vocab_size=4)
    assert complete.tolist() == [False]
    trainer = torch.tensor([[2.0, 1.0, 0.0, -1.0]], device="cuda", requires_grad=True)
    lp = trainer.log_softmax(-1)
    weight = score_centering_weight_fn(rule, threshold)
    correction, shift, mass, _ = score_centering_auto_correction(
        lp.gather(-1, ids),
        torch.as_tensor(recorded, device="cuda"),
        weight,
        torch.as_tensor(complete, device="cuda"),
    )
    assert mass.item() == pytest.approx(0.5)
    assert shift.item() == 0
    sampled = lp[:, sampled_id] - shift
    actual_q_sample = torch.as_tensor(q_sample, device="cuda")
    sampled_weight = weight((sampled.detach() - actual_q_sample).exp())
    loss = -(sampled_weight * sampled - correction).mean()
    assert torch.isfinite(torch.autograd.grad(loss, trainer)[0]).all()
