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

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ


@pytest.fixture
def tokenizer():
    # Zero is an ordinary token, as it is for Qwen; ignoring special tokens must
    # not hide a mask that accidentally includes zero-filled response padding.
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"LEAK": 0, "prompt": 1, "answer": 2, "<pad>": 3, "<eos>": 4})),
        pad_token="<pad>",
        eos_token="<eos>",
    )


def _outputs(lengths):
    return [
        AgentLoopOutput(
            prompt_ids=[1] * prompt_len,
            response_ids=[2] * response_len,
            response_mask=[1] * response_len,
            num_turns=2,
            metrics=AgentLoopMetrics(),
        )
        for prompt_len, response_len in lengths
    ]


async def _score(monkeypatch, tokenizer, outputs, manager_cls=NaiveRewardManager, compute_score=None):
    config = OmegaConf.create(
        {
            "reward": {
                "custom_reward_function": {"path": "test_reward.py"},
                "reward_kwargs": {
                    "max_resp_len": 7,
                    "overlong_buffer_cfg": {"enable": True, "len": 4, "penalty_factor": 1.0, "log": True},
                },
            }
        }
    )
    if compute_score is None:

        def compute_score(*, solution_str, ground_truth, **kwargs):
            return {"score": float(solution_str == ground_truth), "solution_str": solution_str}

    reward_worker = RewardLoopWorker.__new__(RewardLoopWorker)
    reward_worker.config = config
    reward_worker.reward_manager = manager_cls(config, tokenizer, compute_score)
    captured = {}

    async def remote(data):
        captured["data"] = data
        return await reward_worker.compute_score(data)

    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    worker.processor = None
    worker.tokenizer = tokenizer
    worker.mm_processor_kwargs = {}
    worker.reward_loop_worker_handles = [SimpleNamespace(compute_score=SimpleNamespace(remote=remote))]
    worker.distillation_enabled = False
    worker.rollout_config = OmegaConf.create({"topk_log_probs": 0})

    async def put_outputs(**kwargs):
        captured["stored"] = kwargs

    monkeypatch.setattr("verl.trainer.ppo.v1.agent_loop_tq.tq.async_kv_batch_put", put_outputs)
    await AgentLoopWorkerTQ.__ray_metadata__.modified_class._agent_loop_postprocess(
        worker,
        outputs,
        validate=False,
        uid="prompt0",
        session_id=0,
        global_steps=1,
        data_source="test",
        reward_model={"ground_truth": tokenizer.decode(outputs[-1].response_ids)},
        raw_prompt=[{"role": "user", "content": "prompt"}],
    )
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lengths",
    [
        [(3, 5), (9, 2)],  # A longer final prompt must not expose response padding.
        [(10, 2), (1, 6)],  # A compacted final prompt must not truncate the answer.
        [(3, 2)],
        [(3, 2), (3, 2)],
        [(3, 5), (3, 2)],
    ],
)
async def test_async_reward_decodes_only_the_final_response(monkeypatch, tokenizer, lengths):
    outputs = _outputs(lengths)
    await _score(monkeypatch, tokenizer, outputs)

    assert outputs[-1].extra_fields["reward_extra_info"]["solution_str"] == tokenizer.decode(outputs[-1].response_ids)
    assert all(output.reward_score == 1.0 for output in outputs)


@pytest.mark.asyncio
async def test_async_dapo_penalty_uses_response_length(monkeypatch, tokenizer):
    outputs = _outputs([(3, 5), (9, 2)])
    await _score(monkeypatch, tokenizer, outputs, DAPORewardManager, lambda **kwargs: 1.0)

    assert outputs[-1].extra_fields["reward_extra_info"]["overlong_reward"] == 0.0
    assert outputs[-1].reward_score == 1.0


@pytest.mark.asyncio
async def test_async_reward_batch_preserves_prompt_response_boundaries(monkeypatch, tokenizer):
    outputs = _outputs([(3, 5), (9, 2)])
    captured = await _score(monkeypatch, tokenizer, outputs)
    batch = captured["data"].batch
    prompt_width = batch["prompts"].size(1)

    torch.testing.assert_close(batch["input_ids"], torch.cat([batch["prompts"], batch["responses"]], dim=1))
    for i, output in enumerate(outputs):
        prompt_mask = batch["attention_mask"][i, :prompt_width].bool()
        response_mask = batch["attention_mask"][i, prompt_width:].bool()
        assert batch["prompts"][i][prompt_mask].tolist() == output.prompt_ids
        assert batch["responses"][i][response_mask].tolist() == output.response_ids
        valid = batch["attention_mask"][i].bool()
        assert batch["position_ids"][i][valid].tolist() == list(
            range(len(output.prompt_ids) + len(output.response_ids))
        )
    assert captured["data"].non_tensor_batch["prompt_len"].tolist() == [3, 9]
    assert captured["data"].non_tensor_batch["response_len"].tolist() == [5, 2]
    assert captured["stored"]["keys"] == ["prompt0_0_0", "prompt0_0_1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("lengths", [[(3, 5), (9, 2)], [(10, 2), (1, 6)]])
async def test_async_disrm_preprocessing_uses_the_same_response(monkeypatch, tokenizer, lengths):
    outputs = _outputs(lengths)
    captured = await _score(monkeypatch, tokenizer, outputs)
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    worker = RewardLoopWorker.__new__(RewardLoopWorker)
    worker.input_tokenizer = tokenizer
    worker.reward_model_tokenizer = tokenizer

    prompt = await worker._preprocess_reward_inputs(captured["data"][-1:])

    assert prompt == "prompt" + tokenizer.decode(outputs[-1].response_ids)


@pytest.mark.asyncio
async def test_async_reward_keeps_observations_and_real_zero_tokens(monkeypatch, tokenizer):
    outputs = _outputs([(3, 5), (9, 3)])
    outputs[-1].response_ids = [2, 0, 2]
    outputs[-1].response_mask = [1, 0, 1]
    await _score(monkeypatch, tokenizer, outputs)

    # Reward functions still see the full response, including tool observations.
    assert outputs[-1].extra_fields["reward_extra_info"]["solution_str"] == "answer LEAK answer"
    assert outputs[-1].reward_score == 1.0


@pytest.mark.asyncio
async def test_async_reward_preserves_precomputed_scores(monkeypatch, tokenizer):
    outputs = _outputs([(3, 5), (9, 2)])
    outputs[-1].reward_score = 0.75
    outputs[-1].extra_fields["reward_extra_info"] = {"acc": 0.75}
    captured = await _score(monkeypatch, tokenizer, outputs)

    assert "data" not in captured
    assert all(output.reward_score == 0.75 for output in outputs)


@pytest.mark.asyncio
async def test_async_reward_errors_propagate(monkeypatch, tokenizer):
    def fail(**kwargs):
        raise ValueError("reward failed")

    with pytest.raises(ValueError, match="reward failed"):
        await _score(monkeypatch, tokenizer, _outputs([(3, 5), (9, 2)]), compute_score=fail)
