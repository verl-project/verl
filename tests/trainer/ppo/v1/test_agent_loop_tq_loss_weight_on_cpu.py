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

import asyncio
import logging

import pytest
import torch

tq = pytest.importorskip("transfer_queue")

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput  # noqa: E402
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ  # noqa: E402


@pytest.mark.parametrize("explicit_weights", [True, False])
def test_multi_output_postprocess_persists_loss_weights_and_broadcasts_reward(monkeypatch, explicit_weights):
    captured = {}

    async def fake_kv_batch_put(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(tq, "async_kv_batch_put", fake_kv_batch_put)
    # The implicit-1/N warning is memoized process-wide; keep tests order-independent.
    from verl.trainer.ppo.v1 import agent_loop_tq

    agent_loop_tq._warn_unweighted_multi_output_once.cache_clear()

    class WorkerStub:
        async def _compute_score(self, outputs, kwargs):
            outputs[-1].reward_score = 3.0
            outputs[-1].extra_fields["reward_extra_info"] = {"success": True}

        async def _compute_teacher_logprobs(self, output, **kwargs):
            return None

        def _compute_multi_modal_inputs(self, output, input_ids):
            return {}

        def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs):
            return torch.arange(input_ids.shape[-1]).unsqueeze(0)

    outputs = [
        AgentLoopOutput(
            prompt_ids=[1],
            response_ids=[2, 3],
            response_mask=[1, 1],
            metrics=AgentLoopMetrics(),
        ),
        AgentLoopOutput(
            prompt_ids=[1, 2],
            response_ids=[3],
            response_mask=[1],
            metrics=AgentLoopMetrics(),
        ),
    ]
    if explicit_weights:
        for output in outputs:
            output.loss_weight = 0.25

    worker_class = AgentLoopWorkerTQ.__ray_metadata__.modified_class
    asyncio.run(
        worker_class._agent_loop_postprocess(
            WorkerStub(),
            outputs,
            validate=False,
            uid="task",
            session_id=0,
            global_steps=1,
            raw_prompt=[],
        )
    )

    assert captured["keys"] == ["task_0_0", "task_0_1"]
    # Without an explicit weight the adapter stays neutral (1.0) rather than guessing 1/N,
    # which would silently shrink the trajectory under the default token-mean mode.
    expected_loss_weight = 0.25 if explicit_weights else 1.0
    torch.testing.assert_close(captured["fields"]["loss_weight"], torch.tensor([expected_loss_weight] * 2))
    torch.testing.assert_close(
        captured["fields"]["rm_scores"].to_padded_tensor(0.0),
        torch.tensor([[0.0, 3.0], [3.0, 0.0]]),
    )
    assert captured["fields"]["extra_fields"][0]["reward_extra_info"] == {"success": True}


def test_unweighted_multi_output_is_logged_once(caplog):
    """Segments stored without a weight must surface the loss_agg_mode caveat once."""
    from verl.trainer.ppo.v1 import agent_loop_tq

    # The warning is memoized per segment count, so clear it to make the test hermetic.
    agent_loop_tq._warn_unweighted_multi_output_once.cache_clear()

    with caplog.at_level(logging.WARNING, logger=agent_loop_tq.__name__):
        agent_loop_tq._warn_unweighted_multi_output_once(3)
        agent_loop_tq._warn_unweighted_multi_output_once(3)

    warnings = [record for record in caplog.records if "loss_weight" in record.getMessage()]
    assert len(warnings) == 1, "the neutral-weight notice should be reported once per segment count"
    message = warnings[0].getMessage()
    # The caveat only bites under seq-mean-token-mean, so name both the mode and the fix.
    assert "seq-mean-token-mean" in message
    assert "1/3" in message

    agent_loop_tq._warn_unweighted_multi_output_once.cache_clear()
