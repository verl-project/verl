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

"""Check teacher temperature at the vLLM worker's full-vocabulary logits boundary."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import ray
import torch

pytest.importorskip("vllm")

from vllm.v1.sample.sampler import Sampler

from verl.workers.config import RolloutConfig
from verl.workers.rollout.vllm_rollout import utils, vllm_async_server


class FakeModel:
    def compute_logits(self, hidden_states):
        return hidden_states.clone()


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
def test_teacher_logprobs_match_student_temperature(temperature, monkeypatch):
    teacher = FakeModel()
    worker = SimpleNamespace(_iter_all_models=lambda: iter([teacher]))
    monkeypatch.setattr(utils, "patch_vllm_moe_model_weight_loader", lambda model: None)
    utils.vLLMColocateWorkerExtension.monkey_patch_model(
        worker, vocab_size=4, banned_token_ids=[1], logits_temperature=temperature
    )
    logits = torch.tensor([[0.0, 0.8, -1.0, 2.1, 8.0], [-1.3, 0.0, 2.0, 1.0, 10.0]])
    # vLLM's prompt path normalizes compute_logits() directly, without sampling temperature.
    teacher_logprobs = Sampler.compute_logprobs(teacher.compute_logits(logits))

    student_logits = logits[:, :4].clone()
    student_logits[:, 1] = -torch.inf
    student_logprobs = (student_logits / temperature).log_softmax(-1)
    torch.testing.assert_close(teacher_logprobs[:, :4], student_logprobs)
    assert torch.isneginf(teacher_logprobs[:, [1, 4]]).all()
    torch.testing.assert_close(teacher_logprobs.exp().sum(-1), torch.ones(2))

    # Both teacher top-k scores and an observed token outside top-k use full-vocabulary normalization.
    values, ids = teacher_logprobs.topk(2, dim=-1)
    expected_values, expected_ids = student_logprobs.topk(2, dim=-1)
    torch.testing.assert_close(values, expected_values)
    assert torch.equal(ids, expected_ids)
    observed_ids = torch.tensor([[2], [0]])
    assert not (ids == observed_ids).any()
    torch.testing.assert_close(teacher_logprobs.gather(-1, observed_ids), student_logprobs.gather(-1, observed_ids))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_teacher_scores_at_small_temperature(dtype):
    """Temperature scaling must not overflow or round logits in the model dtype."""
    model = FakeModel()
    temperature = 0.0001
    logits = torch.tensor([[20.0, 19.0, -20.0, 80.0, 100.0]], dtype=dtype)
    utils.monkey_patch_compute_logits(model, vocab_size=4, banned_token_ids=[3], logits_temperature=temperature)
    logprobs = Sampler.compute_logprobs(model.compute_logits(logits))

    assert torch.isfinite(logprobs[:, :3]).all()
    assert torch.isneginf(logprobs[:, 3:]).all()
    expected = (logits[:, :3].float() / temperature).log_softmax(-1)
    torch.testing.assert_close(logprobs[:, :3], expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_default_temperature_preserves_rollout_logits(dtype):
    model = FakeModel()
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=dtype)
    utils.monkey_patch_compute_logits(model, vocab_size=4)
    output = model.compute_logits(logits)
    assert output.dtype == dtype
    assert torch.equal(output[:, :4], logits[:, :4])
    assert torch.isneginf(output[:, 4]).all()


@pytest.mark.parametrize("role", ["teacher", "actor", "reward"])
def test_only_teacher_servers_receive_logits_temperature(role, monkeypatch):
    replica = vllm_async_server.vLLMReplica(
        replica_rank=0,
        config=RolloutConfig(name="vllm", temperature=0.7, tensor_model_parallel_size=1),
        model_config=SimpleNamespace(),
        is_teacher_model=role == "teacher",
        is_reward_model=role == "reward",
    )
    worker_info = (ray.NodeID.from_random().hex(), "0")
    replica.workers = [SimpleNamespace(__ray_call__=SimpleNamespace(remote=AsyncMock(return_value=worker_info)))]
    server = MagicMock()
    server.get_master_address.remote = AsyncMock(return_value=("127.0.0.1", 1234, 1235))
    server.get_server_address.remote = AsyncMock(return_value=("127.0.0.1", 8000))
    server.launch_server.remote = AsyncMock()
    replica.server_class = MagicMock()
    replica.server_class.options.return_value.remote.return_value = server
    monkeypatch.setattr(replica, "_get_server_name_prefix", lambda: "test_")
    monkeypatch.setattr(
        vllm_async_server,
        "get_platform",
        lambda: SimpleNamespace(ray_noset_envvars=lambda: [], rollout_env_vars=lambda: {}),
    )

    asyncio.run(replica.launch_servers())

    kwargs = replica.server_class.options.return_value.remote.call_args.kwargs
    assert kwargs["logits_temperature"] == (0.7 if role == "teacher" else 1.0)
