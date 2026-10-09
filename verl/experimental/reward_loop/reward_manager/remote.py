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

import inspect
import itertools

from verl import DataProto
from verl.runtime import ClassWithInitArgs, Worker, current_runtime
from verl.utils.reward_score import default_compute_score

from .base import RewardManagerBase
from .registry import register


class RewardComputeWorker(Worker):
    """
    WARNING: This class cannot have async methods.
    """

    def __init__(self, compute_score_fn):
        super().__init__()
        # since the reward function may not be pickleable, we need to init it in the worker
        self.compute_score_fn = compute_score_fn

    def compute_score(self, **kwargs) -> dict:
        return self.compute_score_fn(**kwargs)


@register("remote")
class RemoteRewardManager(RewardManagerBase):
    """
    The reward manager.
    Some errors exist when using default thread pool to compute reward score, e.g., math-verify.
    https://github.com/verl-project/verl/issues/3407
    To avoid the above issues, we use a separate process to compute reward score.
    Moreover, process may be more suitable for cpu-intensive requests.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        assert not self.is_async_reward_score, "Async reward score is not supported in remote reward manager. "
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        runtime = current_runtime()
        host_pool = runtime.current_host_resource_pool()
        compute_pool = runtime.create_resource_pool(
            nnodes=1,
            processes_per_node=config.reward.num_workers,
            device_type="cpu",
            on=host_pool,
        )
        self._worker_group = runtime.create_worker_group(
            ClassWithInitArgs(RewardComputeWorker, *(self.compute_score,)),
            on=compute_pool,
        )
        remote_group = self._worker_group.remote()
        self.reward_worker_pool = itertools.cycle(remote_group.rank(rank) for rank in range(remote_group.world_size))

    def choose_reward_worker(self):
        return next(self.reward_worker_pool)

    async def run_single(self, data: DataProto) -> dict:
        data = data[-1:]  # for multi-sequence outputs, we only compute reward based on the last sequence
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        reward_worker = self.choose_reward_worker()
        result = await reward_worker.execute_rank_zero_async(
            "compute_score",
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **extra_reward_kwargs,
        )

        reward_extra_info = {}
        if isinstance(result, dict):
            score = result["score"]
            reward_extra_info.update(result)
        else:
            score = result
            reward_extra_info["acc"] = score

        return {"reward_score": score, "reward_extra_info": reward_extra_info}

    def close(self):
        self._worker_group.close()
