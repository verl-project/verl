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

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


@register("batch")
class BatchRewardManager(RewardManagerBase):
    """Batch-friendly reward manager: forwards the whole chunk to a single
    ``compute_score`` call with plural kwargs (``data_sources`` /
    ``solution_strs`` / ``ground_truths`` / ``extra_infos``).

    Field naming is aligned with ``verl/workers/reward_manager/batch.py`` so
    users have a single batch contract across both codepaths.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    @staticmethod
    def _extract_sample(data_item):
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        # Top-level shallow copy is required for batch correctness: agent_loop broadcasts
        # `extra_info` via `np.array([v] * n)`, so every sample in the chunk shares the same
        # dict reference. Without this copy, per-sample injection of `num_turns` /
        # `rollout_reward_scores` overwrites earlier samples within the same batch.
        extra_info = dict(data_item.non_tensor_batch.get("extra_info") or {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        return valid_response_ids, data_source, ground_truth, extra_info

    async def run_batch(self, data: DataProto) -> list[dict]:
        valid_ids_list: list = []
        data_sources: list = []
        ground_truths: list = []
        extra_infos: list = []
        for i in range(len(data)):
            valid_ids, data_source, ground_truth, extra_info = self._extract_sample(data[i])
            valid_ids_list.append(valid_ids)
            data_sources.append(data_source)
            ground_truths.append(ground_truth)
            extra_infos.append(extra_info)

        solution_strs = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.batch_decode(valid_ids_list, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        if self.is_async_reward_score:
            result = await self.compute_score(
                data_sources=data_sources,
                solution_strs=solution_strs,
                ground_truths=ground_truths,
                extra_infos=extra_infos,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_sources=data_sources,
                    solution_strs=solution_strs,
                    ground_truths=ground_truths,
                    extra_infos=extra_infos,
                    **extra_reward_kwargs,
                ),
            )

        outputs: list[dict] = []
        for item in result:
            reward_extra_info: dict = {}
            if isinstance(item, dict):
                score = item["score"]
                for key, value in item.items():
                    reward_extra_info[key] = value
            else:
                score = item
                reward_extra_info["acc"] = score
            outputs.append({"reward_score": score, "reward_extra_info": reward_extra_info})
        return outputs

    async def run_single(self, data: DataProto) -> dict:
        # Preserve `naive.run_single`'s multi-sequence fallback: take only the last row.
        return (await self.run_batch(data[-1:]))[0]
