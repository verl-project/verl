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
from collections.abc import Mapping

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


@register("naive")
class NaiveRewardManager(RewardManagerBase):
    """The reward manager."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    def _get_terminal_tool_names(self) -> set[str]:
        multi_turn = self.config.actor_rollout_ref.rollout.multi_turn
        if isinstance(multi_turn, Mapping):
            terminal_tool_names = multi_turn.get("terminal_tool_names", [])
        else:
            terminal_tool_names = getattr(multi_turn, "terminal_tool_names", [])
        return set(terminal_tool_names or [])

    def _display_answers_responses(self, extra_info: dict) -> list[str] | None:
        """Return terminal display_answers responses, or None to decode normally."""
        if "display_answers" not in self._get_terminal_tool_names():
            return None

        tool_selection = extra_info.get("tool_selection")
        if tool_selection is not None and "display_answers" not in tool_selection:
            return None

        terminal_tool_arguments = extra_info.get("terminal_tool_arguments")
        if terminal_tool_arguments is None:
            return [""] if tool_selection is not None else None
        if not isinstance(terminal_tool_arguments, dict):
            return [""]

        answers = terminal_tool_arguments.get("answers")
        if isinstance(answers, list):
            return [str(answer) for answer in answers] or [""]
        if isinstance(answers, str):
            return [answers]
        return [""]

    async def _compute_score_for_response(
        self,
        data_source,
        response_str,
        ground_truth,
        extra_info,
        extra_reward_kwargs,
    ):
        if self.is_async_reward_score:
            return await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        return await self.loop.run_in_executor(
            None,
            lambda: self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            ),
        )

    @staticmethod
    def _score_from_result(result) -> float:
        if isinstance(result, dict):
            return result["score"]
        return result

    def _aggregate_results(self, results: list):
        if len(results) == 1:
            return results[0]

        return min(results, key=self._score_from_result)

    async def run_single(self, data: DataProto) -> dict:
        data = data[
            -1:
        ]  # for multi-sequence outputs, we only compute reward based on the last sequence
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][
            -response_length:
        ].sum()
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

        response_strs = self._display_answers_responses(extra_info)
        if response_strs is None:
            response_str = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=True
                ),
            )
            response_strs = [response_str]

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        results = [
            await self._compute_score_for_response(
                data_source=data_source,
                response_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                extra_reward_kwargs=extra_reward_kwargs,
            )
            for response_str in response_strs
        ]
        result = self._aggregate_results(results)

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
