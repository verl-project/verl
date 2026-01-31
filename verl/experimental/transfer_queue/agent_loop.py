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
import asyncio

import numpy as np
import ray
from transfer_queue import BatchMeta

import verl.experimental.agent_loop.agent_loop as agent_loop


class AgentLoopManager(agent_loop.AgentLoopManager):
    def generate_sequences(self, prompts: BatchMeta) -> BatchMeta:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (BatchMeta): Input batch.

        Returns:
            BatchMeta: Output batch metadata.
        """

        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.wake_up()

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = BatchMeta.concat(outputs)
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.sleep()

        # calculate performance metrics
        metrics = [output.extra_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.set_extra_info("timing", timing)
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: BatchMeta) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # TODO (TQ): initialize tq during init when enable TQ switch is stable
        tq_client = self._create_transferqueue_client()
        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = asyncio.run(tq_client.async_get_data(output[slowest]))["attention_mask"]
        prompt_length = output.samples[0].fields["prompts"].shape[0]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def create_transferqueue_client_for_workers(self):
        # TODO (TQ): initialize tq during worker init when enable TQ switch is stable
        ray.get([worker.create_transferqueue_client.remote() for worker in self.agent_loop_workers])

    def _create_transferqueue_client(self):
        """Create a client for data system (TransferQueue)."""
        from verl.single_controller.ray.base import get_random_string
        from verl.utils.transferqueue_utils import create_transferqueue_client

        client_name = get_random_string(length=6)

        tq_client = create_transferqueue_client(
            client_id=f"AgentLoopManager_{client_name}",
            config=self.config.transfer_queue,
        )

        return tq_client
