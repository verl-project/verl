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

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""

import asyncio
import logging
import os
from typing import Any

import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

# Config flag (default off) selecting rollout-level dispatch. When False the legacy
# prompt-pinned chunk dispatch + prompt-status readiness are used, byte-for-byte as before.
ROLLOUT_LEVEL_DISPATCH_KEY = "trainer.v1.rollout_level_dispatch"

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


def build_trajectory_info(step, index, validate) -> list[dict]:
    """Synchronous port of ``agent_loop.get_trajectory_info`` (pure CPU, no I/O).

    Lives here so the manager can build trajectory info inline while fanning out rollout
    units, without spinning up an event loop just to await the async original.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


def extract_sample(batch: TensorDict, i: int) -> dict:
    """Extract sample ``i`` from a batched TensorDict into a plain per-prompt dict.

    Moved out of the worker so the manager can build per-prompt dicts before dispatching
    individual rollout units. Mirrors the original per-key type handling.
    """
    sample: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            sample[k] = v[i]
        elif isinstance(v, NonTensorStack):
            sample[k] = v[i].data
        elif isinstance(v, NonTensorData):
            sample[k] = v.data
        else:
            logger.exception(f"Unsupported type {type(v)} for key {k}")
    return sample


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    """Rollout-level agent loop worker.

    Unlike the legacy design, a worker is no longer pinned to a whole prompt and its ``n``
    GRPO sessions. The manager dispatches **one rollout (one session of one prompt) at a
    time** via :meth:`run_rollout`; sibling sessions of the same prompt run on other workers.
    A long-tail rollout therefore only occupies a single slot here and never blocks the rest.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()
        # Optional per-worker cap on concurrently-executing rollouts (the event-loop / GIL
        # pressure knob). 0 = unbounded, preserving the legacy concurrency level. Read from the
        # raw config (never strict-instantiated), so it works without a dataclass field.
        try:
            self._rollout_cap = int(self.config.trainer.v1.get("max_concurrent_rollouts_per_worker", 0) or 0)
        except Exception:
            self._rollout_cap = 0
        # Created lazily on the actor's own running event loop (asyncio.Semaphore binds to the
        # loop active at construction time, which is not available in __init__).
        self._sem: asyncio.Semaphore | None = None

    def _semaphore(self) -> asyncio.Semaphore | None:
        if self._rollout_cap <= 0:
            return None
        if self._sem is None:
            # Single-threaded event loop: no await between check and set, so no race.
            self._sem = asyncio.Semaphore(self._rollout_cap)
        return self._sem

    async def run_rollout(self, prompt: dict, sampling_params: dict, trajectory: dict, session_id: int) -> None:
        """Schedule a single rollout as a background task and return immediately.

        The fast return lets the manager's dispatch loop ack all units without blocking on
        rollout execution, while still surfacing a dead worker to the manager (the ``.remote``
        call fails). ``prompt`` must already have ``__rollout_n__`` / ``__do_sample__`` removed
        by the manager; ``agent_name`` stays in it (consumed by ``_run_agent_loop``).
        """
        task = asyncio.create_task(self._execute_rollout(prompt, sampling_params, trajectory, session_id))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _execute_rollout(self, prompt: dict, sampling_params: dict, trajectory: dict, session_id: int) -> None:
        uid = prompt["uid"]
        partition_id = "train" if not trajectory["validate"] else "val"
        sem = self._semaphore()
        status = "success"
        try:
            if sem is not None:
                async with sem:
                    await self._run_agent_loop(
                        sampling_params, trajectory=trajectory, trace=False, session_id=session_id, **prompt
                    )
            else:
                await self._run_agent_loop(
                    sampling_params, trajectory=trajectory, trace=False, session_id=session_id, **prompt
                )
        except Exception as e:
            logger.exception(f"Error in rollout {uid}_sess{session_id}: {e}")
            status = "failure"

        # Per-session completion marker. Written AFTER _run_agent_loop (whose postprocess has
        # already put the {uid}_{session_id}_* data keys into TransferQueue), so "marker present"
        # implies "this session's data is present" for the replay buffer's session-counting
        # readiness. A failed session still writes a marker so its GRPO group can complete
        # (matching the legacy behavior where a failed session marked the whole prompt done).
        await tq.async_kv_put(
            key=f"{uid}_sess{session_id}",
            partition_id=partition_id,
            tag={
                "is_session": True,
                "session_id": session_id,
                "status": status,
                "global_steps": trajectory["step"],
            },
        )

    # ---- Legacy prompt-pinned path (used when rollout_level_dispatch=False) ----
    async def generate_sequences(self, batch: TensorDict) -> None:
        """Legacy: spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Legacy: spawn multiple agent loops in parallel according to rollout.n / val_kwargs.n,
        and track GRPO-group status via the prompt-status tag (running -> finished/failure)."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params, trajectory=trajectory, trace=trace, session_id=i, **prompt
                    )
                )
                tasks.append(task)
            await asyncio.gather(*tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        # TODO: Support output:list[AgentLoopOutput]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    # These tags are used for off-policy staleness control, if a trajectory
                    # spans too many global steps, we need to filter it out.
                    # global_steps: which global steps this sample is from dataloader
                    "global_steps": kwargs["global_steps"],
                    # min_global_steps: start generation model weights version of this trajectory
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    # max_global_steps: end generation model weights version of this trajectory
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)
        # Round-robin cursor for rollout-level dispatch; persists across batches so prompts
        # fed back-to-back keep spreading evenly across the worker pool.
        self._dispatch_rr = 0
        # Opt-in: rollout-level dispatch. Default False keeps the legacy prompt-pinned chunk path.
        self._rollout_level_dispatch = bool(OmegaConf.select(self.config, ROLLOUT_LEVEL_DISPATCH_KEY, default=False))

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """Rollout-level dispatch: fan each prompt's ``n`` GRPO sessions out across the worker
        pool round-robin, decoupling a prompt from any single worker.

        Each ``(prompt, session_id)`` is sent as its own :meth:`AgentLoopWorkerTQ.run_rollout`
        unit, so sibling sessions of one prompt land on different workers and a long-tail
        rollout never blocks the rest. Returns once all workers have ack'd dispatch (each
        schedules a background task and returns) without waiting for rollout completion.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        if not self._rollout_level_dispatch:
            # Legacy prompt-pinned path: chunk the batch across workers; each worker fans out a
            # whole prompt's n sessions internally and tracks GRPO-group status tags.
            chunkes = prompts.chunk(len(self.agent_loop_workers))
            ray.get(
                [
                    worker.generate_sequences.remote(chunk)
                    for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
                ]
            )
            return

        validate = bool(prompts["validate"]) if "validate" in prompts else False
        prompts.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout

        base_sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        if validate:
            base_sampling_params["top_p"] = config.val_kwargs.top_p
            base_sampling_params["top_k"] = config.val_kwargs.top_k
            base_sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in prompts:
            prompts["agent_name"] = NonTensorData(config.agent.default_agent_loop)

        trajectory_info = build_trajectory_info(prompts["global_steps"], prompts["index"], validate)

        num_workers = len(self.agent_loop_workers)
        futures = []
        for i in range(len(prompts)):
            prompt = extract_sample(prompts, i)
            # NOTE: user can dynamically adjust n per sample here (e.g. by task difficulty);
            # the prompt's TransferQueue tag carries the same n for session-counting readiness.
            n = prompt.pop("__rollout_n__", config.n if not validate else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)
            sampling_params = dict(base_sampling_params)
            if not validate and not do_sample:
                apply_greedy_sampling_params(sampling_params)

            # TODO(perf): a prompt is currently re-serialized once per session; batch sessions
            # by target worker to send it at most once per worker when n > num_workers.
            for session_id in range(int(n)):
                worker = self.agent_loop_workers[self._dispatch_rr % num_workers]
                self._dispatch_rr += 1
                futures.append(worker.run_rollout.remote(prompt, sampling_params, trajectory_info[i], session_id))

        # Wait only for the fast dispatch acks (surfaces a dead worker; does not block on rollout
        # execution, which proceeds as background tasks on the workers).
        ray.get(futures)
