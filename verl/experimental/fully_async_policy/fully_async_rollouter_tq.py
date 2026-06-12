# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import os
import uuid
from collections import defaultdict

import numpy as np
import ray
import torch

from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    safe_create_task,
)
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter,
)
from verl.experimental.fully_async_policy.replay_buffer import tq_kv_clear
from verl.trainer.main_ppo_sync import AgentLoopWorkerTQ
from verl.utils import tensordict_utils as tu
from verl.utils.profiler import marked_timer

try:
    import transfer_queue as tq
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.8` and try again.")
    from verl.utils.transferqueue_utils import tq

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FullyAsyncAgentLoopManagerTQ(FullyAsyncAgentLoopManager):
    """Agent loop manager that uses :class:`AgentLoopWorkerTQ` workers.

    Key difference from base :class:`FullyAsyncAgentLoopManager`:
    - Overrides ``generate_sequences_single`` to convert ``DataProto`` → ``TensorDict``
      before dispatching to workers, aligning with :class:`AgentLoopWorkerTQ`'s
      ``generate_sequences`` signature which expects ``TensorDict`` (not ``DataProto``).
    - Default ``wait=True``: waits for all tasks to complete before returning.
      This ensures Rollouter knows exactly when generation finishes (avoids deadlock).
    """

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(AgentLoopWorkerTQ)
        super().__init__(*args, **kwargs)

    async def generate_sequences_single(self, prompts):
        """Convert DataProto to TensorDict, then dispatch to agent loop worker.

        Aligns with main_ppo_sync.py PPOTrainer.step() which calls::

            batch = tu.get_tensordict(batch_dict)
            self.async_rollout_manager.generate_sequences(batch)

        Args:
            prompts: Input batch (DataProto or TensorDict).

        Returns:
            None — data is written directly to TransferQueue by the worker.
        """
        worker = self._select_best_worker()
        return await worker.generate_sequences.remote(prompts, wait=True)

    async def generate_sequences(self, prompts):
        """
        Dispatch input batch to agent loop workers with sync wait. Workers put agent loop outputs
        into TransferQueue once an agent loop finished, and this method returns only after
        all workers have completed their work.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        return await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk, wait=True)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )


class FullyAsyncRollouterTQ(FullyAsyncRollouter):
    """
    FullyAsyncRollouter variant that uses TransferQueue + ReplayBuffer instead of MessageQueue.

    Core design:
    - ReplayBuffer.acquire_slot() in _feed_samples() provides source-level flow control
      (replaces _should_pause_generation + MessageQueue.queue_size backpressure)
    - Generated samples are written to TQ (zero-copy) instead of MessageQueue (pickle)
    - FullyAsyncAgentLoopManager is unchanged — still used for actual generation
    """

    def __init__(
        self,
        config,
        tokenizer,
        processor=None,
        device_name=None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, processor=processor, device_name=device_name)

        # ==================== TQ-specific overrides ====================
        self.replay_buffer = None  # Ray Actor handle, set via set_replay_buffer()
        self.agent_loop_manager_class = FullyAsyncAgentLoopManagerTQ
        print("[TQFullyAsyncRollouter] initialized (TQ mode)")

    async def set_replay_buffer(self, replay_buffer):
        """Set ReplayBuffer Ray Actor handle."""
        async with self.lock:
            self.replay_buffer = replay_buffer

            tq.init()
            print("[TQFullyAsyncRollouter] TQ initialized in Rollouter actor process", flush=True)

    async def _feed_samples(self):
        """Feed samples from dataloader to pending_queue, with source-level flow control.

        Key difference from base class: acquire_slot() is called BEFORE putting
        to pending_queue. This blocks the dataloader when too many samples are
        in-flight, replacing the need for _should_pause_generation().

        Alignment with main_ppo_sync.py PPOTrainer.step():1740-1758:
            - Do NOT repeat(n) here — let AgentLoopWorkerTQ._run_prompt loop n times
              via __rollout_n__ field (avoids double-repeat bug).
            - Set uid/__rollout_n__/sample_id/global_steps in batch_dict (plain dict)
              BEFORE calling tu.get_tensordict(), so these np.array values become NonTensorStack
              (supporting per-index access in AgentLoopWorkerTQ.generate_sequences).
        """
        continuous_iterator = self._create_continuous_iterator()
        rollout_n = self.config.actor_rollout_ref.rollout.n

        for epoch, batch_dict in continuous_iterator:
            sample_id = f"sample_{epoch}_{self.global_steps}"
            acquired = await self.replay_buffer.acquire_slot.remote(timeout=None, uid=sample_id)
            if not acquired:
                print(
                    f"[TQFullyAsyncRollouter][Feed] ReplayBuffer finished or closed, "
                    f"stop feeding after {self.global_steps} samples"
                )
                break

            # Inject fields into batch_dict (plain dict) BEFORE tu.get_tensordict().
            # All np.array values become NonTensorStack via get_tensordict:424,
            # supporting per-index access in AgentLoopWorkerTQ.generate_sequences().
            batch_dict["uid"] = np.array([sample_id], dtype=object)
            batch_dict["__rollout_n__"] = np.full(1, rollout_n, dtype=np.int64)
            batch_dict["sample_id"] = np.array([sample_id], dtype=object)
            batch_dict["global_steps"] = np.full(1, self.global_steps, dtype=np.int64)

            # Convert to TensorDict (np.array values → NonTensorStack via get_tensordict:424)
            full_batch = tu.get_tensordict(batch_dict)

            # Set agent_name for non-multi-turn mode (same as prepare_single_generation_data)
            if not self.config.actor_rollout_ref.rollout.multi_turn.enable:
                batch_dict["agent_name"] = np.array(["single_turn_agent"], dtype=object)
                full_batch = tu.get_tensordict(batch_dict)

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                sample_id=sample_id,
                epoch=epoch,
                rollout_status={},
            )

            await self.pending_queue.put(rollout_sample)

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[TQFullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples: "
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.pending_queue.put(None)
        print(f"[TQFullyAsyncRollouter][Feed] Sample addition is complete, {self.global_steps} samples have been added")

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample: generate via ALM worker (which writes to TQ blocking), then release slot.

        Simplified from base class:
        - Base class: generate → put to MessageQueue
        - TQ path: generate + TQ write both happen INSIDE AgentLoopWorkerTQ
          (via overridden _agent_loop_postprocess)
          → we just call it and release the slot

        Note: full_batch now has bsz=1 (no repeat(n)), and contains __rollout_n__.
              AgentLoopWorkerTQ._run_prompt will loop n times internally to produce
              n responses per prompt, each written to TQ with a unique key.
        """
        try:
            await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
            self.total_generated_samples += 1
        except Exception as e:
            logger.exception(f"[TQFullyAsyncRollouter] Failed to process {rollout_sample.sample_id}: {e}")
        finally:
            # Always release the slot regardless of success/failure
            await self.replay_buffer.release_slot.remote()

        self.processed_sample_count += 1

    async def _streaming_generation_main(self):
        """Main entry for stream processing."""
        # Start feed and processor tasks (same as base class)
        print("[TQFullyAsyncRollouter] Starting feed_task...", flush=True)
        self.feed_task = safe_create_task(self._feed_samples(), name="feed_task")
        print("[TQFullyAsyncRollouter] Starting processor_task...", flush=True)
        self.processor_task = safe_create_task(self._processor_worker(), name="processor_task")
        try:
            done, pending = await asyncio.wait(
                [self.feed_task, self.processor_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.exception():
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor task exited prematurely")

            print("[TQFullyAsyncRollouter] Sample feed completed")
            await self.processor_task
            print("[TQFullyAsyncRollouter] Streaming process completed")
            await self.pending_queue.join()
            print("[TQFullyAsyncRollouter] pending_queue joined")

        except Exception as e:
            print(f"[TQFullyAsyncRollouter] Streaming process exception: {e}")
            raise e
        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)

            if self.processor_task and not self.processor_task.done():
                self.processor_task.cancel()
                await asyncio.gather(self.processor_task, return_exceptions=True)

            self.feed_task = None
            self.processor_task = None

            await self.replay_buffer.signal_finish.remote()

            async with self.lock:
                self.running = False

    async def do_validate(self):
        """Run validation and return metrics.

        Overrides FullyAsyncRollouter.do_validate() to use PPOTrainer's
        _validate() which operates on TransferQueue + ReplayBuffer (TQ path)
        instead of SeparateRayPPOTrainer's _validate() which uses MessageQueue.
        """
        timing_raw = {}
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = await self._validate()
        return timing_raw | val_metrics

    async def _validate(self) -> dict[str, float]:
        # Lists to collect samples for the table
        sample_uids = []
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        data_sources = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        dump_all_inputs: list[str] = []
        dump_all_outputs: list[str] = []
        dump_all_keys: list[str] = []
        session_to_sample_idx: dict[str, int] = {}

        for batch_dict in self.val_dataloader:
            # 1. put batch to agent loop manager
            batch_dict["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
            )
            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            await self.async_rollout_manager.generate_sequences(batch)

            # 2. sample batch from replay buffer
            sampled = await self.replay_buffer.sample.remote(partition_id="val", sample_size=len(batch))

            # Unpack list[tuple[str, dict]] into KVBatchMeta format
            from verl.utils.transferqueue_utils import KVBatchMeta

            keys = [k for k, _ in sampled]
            tags = [meta for _, meta in sampled]
            batch = KVBatchMeta(partition_id="val", keys=keys, tags=tags)

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                self.checkpoint_manager.sleep_replicas()
                batch = self._compute_reward_colocate(batch)
                self.checkpoint_manager.update_weights()

            # 4. collect necessary data for logging
            # For multi-output agent loops, only use the final output per session for metrics.
            # Keys have format {uid}_{session_id}_{index}; keep only the highest index per session.
            session_max: dict[str, tuple[int, int]] = {}  # session_key -> (max_index, position)
            for pos, key in enumerate(batch.keys):
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    session_key = f"{parts[0]}_{parts[1]}"
                    index = int(parts[2])
                    if session_key not in session_max or index > session_max[session_key][0]:
                        session_max[session_key] = (index, pos)
                else:
                    session_max[key] = (0, pos)
            sorted_sessions = sorted(session_max.items(), key=lambda x: x[1][1])
            final_indices = [pos for _, (_, pos) in sorted_sessions]
            final_keys = [batch.keys[i] for i in final_indices]
            base_offset = len(sample_scores)
            session_to_sample_idx.update(
                {session_key: base_offset + j for j, (session_key, _) in enumerate(sorted_sessions)}
            )

            text_data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses"]
            )
            text_data["prompts"] = text_data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            text_data["responses"] = text_data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            all_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["prompts"]]
            all_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["responses"]]

            fields = ["uid", "rm_scores", "num_turns", "reward_model", "data_source", "extra_fields"]

            data = tq.kv_batch_get(keys=final_keys, partition_id=batch.partition_id, select_fields=fields)

            from tensordict import NonTensorData, NonTensorStack

            # Extract values handling NonTensorStack/NonTensorData (aligned with main_ppo_sync.py:334-337)
            def _extract(v):
                if isinstance(v, torch.Tensor):
                    return v
                elif isinstance(v, NonTensorStack):
                    # NonTensorStack may contain non-unique values; use .tolist()
                    return v.tolist()
                elif isinstance(v, NonTensorData):
                    return v.data
                return v

            _uid = _extract(data.pop("uid"))
            _rm_scores = _extract(data["rm_scores"])
            _num_turns = _extract(data.pop("num_turns"))

            # Ensure plain Python types for lists (aligned with main_ppo_sync.py:932-937)
            sample_uids.extend(_uid.tolist() if hasattr(_uid, "tolist") else list(_uid))
            sample_outputs.extend(all_outputs[i] for i in final_indices)
            sample_inputs.extend(all_inputs[i] for i in final_indices)
            scores = _rm_scores.sum(dim=1).tolist()
            sample_scores.extend(scores)
            # np.concatenate requires array-like elements, not scalars (compatible with ray_trainer.py:660)
            _turns = _num_turns.tolist() if hasattr(_num_turns, "tolist") else list(_num_turns)
            sample_turns.extend([np.array([t]) for t in _turns])
            reward_extra_infos_dict["reward"].extend(scores)

            extra_fields_list = data.pop("extra_fields", None)
            if extra_fields_list is not None:
                n_prior = len(reward_extra_infos_dict["reward"]) - len(extra_fields_list.tolist())
                for extra_field in extra_fields_list.tolist():
                    reward_extra_info = (
                        extra_field.get("reward_extra_info", {}) if isinstance(extra_field, dict) else {}
                    )
                    for key in reward_extra_infos_dict:
                        if key != "reward" and key not in reward_extra_info:
                            reward_extra_infos_dict[key].append(None)
                    for key, value in reward_extra_info.items():
                        if key not in reward_extra_infos_dict:
                            reward_extra_infos_dict[key] = [None] * n_prior
                        reward_extra_infos_dict[key].append(value)
                    n_prior += 1

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                sample_gts.extend([item.get("ground_truth", None) for item in reward_model.tolist()])
            else:
                sample_gts.extend([None] * len(final_indices))

            data_source = data.pop("data_source", None)
            if data_source is not None:
                data_sources.extend(data_source.tolist())
            else:
                data_sources.extend(["unknown"] * len(final_indices))

            dump_all_inputs.extend(all_inputs)
            dump_all_outputs.extend(all_outputs)
            dump_all_keys.extend(batch.keys)

            # 5. cleanup transfer queue and replay buffer
            tq_kv_clear(batch)

        # logger to wandb
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump to local dir
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Sort according to uid (so that generations in the same rollout are together)
            sort_keys = []
            for key in dump_all_keys:
                parts = key.rsplit("_", 2)
                sort_keys.append((parts[0], int(parts[1]), int(parts[2])) if len(parts) == 3 else (key, 0, 0))
            sorted_indices = sorted(range(len(dump_all_keys)), key=lambda i: sort_keys[i])
            dump_all_inputs = [dump_all_inputs[i] for i in sorted_indices]
            dump_all_outputs = [dump_all_outputs[i] for i in sorted_indices]
            dump_all_keys = [dump_all_keys[i] for i in sorted_indices]

            # For ground truths, scores and reward extra infos, find the values in the
            # lists for the final samples of each session
            dump_all_sessions = [
                f"{parts[0]}_{parts[1]}" if len(parts) == 3 else key
                for key in dump_all_keys
                for parts in [key.rsplit("_", 2)]
            ]
            session_final_indices = [session_to_sample_idx[session] for session in dump_all_sessions]
            self._dump_generations(
                inputs=dump_all_inputs,
                outputs=dump_all_outputs,
                gts=[sample_gts[i] for i in session_final_indices],
                scores=[sample_scores[i] for i in session_final_indices],
                reward_extra_infos_dict={
                    k: [v[i] for i in session_final_indices] for k, v in reward_extra_infos_dict.items()
                }
                | {"uid": dump_all_keys},
                dump_path=val_data_dir,
            )

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    async def reset_staleness(self):
        """Reset version window after parameter update."""
        rb_timing = await self.replay_buffer.reset_staleness.remote()
        return rb_timing

    async def _should_pause_generation(self) -> bool:
        return False

    async def _async_monitor_loop(self):
        pass
