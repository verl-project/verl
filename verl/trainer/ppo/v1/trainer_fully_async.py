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
"""Streaming (``fully_async``) PPO trainer for V1.

Builds on ``separate_async`` (standalone rollout + checkpoint-engine weight sync). The only
behavioral difference is *who drives prompt feeding*: instead of ``step()`` feeding exactly one
batch per training step, an autonomous background feeder thread continuously streams prompts into
TransferQueue (bounded by a staleness / in-flight budget), while ``step()`` only samples + trains.
This decouples production rate from consumption rate so rollout overlaps training.

The whole feeder (thread loop, throttling, abnormal-rollout discard, weight-sync pause/resume)
lives here and touches the base trainer only through its public state (``train_dataloader``,
``agent_loop_manager``, ``global_steps``, ``replay_buffer``). ``trainer_base`` is left exactly as
upstream: the base ``step()`` still calls ``self._add_batch_to_generate()`` unconditionally, and we
make that a no-op once the feeder owns generation (see :meth:`_add_batch_to_generate`).
"""

import logging
import os
import threading
import uuid

import numpy as np
import transfer_queue as tq
from omegaconf import DictConfig

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.utils import tensordict_utils as tu

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def compute_max_inflight_prompts(staleness_threshold: float, parameter_sync_step: int, train_batch_size: int) -> int:
    """Compute the in-flight prompt budget for the streaming feeder.

    The budget caps how many un-consumed prompts may exist in TransferQueue at once, bounding how
    far the rollouter runs ahead of training. Mirrors the ``fully_async_policy`` reference formula
    with ``require_batches=1``.
    """
    budget = int((1 + staleness_threshold) * parameter_sync_step * train_batch_size)
    assert budget >= train_batch_size, (
        f"in-flight budget ({budget}) must be >= train_batch_size ({train_batch_size}); "
        f"check staleness_threshold/parameter_sync_step"
    )
    return budget


@register_trainer("fully_async")
class PPOTrainerFullyAsync(PPOTrainerSeparateAsync):
    """Streaming asynchronous PPO trainer (autonomous background feeder + standalone rollout)."""

    def __init__(self, config: DictConfig):
        super().__init__(config)  # inherits separate_async asserts + bypass_mode
        fa = self.config.trainer.v1.fully_async
        self._poll_interval = fa.feeder_poll_interval
        self._budget = compute_max_inflight_prompts(
            fa.staleness_threshold, fa.parameter_sync_step, self.config.data.train_batch_size
        )
        # parameter version used to tag fed prompts; tracks self.global_steps (see on_step_end).
        self._param_version = 0
        self._param_version_lock = threading.Lock()
        # Serializes the (non-thread-safe) train dataloader iterator between the feeder thread and
        # the main thread's checkpoint save. Owned here so trainer_base needs no change.
        self._dataloader_lock = threading.Lock()
        # Feeder thread + signals. self._feeder is None until on_train_begin starts it; that None
        # check is also what makes warmup feed but step()'s feed call a no-op (see below).
        self._feeder: threading.Thread | None = None
        self._feeder_stop = threading.Event()
        self._feeder_paused = threading.Event()  # set -> loop does not dispatch new prompts
        self._feeder_error = False

    # ------------------------------------------------------------------ feeding

    def _add_batch_to_generate(self):
        """Override the base per-step feed.

        Before the feeder starts (``on_train_begin`` warmup) this feeds one batch so the pipeline
        is primed; once the feeder owns generation it is a no-op, so the base ``step()`` calling
        this unconditionally does not double-feed. This is what lets ``trainer_base`` stay unchanged.
        """
        if self._feeder is None:
            self._feed_one_batch(self.global_steps)

    def _feed_one_batch(self, global_steps: int):
        """Pull one batch from the dataloader, tag its prompts, register them in TransferQueue, and
        dispatch generation. Tags carry both ``status`` (read by the prompt-level buffer) and ``n``
        (read by the rollout-level :class:`SessionReplayBuffer`), so either buffer works."""
        with self._dataloader_lock:
            try:
                if self.train_dataloader_it is None:
                    self.train_dataloader_it = iter(self.train_dataloader)
                batch_dict = next(self.train_dataloader_it)
            except StopIteration:
                self.train_dataloader_it = iter(self.train_dataloader)
                batch_dict = next(self.train_dataloader_it)

        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        batch = tu.get_tensordict(batch_dict)
        tu.assign_non_tensor_data(batch, "global_steps", global_steps)

        n_sessions = int(self.config.actor_rollout_ref.rollout.n)
        tags = [{"is_prompt": True, "status": "pending", "global_steps": global_steps, "n": n_sessions}] * len(batch)
        tq.kv_batch_put(keys=list(batch["uid"]), partition_id="train", tags=tags)

        self.agent_loop_manager.generate_sequences(batch)

    def _discard_dead_prompts(self) -> int:
        """Discard prompts whose every rollout failed (no usable trajectory) so a fresh prompt can
        take their in-flight slot. No-op unless the rollout-level buffer is in use (the base buffer's
        :meth:`dead_prompt_keys` returns ``[]``). Reuses the metadata just synced by the preceding
        ``count_inflight`` call in the feeder loop."""
        keys = self.replay_buffer.dead_prompt_keys("train")
        if not keys:
            return 0
        tq.kv_clear(partition_id="train", keys=keys)
        # one prompt key per discarded prompt (the rest are its session markers)
        return sum(1 for k in keys if "_sess" not in k)

    # ------------------------------------------------------------------ feeder thread

    def _current_param_version(self) -> int:
        with self._param_version_lock:
            return self._param_version

    def _feeder_loop(self):
        while not self._feeder_stop.is_set():
            if self._feeder_paused.is_set():
                # paused (e.g. during a weight sync): do not dispatch; generation already in flight
                # keeps running and is aborted+continued by the checkpoint engine.
                self._feeder_stop.wait(self._poll_interval)
                continue
            try:
                counts = self.replay_buffer.count_inflight("train")
                # Discard all-failed prompts (uses the state count_inflight just synced); each frees
                # its in-flight slot so the budget check below refills it with a fresh prompt.
                n = self._discard_dead_prompts()
                if n:
                    logger.info("Streaming feeder: discarded %d all-failed prompt(s)", n)
                # Bucket names are buffer-specific; the budget bounds total un-consumed prompts.
                inflight = sum(counts.values())
                if inflight < self._budget:
                    self._feed_one_batch(self._current_param_version())
                else:
                    self._feeder_stop.wait(self._poll_interval)  # interruptible sleep, no busy-wait
            except StopIteration:
                logger.info("Streaming feeder: dataset exhausted, stopping feeder")
                break
            except Exception:
                logger.exception("Streaming feeder thread crashed")
                self._feeder_error = True
                break

    def _start_feeder(self):
        self._feeder_stop.clear()
        self._feeder_paused.clear()
        self._feeder_error = False
        self._feeder = threading.Thread(target=self._feeder_loop, name="streaming-rollout-feeder", daemon=True)
        self._feeder.start()

    def _stop_feeder(self, timeout: float = 30.0):
        self._feeder_stop.set()
        if self._feeder is not None and self._feeder.is_alive():
            self._feeder.join(timeout=timeout)

    # ------------------------------------------------------------------ lifecycle

    def on_train_begin(self):
        # separate_async: prime the pipeline with num_warmup_batches (feeder is None -> warmup feeds
        # via _add_batch_to_generate). Then start the autonomous feeder.
        super().on_train_begin()
        with self._param_version_lock:
            self._param_version = self.global_steps
        self._start_feeder()
        # print (not logger.info): verl configures no logging handler in the trainer actor, so INFO
        # is swallowed. These lifecycle markers must reach stdout to be observable/asserted.
        print(f"Streaming rollout feeder started; budget={self._budget} prompts", flush=True)

    def step(self, metrics, timing_raw):
        # fail fast instead of hanging in replay_buffer.sample if the feeder died
        if self._feeder_error:
            raise RuntimeError("Streaming feeder thread died; aborting training")
        return super().step(metrics, timing_raw)

    def on_step_end(self):
        # Pause the feeder around the periodic standalone weight sync so it does not dispatch prompts
        # into a server that is mid-sync. Generation already in flight is aborted+continued by the
        # checkpoint engine / FullyAsyncLLMServerClient (partial rollout), independent of this pause.
        is_sync_step = self._feeder is not None and self.global_steps % self.parameter_sync_step == 0
        if is_sync_step:
            print(f"Pausing streaming feeder for weight sync at step {self.global_steps}", flush=True)
            self._feeder_paused.set()
        try:
            super().on_step_end()  # separate_async: standalone update_weights on sync steps
        finally:
            if is_sync_step:
                self._feeder_paused.clear()
                print(f"Resumed streaming feeder after weight sync at step {self.global_steps}", flush=True)
        with self._param_version_lock:
            self._param_version = self.global_steps

    def _save_checkpoint(self):
        # The feeder thread may be iterating the (non-thread-safe) dataloader; serialize against it
        # so the base's StatefulDataLoader.state_dict() is never read mid-iteration.
        with self._dataloader_lock:
            super()._save_checkpoint()

    def on_train_end(self):
        self._stop_feeder()
        super().on_train_end()
