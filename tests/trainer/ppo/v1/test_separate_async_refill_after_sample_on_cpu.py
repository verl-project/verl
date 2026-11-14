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
"""CPU tests for ``trainer.v1.separate_async.refill_after_sample``.

With the option on, the next step's batch is submitted right after the last mini-batch of the current step is
sampled (``on_sample_end``) instead of in the next ``prepare_step``. The number of batches submitted per step
is unchanged, and no batch is submitted after the final step's sample.
"""

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode, PPOTrainerSeparateAsync


def _stub(*, refill_after_sample: bool, parameter_sync_step: int = 2, total_training_steps: int = 3):
    trainer = PPOTrainerSeparateAsync.__new__(PPOTrainerSeparateAsync)
    trainer.config = OmegaConf.create({"data": {"train_batch_size": 4}})
    trainer.trainer_mode = "separate_async"
    trainer.current_mode = HybridEngineMode.TRAINER
    trainer.hybrid_rollout_config = OmegaConf.create({"enable_switch": False})
    trainer.parameter_sync_step = parameter_sync_step
    trainer.total_training_steps = total_training_steps
    trainer._refill_after_sample = refill_after_sample
    trainer._next_batch_submitted = False
    trainer._step_sample_wait_seconds = 0.0
    trainer.events = []
    trainer._add_batch_to_generate = lambda: trainer.events.append(("submit", trainer.global_steps, trainer.where))
    return trainer


def _run(trainer):
    """Mimic PPOTrainer.fit()/step() hook order for every global step."""
    for step in range(1, trainer.total_training_steps + 1):
        trainer.global_steps = step
        trainer.where = "prepare_step"
        trainer.prepare_step()
        for trigger_idx in range(trainer.parameter_sync_step):
            trainer.local_trigger_step = trigger_idx
            trainer.where = f"sample_end[{trigger_idx}]"
            trainer.on_sample_begin()
            trainer.on_sample_end()
    return trainer.events


def test_default_submits_in_prepare_step():
    events = _run(_stub(refill_after_sample=False))
    assert events == [("submit", 1, "prepare_step"), ("submit", 2, "prepare_step"), ("submit", 3, "prepare_step")]


def test_refill_after_sample_submits_after_last_mini_batch():
    events = _run(_stub(refill_after_sample=True))
    # Step 1 still submits in prepare_step (nothing was prefetched); later steps are submitted one step early,
    # right after the last mini-batch is sampled; the final step submits nothing extra.
    assert events == [("submit", 1, "prepare_step"), ("submit", 1, "sample_end[1]"), ("submit", 2, "sample_end[1]")]


def test_refill_after_sample_keeps_batch_count():
    for parameter_sync_step in (1, 4):
        default = _run(_stub(refill_after_sample=False, parameter_sync_step=parameter_sync_step))
        refill = _run(_stub(refill_after_sample=True, parameter_sync_step=parameter_sync_step))
        assert len(default) == len(refill) == 3
