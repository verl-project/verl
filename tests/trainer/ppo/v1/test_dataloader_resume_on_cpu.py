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

"""Check V1 checkpoint data order using the real sampler, loader, and trainer methods."""

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.ppo.utils import create_rl_sampler
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    """Disable callbacks while exercising the production V1 dataloader path."""

    def on_step_end(self):
        """Skip step callbacks in the dataloader-only test trainer."""
        pass

    def on_sample_end(self):
        """Skip sample callbacks in the dataloader-only test trainer."""
        pass


def _make_trainer(checkpoint_dir, seed, num_workers):
    """Build a minimal V1 trainer with a real random sampler and stateful loader."""
    # Bypass model/worker initialization, but exercise the production checkpoint
    # loader and _fetch_one_gen_batch, including its StopIteration handling.
    trainer = _StubTrainer.__new__(_StubTrainer)
    dataset = [{"index": index, "raw_prompt": index} for index in range(64)]
    sampler = create_rl_sampler(OmegaConf.create({"shuffle": True, "seed": seed}), dataset)
    trainer.train_dataloader = StatefulDataLoader(
        dataset, batch_size=8, sampler=sampler, num_workers=num_workers, drop_last=True
    )
    trainer.train_dataloader_it = None
    trainer.steps_per_epoch = len(trainer.train_dataloader)
    trainer.global_steps = 0
    trainer.trainer_mode = "sync"
    trainer.use_critic = False
    trainer.actor_rollout_wg = MagicMock()
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "resume_mode": "resume_path",
                "resume_from_path": str(checkpoint_dir),
                "del_local_ckpt_after_load": False,
            }
        }
    )
    return trainer


def _fetch_indices(trainer, count):
    """Fetch batches and return their dataset indices in traversal order."""
    return [trainer._fetch_one_gen_batch()["index"].tolist() for _ in range(count)]


@pytest.mark.parametrize(
    "seed,checkpoint_step,num_workers",
    [(42, 3, 0), (None, 3, 0), (42, 8, 0), (None, 8, 0)],
)
def test_resume_preserves_batch_order(tmp_path, seed, checkpoint_step, num_workers):
    """Preserve V1 sample order across mid-epoch and epoch-boundary resumes."""
    checkpoint_dir = tmp_path / f"global_step_{checkpoint_step}"
    checkpoint_dir.mkdir()
    continuous = _make_trainer(checkpoint_dir, seed, num_workers)
    steps_per_epoch = continuous.steps_per_epoch
    total_steps = 3 * steps_per_epoch

    expected = _fetch_indices(continuous, checkpoint_step)
    # Save inside the last batch body, before asking the iterator for
    # StopIteration, just like a checkpoint at the end of a training step.
    torch.save(continuous.train_dataloader.state_dict(), checkpoint_dir / "data.pt")
    expected += _fetch_indices(continuous, total_steps - checkpoint_step)
    assert expected[:steps_per_epoch] != expected[steps_per_epoch : 2 * steps_per_epoch]

    resumed = _make_trainer(checkpoint_dir, seed, num_workers)
    resumed._load_checkpoint()
    assert resumed.global_steps == checkpoint_step
    actual = expected[:checkpoint_step] + _fetch_indices(resumed, total_steps - checkpoint_step)

    # Dropping the exhausted state loses RandomSampler's RNG progress and
    # replays epoch 0, including when seed=None creates a fresh Generator.
    assert actual == expected
