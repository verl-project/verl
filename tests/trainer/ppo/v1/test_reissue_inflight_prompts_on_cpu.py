# Copyright contributors to the PEAR project
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

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


@pytest.fixture
def restore(monkeypatch):
    """Runs _reissue_inflight_prompts on a restored partition; returns (prompts re-issued, keys cleared)."""

    def run(items: dict) -> tuple[int, set]:
        tq = MagicMock(kv_list=MagicMock(return_value={"train": items}))
        monkeypatch.setattr(trainer_base, "tq", tq)
        monkeypatch.setattr(trainer_base.tu, "assign_non_tensor_data", MagicMock())
        trainer = SimpleNamespace(
            trainer_mode="separate_async",
            global_steps=7,
            agent_loop_manager=MagicMock(),
            INFLIGHT_PROMPT_STATUSES=PPOTrainer.INFLIGHT_PROMPT_STATUSES,
        )
        trainer._reissue_prompt_groups = partial(PPOTrainer._reissue_prompt_groups, trainer)
        reissued = PPOTrainer._reissue_inflight_prompts(trainer)
        return reissued, {key for call in tq.kv_clear.call_args_list for key in call.kwargs["keys"]}

    return run


def test_consumed_rows_are_dropped_and_the_finished_groups_kept(restore):
    items = {
        "fin": {"is_prompt": True, "status": "finished", "global_steps": 6},
        "fin_0_0": {"status": "success"},
        "run": {"is_prompt": True, "status": "running", "global_steps": 6},
        "old_0_0": {},
        "pad_0_0": {"is_padding": True},  # the batch trained before the checkpoint, prompt keys gone
    }
    assert restore(items) == (1, {"old_0_0", "pad_0_0"})


def test_without_inflight_prompts(restore):
    items = {"fin": {"is_prompt": True, "status": "finished", "global_steps": 6}, "fin_0_0": {}, "old_0_0": {}}
    assert restore(items) == (0, {"old_0_0"})


def test_the_groups_of_the_trainer_s_statuses_go_to_its_hook_whose_count_is_returned(monkeypatch):
    items = {
        "que": {"is_prompt": True, "status": "queued", "global_steps": 5},
        "run": {"is_prompt": True, "status": "running", "global_steps": 6},
        "fin": {"is_prompt": True, "status": "finished", "global_steps": 6},
    }
    monkeypatch.setattr(trainer_base, "tq", MagicMock(kv_list=MagicMock(return_value={"train": items})))
    trainer = SimpleNamespace(
        trainer_mode="separate_async",
        INFLIGHT_PROMPT_STATUSES=("queued", "running"),
        _reissue_prompt_groups=MagicMock(return_value=5),
    )

    assert PPOTrainer._reissue_inflight_prompts(trainer) == 5
    trainer._reissue_prompt_groups.assert_called_once_with(["que", "run"], items, "train")
