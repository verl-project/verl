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
        trainer = SimpleNamespace(trainer_mode="separate_async", global_steps=7, agent_loop_manager=MagicMock())
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
