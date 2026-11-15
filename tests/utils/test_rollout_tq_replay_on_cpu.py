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

"""CPU tests for ``RolloutTqSkip.load_dump_data`` replay key/row fidelity.

A cached prompt group can hold several trajectories per session (agent-loop
outputs ``{uid}_{session_id}_{index}``). The replay must inject every cached
row under the new uid while keeping each row's saved ``session_id`` and
``index``, so that ``compute_advantage_for_multi_trajectories`` still anchors
each session on its highest-index output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.skip.config import RolloutTqSkipConfig
from verl.utils.skip.rollout_skip import RolloutTqSkip
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


def _make_skip(tmp_path: Path) -> RolloutTqSkip:
    cfg = OmegaConf.create(
        {
            "skip": {"rollout_tq": {"enable": True, "dump_dir": str(tmp_path), "steps": [1], "action": "cache"}},
            "actor_rollout_ref": {"rollout": {"n": 2}},
            "trainer": {
                "experiment_name": "replay_exp",
                "project_name": "replay_proj",
                "v1": {"trainer_mode": "sync"},
            },
            "data": {"train_batch_size": 1, "gen_batch_size": 1},
        }
    )
    return RolloutTqSkip(omega_conf_to_dataclass(cfg.skip.rollout_tq, RolloutTqSkipConfig), cfg)


def _write_dump(skip: RolloutTqSkip, keys: list[str], session_ids: list[int]) -> None:
    """Save one cached step: one trajectory per key, ``responses=[i + 1]``."""
    rows = [
        {
            "responses": torch.tensor([i + 1]),
            "response_mask": torch.tensor([1]),
            "uid": keys[i].rsplit("_", 2)[0],
            "session_id": session_ids[i],
        }
        for i in range(len(keys))
    ]
    inner_dir = skip._get_v1_inner_dir(1, 0)
    inner_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tensordict": list_of_dict_to_tensordict(rows),
            "tags": [{"status": "success", "seq_len": 2}] * len(keys),
            "keys": keys,
            "global_steps": 1,
        },
        inner_dir / skip.tq_batch_name,
    )
    (inner_dir / skip.meta_name).write_text(
        json.dumps({"global_steps": 1, "num_trajectories": len(keys)}), encoding="utf-8"
    )


@pytest.fixture()
def tq_puts(monkeypatch):
    """Capture ``kv_batch_put`` calls instead of writing to a TransferQueue."""
    import transfer_queue as tq

    calls: list[dict] = []

    def fake_kv_batch_put(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(tq, "kv_batch_put", fake_kv_batch_put, raising=True)
    return calls


def test_replay_preserves_every_trajectory_and_key_suffix(tmp_path, tq_puts):
    skip = _make_skip(tmp_path)
    # Session 0 produced three outputs, session 1 produced one.
    _write_dump(skip, ["old_0_0", "old_0_1", "old_0_2", "old_1_0"], session_ids=[0, 0, 0, 1])

    skip.load_dump_data(step=1, new_prompt_uids=["new"], global_steps=9, partition_id="replay")

    assert len(tq_puts) == 2, "expected one trajectory put and one prompt-level put"
    traj_put, prompt_put = tq_puts
    # All four cached rows survive, re-keyed onto the new uid with their own
    # session id and output index.
    assert traj_put["keys"] == ["new_0_0", "new_0_1", "new_0_2", "new_1_0"]
    # Row content stays aligned with the key: responses 1..4 in saved order.
    assert [row.tolist() for row in traj_put["fields"]["responses"].unbind()] == [[1], [2], [3], [4]]
    assert [int(sid) for sid in traj_put["fields"]["session_id"]] == [0, 0, 0, 1]
    # Staleness tags are overridden for the current step.
    assert all(tag["global_steps"] == 9 for tag in traj_put["tags"])
    assert prompt_put["keys"] == ["new"]
    assert all(tag["is_prompt"] and tag["status"] == "finished" for tag in prompt_put["tags"])


def test_replay_maps_groups_to_new_prompts_faithfully(tmp_path, tq_puts):
    skip = _make_skip(tmp_path)
    # Two cached prompt groups; the batch also has two new prompts.
    _write_dump(
        skip,
        ["old1_0_0", "old1_0_1", "old1_1_0", "old2_0_0"],
        session_ids=[0, 0, 1, 0],
    )

    skip.load_dump_data(step=1, new_prompt_uids=["new1", "new2"], global_steps=9, partition_id="replay")

    traj_put = tq_puts[0]
    assert traj_put["keys"] == ["new1_0_0", "new1_0_1", "new1_1_0", "new2_0_0"]
    assert [row.tolist() for row in traj_put["fields"]["responses"].unbind()] == [[1], [2], [3], [4]]
