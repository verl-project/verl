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

import random
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorData, TensorDict

from verl.trainer.ppo.v1 import agent_loop_tq, trainer_base
from verl.utils import tensordict_utils as tu


@pytest.fixture
def selector_config(tmp_path):
    policy = tmp_path / "source_policy.py"
    policy.write_text(
        "def choose(*, batch, global_step, seed, threshold):\n"
        "    return 'teacher' if int(batch['index'][0]) + global_step >= threshold and seed == 42 else 'student'\n"
        "\n"
        "def invalid(*, batch, global_step, seed):\n"
        "    return ['student', 'teacher']\n"
        "\n"
        "def fail(*, batch, global_step, seed):\n"
        "    raise RuntimeError('policy failed')\n"
        "\n"
        "not_callable = 3\n"
    )
    return OmegaConf.create(
        {
            "distillation": {
                "rollout_source_selector": {"path": str(policy), "name": "choose", "kwargs": {"threshold": 5}}
            }
        }
    )


def test_selector_uses_prompt_metadata_and_restored_step_without_consuming_global_rng(selector_config):
    from verl.experimental.teacher_rollout import build_rollout_source_selector

    batch = TensorDict({"index": torch.tensor([3])}, batch_size=[1])
    selector = build_rollout_source_selector(selector_config)
    rng = random.getstate()
    assert selector(batch=batch, global_step=1, seed=42) == "student"
    assert selector(batch=batch, global_step=2, seed=42) == "teacher"
    restored_selector = build_rollout_source_selector(selector_config)
    assert restored_selector(batch=batch, global_step=2, seed=42) == "teacher"
    assert random.getstate() == rng


@pytest.mark.parametrize("name,error", [("not_callable", TypeError), ("missing", AttributeError)])
def test_selector_load_errors_fail_before_generation(selector_config, name, error):
    from verl.experimental.teacher_rollout import build_rollout_source_selector

    selector_config.distillation.rollout_source_selector.name = name
    with pytest.raises(error):
        build_rollout_source_selector(selector_config)


@pytest.mark.parametrize("num_workers", [1, 2, 4])
@pytest.mark.parametrize("step,configured,expected", [(1, False, None), (1, True, "student"), (2, True, "teacher")])
def test_driver_decides_before_real_manager_fanout(
    monkeypatch, selector_config, num_workers, step, configured, expected
):
    from verl.experimental.teacher_rollout import build_rollout_source_selector

    chunks = []
    manager = SimpleNamespace(
        agent_loop_workers=[
            SimpleNamespace(generate_sequences=SimpleNamespace(remote=chunks.append)) for _ in range(num_workers)
        ]
    )
    manager.generate_sequences = lambda batch: agent_loop_tq.AgentLoopManagerTQ.generate_sequences(manager, batch)
    monkeypatch.setattr(agent_loop_tq.ray, "get", lambda refs: refs)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kwargs: None)
    config = selector_config if configured else OmegaConf.create({})
    OmegaConf.update(config, "actor_rollout_ref.rollout.seed", 42, force_add=True)
    trainer = SimpleNamespace(
        trainer_mode="sync",
        global_steps=step,
        config=config,
        rollout_source_selector=build_rollout_source_selector(config),
        agent_loop_manager=manager,
    )
    batch = TensorDict({"uid": NonTensorData("prompt"), "index": torch.full((8,), 3)}, batch_size=[8])
    for _ in range(2):
        assert trainer_base.PPOTrainer._submit_batch_to_rollout(trainer, batch.clone()) == 8
    assert len(chunks) == 2 * num_workers
    assert sum(len(chunk) for chunk in chunks) == 16
    assert all(tu.get_non_tensor_data(chunk, "rollout_source", None) == expected for chunk in chunks)


@pytest.mark.parametrize("name,error", [("invalid", ValueError), ("fail", RuntimeError)])
def test_bad_policy_result_never_publishes_or_generates(monkeypatch, selector_config, name, error):
    from verl.experimental.teacher_rollout import build_rollout_source_selector

    selector_config.distillation.rollout_source_selector.name = name
    selector_config.distillation.rollout_source_selector.kwargs = {}

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid policy must fail before TransferQueue writes or generation")

    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", forbidden)
    trainer = SimpleNamespace(
        config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"seed": 42}}}),
        global_steps=1,
        rollout_source_selector=build_rollout_source_selector(selector_config),
        agent_loop_manager=SimpleNamespace(generate_sequences=forbidden),
    )
    with pytest.raises(error):
        trainer_base.PPOTrainer._submit_batch_to_rollout(trainer, TensorDict({}, batch_size=[1]))


@pytest.mark.parametrize(
    "probability,want",
    [(0.0, ["student"] * 5), (1.0, ["teacher"] * 5), (0.5, ["teacher", "student", "teacher", "teacher", "student"])],
)
def test_example_policy_is_optional_and_reproducible(probability, want):
    from verl.experimental.teacher_rollout import build_rollout_source_selector

    config = OmegaConf.create(
        {
            "distillation": {
                "rollout_source_selector": {
                    "path": "pkg://verl.experimental.teacher_rollout",
                    "name": "select_source",
                    "kwargs": {"teacher_probability": probability},
                }
            }
        }
    )
    selector = build_rollout_source_selector(config)
    batch = TensorDict({}, batch_size=[1])
    rng = random.getstate()
    assert [selector(batch=batch, global_step=step, seed=42) for step in range(1, 6)] == want
    assert [selector(batch=batch, global_step=step, seed=42) for step in (4, 5)] == want[3:]
    assert random.getstate() == rng


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_example_policy_rejects_invalid_probability(probability):
    from verl.experimental.teacher_rollout import build_rollout_source_selector

    config = OmegaConf.create(
        {
            "distillation": {
                "rollout_source_selector": {
                    "path": "pkg://verl.experimental.teacher_rollout",
                    "name": "select_source",
                    "kwargs": {"teacher_probability": probability},
                }
            }
        }
    )
    with pytest.raises(ValueError, match="teacher_probability"):
        build_rollout_source_selector(config)(batch=TensorDict({}, batch_size=[1]), global_step=1, seed=42)
