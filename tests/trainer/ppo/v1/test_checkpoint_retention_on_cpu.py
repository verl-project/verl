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

import json
import os
import random
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback
from verl.trainer.ppo.checkpoint_retention import (
    RETENTION_REGISTRY_FILENAME,
    CheckpointRetention,
    build_checkpoint_retention,
)
from verl.trainer.ppo.v1.trainer_base import PPOTrainer

METRIC = "val-core/aime24/acc/mean@32"


def _config(tmp_path, retention=None, **trainer_overrides):
    trainer = {
        "default_local_dir": str(tmp_path),
        "save_freq": 10,
        "test_freq": 10,
        "max_actor_ckpt_to_keep": None,
        "max_critic_ckpt_to_keep": None,
        "checkpoint_retention": {"enable": True, "metric": METRIC, **(retention or {})},
    }
    trainer.update(trainer_overrides)
    return OmegaConf.create({"trainer": trainer, "actor_rollout_ref": {"actor": {"checkpoint": {}}}})


def _save(policy: CheckpointRetention, step: int):
    """Mimic the trainer: write the checkpoint dir, register it, then update the tracker."""
    os.makedirs(os.path.join(policy.root_dir, f"global_step_{step}", "actor"), exist_ok=True)
    policy.on_save(step)
    with open(os.path.join(policy.root_dir, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write(str(step))


def _run(policy, scores: dict[int, float]):
    for step, score in scores.items():
        _save(policy, step)
        policy.on_step_end(step, {METRIC: score})


def _on_disk(tmp_path) -> list[int]:
    return sorted(int(d.split("_")[-1]) for d in os.listdir(tmp_path) if d.startswith("global_step_"))


def test_keeps_best_and_latest(tmp_path):
    policy = build_checkpoint_retention(_config(tmp_path))

    _run(policy, {10: 0.3, 20: 0.5, 30: 0.4, 40: 0.45})

    assert _on_disk(tmp_path) == [20, 40]
    assert policy.keep_reasons() == {20: ["best"], 40: ["latest", "resume"]}


def test_converged_moves_forward_as_best_rises(tmp_path):
    policy = build_checkpoint_retention(_config(tmp_path, {"converge_tolerance": 0.05}))

    _run(policy, {10: 0.1, 20: 0.3, 30: 0.45, 40: 0.50})
    # Best 0.50, band >= 0.45: 30 is the earliest converged checkpoint for now.
    assert policy.converged_step() == 30

    _run(policy, {50: 0.48, 60: 0.52, 70: 0.53, 80: 0.51})
    # Best 0.53, band >= 0.48: 30 dropped out; 40 is the earliest checkpoint in the band.
    assert policy.best_step() == 70
    assert policy.converged_step() == 40
    assert policy.keep_reasons() == {
        40: ["converged"],
        60: ["converge_candidate"],
        70: ["best"],
        80: ["latest", "resume"],
    }
    assert _on_disk(tmp_path) == [40, 60, 70, 80]


@pytest.mark.parametrize("seed", range(20))
def test_online_choice_matches_offline_answer(tmp_path, seed):
    rng = random.Random(seed)
    tolerance = 0.05
    policy = build_checkpoint_retention(_config(tmp_path, {"converge_tolerance": tolerance}))
    # Noisy saturating curve.
    scores = {10 * (i + 1): min(0.6, 0.02 * i) + rng.gauss(0, 0.03) for i in range(60)}

    _run(policy, scores)

    best_score = max(scores.values())
    expected_best = min(s for s, v in scores.items() if v == best_score)
    expected_converged = min(s for s, v in scores.items() if v >= best_score - tolerance)
    assert policy.best_step() == expected_best
    assert policy.converged_step() == expected_converged
    on_disk = set(_on_disk(tmp_path))
    assert {expected_best, expected_converged, max(scores)} <= on_disk


def test_milestones_are_sparse_new_bests(tmp_path):
    policy = build_checkpoint_retention(_config(tmp_path, {"milestone_interval": 200}))

    # New bests at 80..400, 560, 640; only those >= 200 steps after the previous milestone
    # (or the start) become milestones.
    _run(policy, {80: 0.2, 160: 0.3, 240: 0.4, 320: 0.5, 400: 0.6, 480: 0.55, 560: 0.65, 640: 0.66, 720: 0.5})

    assert sorted(s for s, r in policy.records.items() if r.milestone) == [240, 560]
    assert policy.keep_reasons() == {
        240: ["milestone"],
        560: ["milestone"],
        640: ["best"],
        720: ["latest", "resume"],
    }
    assert _on_disk(tmp_path) == [240, 560, 640, 720]


def test_milestone_interval_counts_from_resume_step(tmp_path):
    cfg = _config(tmp_path, {"milestone_interval": 200})
    policy = build_checkpoint_retention(cfg)
    policy.load(current_step=2080)

    _run(policy, {2160: 0.5, 2240: 0.6, 2320: 0.7})

    assert sorted(s for s, r in policy.records.items() if r.milestone) == [2320]
    resumed = build_checkpoint_retention(cfg)
    resumed.load(current_step=2320)
    assert resumed.start_step == 2080


@pytest.mark.parametrize("window, expected_best", [(1, 20), (3, 60)])
def test_window_damps_a_single_lucky_evaluation(tmp_path, window, expected_best):
    policy = build_checkpoint_retention(_config(tmp_path, {"window": window}))

    # 20 is a one-off spike; 40-60 is a sustained level just below it.
    _run(policy, {10: 0.5, 20: 0.7, 30: 0.5, 40: 0.66, 50: 0.68, 60: 0.68})

    assert policy.best_step() == expected_best


def test_validation_without_checkpoint_feeds_smoothing(tmp_path):
    policy = build_checkpoint_retention(_config(tmp_path, {"window": 2}))
    policy.observe_validation(0, {METRIC: 0.1})

    _run(policy, {10: 0.5})

    assert policy.records[10].smoothed_score == pytest.approx(0.3)


def test_min_mode(tmp_path):
    policy = build_checkpoint_retention(_config(tmp_path, {"mode": "min", "converge_tolerance": 0.2}))

    _run(policy, {10: 2.0, 20: 1.1, 30: 1.0, 40: 1.5})

    assert policy.best_step() == 30
    assert policy.converged_step() == 20
    assert _on_disk(tmp_path) == [20, 30, 40]


def test_unscored_checkpoint_kept_only_while_latest(tmp_path):
    policy = build_checkpoint_retention(_config(tmp_path))

    _save(policy, 5)
    policy.on_step_end(5, None)
    _run(policy, {10: 0.2})

    assert _on_disk(tmp_path) == [10]


def test_tracker_step_is_never_deleted(tmp_path):
    policy = build_checkpoint_retention(_config(tmp_path))
    _run(policy, {10: 0.1, 20: 0.3})
    # A newer checkpoint is registered but the tracker still points at 20 (e.g. async save in flight).
    os.makedirs(tmp_path / "global_step_30")
    policy.on_save(30)
    policy.on_step_end(30, {METRIC: 0.2})
    os.makedirs(tmp_path / "global_step_40")
    policy.on_save(40)
    policy.on_step_end(40, {METRIC: 0.2})

    assert _on_disk(tmp_path) == [20, 40]
    assert policy.keep_reasons()[20] == ["resume", "best"]


def test_unregistered_directories_are_left_alone(tmp_path):
    os.makedirs(tmp_path / "global_step_7")
    policy = build_checkpoint_retention(_config(tmp_path))

    _run(policy, {10: 0.1, 20: 0.2, 30: 0.1})

    assert _on_disk(tmp_path) == [7, 20, 30]


def test_registry_persists_and_drops_future_steps_on_resume(tmp_path):
    cfg = _config(tmp_path, {"converge_tolerance": 0.2})
    policy = build_checkpoint_retention(cfg)
    _run(policy, {10: 0.6, 20: 0.3, 30: 0.7})

    with open(tmp_path / RETENTION_REGISTRY_FILENAME) as f:
        state = json.load(f)
    assert state["score_policy"] == {"metric": [METRIC], "mode": "max", "window": 1}
    assert [r["step"] for r in state["records"]] == [10, 30]

    resumed = build_checkpoint_retention(cfg)
    resumed.load(current_step=10)
    assert sorted(resumed.records) == [10]
    assert resumed.best_step() == 10
    _run(resumed, {20: 0.65})
    assert resumed.best_step() == 20
    assert resumed.converged_step() == 10


@pytest.mark.parametrize(
    "changed_policy",
    [{"mode": "min"}, {"window": 2}, {"metric": "val-core/profiles/high_temperature/aime24/acc/mean@32"}],
)
def test_resume_rejects_changed_score_policy_without_writing_or_deleting(tmp_path, changed_policy):
    policy = build_checkpoint_retention(_config(tmp_path))
    # Both checkpoints survive, but the old max-policy record flags would select
    # step 30=.85 and delete the true min-policy best at step 20=.8 after a restart.
    _run(policy, {10: 0.9, 20: 0.8})
    registry = tmp_path / RETENTION_REGISTRY_FILENAME
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    before_registry, before_tracker = registry.read_bytes(), tracker.read_bytes()
    resumed = build_checkpoint_retention(_config(tmp_path, changed_policy))

    with pytest.raises(ValueError, match="score policy changed"):
        resumed.load(current_step=20)

    assert resumed.records == {} and resumed.history == [] and resumed.start_step == 0
    assert registry.read_bytes() == before_registry
    assert tracker.read_bytes() == before_tracker
    assert _on_disk(tmp_path) == [10, 20]


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("enabled", [False, True])
def test_registry_without_score_provenance_is_preserved(tmp_path, version, enabled):
    policy = build_checkpoint_retention(_config(tmp_path))
    _run(policy, {10: 0.9, 20: 0.8})
    registry = tmp_path / RETENTION_REGISTRY_FILENAME
    state = json.loads(registry.read_text())
    state["version"] = version
    state.pop("score_policy")
    registry.write_text(json.dumps(state))
    before = registry.read_bytes()
    resumed = build_checkpoint_retention(_config(tmp_path, {"enable": enabled}))

    if enabled:
        with pytest.raises(ValueError, match="no score-policy provenance"):
            resumed.load(current_step=20)
        assert resumed.start_step == 0
    else:
        resumed.load(current_step=20)
        assert resumed.start_step == 20

    assert resumed.records == {} and resumed.history == []
    assert registry.read_bytes() == before
    assert _on_disk(tmp_path) == [10, 20]


def test_matching_score_policy_can_resume_with_a_different_keep_last(tmp_path):
    policy = build_checkpoint_retention(_config(tmp_path))
    _run(policy, {10: 0.9, 20: 0.8})
    # The normalized singleton metric list matches the original scalar setting.
    resumed = build_checkpoint_retention(_config(tmp_path, {"metric": [METRIC], "keep_last": 2}))
    resumed.load(current_step=20)
    _run(resumed, {30: 0.85})

    assert resumed.best_step() == 10
    assert _on_disk(tmp_path) == [10, 20, 30]


def test_missing_metric_raises(tmp_path):
    policy = build_checkpoint_retention(_config(tmp_path))
    _save(policy, 10)

    with pytest.raises(KeyError, match="val-core/aime25"):
        policy.on_step_end(10, {"val-core/aime25/acc/mean@32": 0.4})


def test_metric_list_is_averaged(tmp_path):
    other = "val-core/aime25/acc/mean@32"
    policy = build_checkpoint_retention(_config(tmp_path, {"metric": [METRIC, other]}))

    assert policy.score({METRIC: 0.6, other: 0.2}) == pytest.approx(0.4)


def test_disabled_by_default(tmp_path):
    cfg = OmegaConf.create(
        {"trainer": {"default_local_dir": str(tmp_path)}, "actor_rollout_ref": {"actor": {"checkpoint": {}}}}
    )
    assert not build_checkpoint_retention(cfg).enabled


@pytest.mark.parametrize(
    "retention, trainer_overrides, match",
    [
        ({"metric": None}, {}, "metric is required"),
        ({}, {"max_actor_ckpt_to_keep": 3}, "max_actor_ckpt_to_keep must be null"),
        ({}, {"test_freq": -1}, "test_freq > 0"),
        ({"keep_last": 0}, {}, "keep_last must be >= 1"),
        ({"mode": "best"}, {}, "mode must be"),
        ({"window": 0}, {}, "window must be >= 1"),
        ({"converge_tolerance": -0.1}, {}, "converge_tolerance must be >= 0"),
        ({"milestone_interval": 0}, {}, "milestone_interval must be >= 1"),
    ],
)
def test_invalid_config_rejected(tmp_path, retention, trainer_overrides, match):
    with pytest.raises(ValueError, match=match):
        build_checkpoint_retention(_config(tmp_path, retention, **trainer_overrides))


def test_async_save_requires_two_latest(tmp_path):
    cfg = _config(tmp_path)
    cfg.actor_rollout_ref.actor.checkpoint.async_save = True
    with pytest.raises(ValueError, match="keep_last must be >= 2"):
        build_checkpoint_retention(cfg)


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def test_trainer_save_registers_checkpoint(tmp_path):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = "sync"
    trainer.global_steps = 10
    trainer.use_critic = False
    trainer.config = _config(tmp_path, default_hdfs_dir=None)
    trainer.checkpoint_callback = CheckpointCallback(config=trainer.config)
    trainer.checkpoint_retention = build_checkpoint_retention(trainer.config)
    trainer.actor_rollout_wg = MagicMock()
    trainer.train_dataloader = MagicMock()
    trainer.train_dataloader.state_dict.return_value = {}

    trainer._save_checkpoint()

    assert sorted(trainer.checkpoint_retention.records) == [10]
    # Worker-side rotation stays disabled so it cannot delete retained shards.
    assert trainer.actor_rollout_wg.save_checkpoint.call_args.kwargs["max_ckpt_to_keep"] is None


@pytest.mark.parametrize("durable_step", [None, 10])
def test_async_retention_preserves_every_pending_save(tmp_path, durable_step):
    cfg = _config(tmp_path, {"keep_last": 2})
    cfg.actor_rollout_ref.actor.checkpoint.async_save = True
    policy = build_checkpoint_retention(cfg)
    if durable_step is not None:
        _save(policy, durable_step)
        policy.on_step_end(durable_step, {METRIC: 1.0})
    # Several saves can be queued before the worker durability tracker advances.
    for step in (20, 30, 40, 50):
        os.makedirs(tmp_path / f"global_step_{step}" / "actor")
        policy.on_save(step)
        policy.on_step_end(step, {METRIC: 0.1})
    assert {20, 30, 40, 50} <= set(_on_disk(tmp_path))
    for step in (20, 30, 40, 50):
        assert "pending_save" in policy.keep_reasons()[step]
    # Once FIFO completion confirms all queued saves, obsolete checkpoints may prune.
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("50")
    policy.prune()
    assert 30 not in _on_disk(tmp_path)
    assert 40 in _on_disk(tmp_path) and 50 in _on_disk(tmp_path)


@pytest.mark.parametrize("actor_async", [False, True])
@pytest.mark.parametrize("critic_enable,adv_estimator", [(None, "gae"), (True, "grpo")])
def test_retention_rejects_active_async_critic_from_composed_config(actor_async, critic_enable, adv_estimator):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from verl.trainer.ppo.utils import need_critic

    config_dir = str(Path(__file__).resolve().parents[4] / "verl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "model_engine=megatron",
                f"algorithm.adv_estimator={adv_estimator}",
                "critic.enable=" + ("null" if critic_enable is None else "true"),
                "critic.checkpoint.async_save=true",
                f"actor_rollout_ref.actor.checkpoint.async_save={str(actor_async).lower()}",
                "trainer.checkpoint_retention.enable=true",
                "trainer.checkpoint_retention.keep_last=2",
                "trainer.checkpoint_retention.metric=[val-core/example/acc/mean@32]",
                "trainer.test_freq=10",
                "trainer.save_freq=10",
            ],
        )
    assert need_critic(config)
    assert config.critic.strategy == "megatron"
    with pytest.raises(ValueError, match="synchronous critic"):
        build_checkpoint_retention(config)


def test_retention_allows_unused_async_critic_config(tmp_path):
    config = _config(tmp_path)
    config.critic = {"enable": False, "checkpoint": {"async_save": True}}
    config.algorithm = {"adv_estimator": "gae"}
    assert build_checkpoint_retention(config).enabled


def test_disabled_retention_accepts_active_async_critic_config(tmp_path):
    config = _config(tmp_path, {"enable": False})
    config.critic = {"enable": True, "checkpoint": {"async_save": True}}
    config.algorithm = {"adv_estimator": "gae"}
    assert not build_checkpoint_retention(config).enabled


@pytest.mark.parametrize("enabled,async_save", [(True, True), (True, False), (False, True)])
def test_rollback_behind_tracker_fails_only_for_enabled_async_retention(tmp_path, enabled, async_save):
    config = _config(tmp_path, {"enable": enabled, "keep_last": 2})
    config.actor_rollout_ref.actor.checkpoint.async_save = async_save
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("100")
    policy = build_checkpoint_retention(config)
    before = sorted(tmp_path.iterdir())
    if enabled and async_save:
        with pytest.raises(ValueError, match="cannot resume behind the durability tracker"):
            policy.load(current_step=10)
        assert not policy.records
        assert policy.start_step == 0
    else:
        policy.load(current_step=10)
        assert policy.start_step == 10
    # Loading/rejecting does not rewind another run's completion marker or write a new checkpoint.
    assert tracker.read_text() == "100"
    assert sorted(tmp_path.iterdir()) == before


@pytest.mark.parametrize("tracker_step", [None, 10, 5])
def test_async_retention_accepts_resume_without_future_tracker(tmp_path, tracker_step):
    config = _config(tmp_path, {"keep_last": 2})
    config.actor_rollout_ref.actor.checkpoint.async_save = True
    if tracker_step is not None:
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(str(tracker_step))
    policy = build_checkpoint_retention(config)
    policy.load(current_step=10)
    assert policy.start_step == 10
