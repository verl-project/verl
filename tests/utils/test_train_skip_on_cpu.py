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

"""Configuration, selection, and passthrough contracts for training skip."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.utils.config import validate_config
from verl.utils.skip import SkipManager, TrainSkip
from verl.utils.skip.base_skip import SKIP_REGISTRY
from verl.utils.skip.config import SkipManagerConfig, TrainSkipConfig


@pytest.fixture(autouse=True)
def reset_manager(monkeypatch):
    monkeypatch.setattr(SkipManager, "config", None)
    monkeypatch.setattr(SkipManager, "step", -1)
    monkeypatch.setattr(SkipManager, "skip_instances", {})


def _config(**train):
    return OmegaConf.create(
        {
            "skip": {"train": {"enable": True, **train}},
            "trainer": {"use_v1": True, "v1": {"trainer_mode": "sync"}},
        }
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"enable": 1},
        {"scope": "rollout"},
        {"action": "cache"},
        {"action": "repeat"},
        {"steps": "all"},
        {"steps": [0]},
        {"steps": [-1]},
        {"steps": [True]},
        {"steps": [1.5]},
        {"steps": ["1"]},
    ],
)
def test_invalid_config(fields):
    with pytest.raises(ValueError, match="skip.train"):
        TrainSkipConfig(**fields)


def test_defaults_and_registration():
    cfg = SkipManagerConfig()
    assert not cfg.train.enable
    assert cfg.train.scope == "updates"
    assert cfg.train.steps is None
    assert cfg.train.action == "empty"
    assert SKIP_REGISTRY["train"] is TrainSkip


@pytest.mark.parametrize("steps, expected", [(None, [1, 2, 3]), ([], []), ([1, 3], [1, 3])])
def test_step_selection(steps, expected):
    SkipManager.init(_config(steps=steps))
    assert SkipManager.is_enabled("train")
    assert not SkipManager.is_active("train")  # init has not set a trainer step
    assert [s for s in range(4) if SkipManager.is_active("train", step=s)] == expected
    SkipManager.set_step(2)
    assert SkipManager.is_active("train") == (2 in expected)


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
def test_scope_selection(scope):
    SkipManager.init(_config(scope=scope))
    SkipManager.set_step(1)
    assert SkipManager.is_active("train")
    assert SkipManager.is_active("train", scope="updates")
    assert SkipManager.is_active("train", scope="train_compute") == (scope == "train_compute")
    assert not SkipManager.is_active("train", scope="unknown")


def test_disabled_and_uninitialized_are_noops():
    assert not SkipManager.is_enabled("train")
    assert not SkipManager.is_active("train", step=1)
    assert not SkipManager.is_active("unknown", step=1)
    SkipManager.init(_config(enable=False))
    assert not SkipManager.is_enabled("train")
    assert not SkipManager.is_active("train", step=1)


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("keyword_call", [False, True])
@pytest.mark.parametrize("batch_type", ["queue_metadata", "data_proto"])
def test_decorator_preserves_identity_without_executing_or_caching(async_call, keyword_call, batch_type):
    SkipManager.init(_config(scope="train_compute", steps=[2]))
    calls = []

    def work(self, batch, metrics):
        calls.append(batch)
        metrics["executed"] = 1
        return batch

    async def async_work(self, batch, metrics):
        return work(self, batch, metrics)

    wrapped = SkipManager.annotate(role="train", scope="train_compute")(async_work if async_call else work)
    batch = DataProto() if batch_type == "data_proto" else SimpleNamespace(partition_id="train")

    for step in [1, 2, 3]:
        SkipManager.set_step(step)
        metrics = {}
        result = wrapped(None, batch=batch, metrics=metrics) if keyword_call else wrapped(None, batch, metrics)
        if async_call:
            result = asyncio.run(result)
        assert result is batch
        assert ("executed" in metrics) == (step != 2)
    assert calls == [batch, batch]
    assert SkipManager.skip_instances["train"].dump_dir is None


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("batch_type", ["queue_metadata", "data_proto", "unknown"])
def test_validation_or_unknown_batch_is_not_skipped(async_call, batch_type):
    SkipManager.init(_config())
    SkipManager.set_step(1)
    calls = []

    def work(self, batch):
        calls.append(batch)
        return batch

    async def async_work(self, batch):
        return work(self, batch)

    wrapped = SkipManager.annotate(role="train", scope="updates")(async_work if async_call else work)
    batches = {
        "queue_metadata": SimpleNamespace(partition_id="val"),
        "data_proto": DataProto(meta_info={"validate": True}),
        "unknown": object(),
    }
    batch = batches[batch_type]
    result = wrapped(None, batch)
    assert (asyncio.run(result) if async_call else result) is batch
    assert calls == [batch]


@pytest.mark.parametrize("entry", ["v0", "v1", "fully_async"])
def test_all_rollout_entry_points_accept_training_skip(entry):
    cfg = _config()
    if entry == "v0":
        cfg.trainer.use_v1 = False
    elif entry == "fully_async":
        cfg.async_training = {"trigger_parameter_sync_step": 1}
    SkipManager.init(cfg)
    assert SkipManager.is_active("train", step=1)


def test_invalid_training_options_fail_before_worker_configuration():
    cfg = _config(action="cache")
    with pytest.raises(ValueError, match="training results cannot be cached"):
        validate_config(cfg, use_reference_policy=False, use_critic=False)


def test_old_configs_keep_training_enabled():
    cfg = _config()
    del cfg.skip.train
    SkipManager.init(cfg)
    assert not SkipManager.is_active("train", step=1)
    assert SkipManager.config.rollout.steps == []
    assert SkipManager.config.async_rollout.steps == []
    assert SkipManager.config.rollout_tq.steps == []


@pytest.mark.parametrize("rollout_role", ["rollout", "rollout_tq", "async_rollout"])
def test_training_and_rollout_selection_remain_independent(monkeypatch, rollout_role):
    cfg = _config(scope="train_compute", steps=[2])
    cfg.skip[rollout_role] = {"enable": True, "steps": [1], "action": "cache"}
    SkipManager.init(cfg)
    rollout_skip = SkipManager.skip_instances[rollout_role]
    events = []
    training_batch = SimpleNamespace(partition_id="train") if rollout_role == "rollout_tq" else DataProto()

    @SkipManager.annotate(role="train", scope="train_compute")
    def train(self, batch):
        events.append((SkipManager.step, "train"))
        return batch

    if rollout_role == "rollout_tq":
        monkeypatch.setattr(rollout_skip, "maybe_load_and_inject", lambda *args: True)

        class Trainer:
            def _next_train_batch(self):
                events.append((SkipManager.step, "dataloader"))
                return {"uid": ["prompt"]}

            def _submit_batch_to_rollout(self, batch):
                events.append((SkipManager.step, "generate"))

            @SkipManager.annotate_tq(role="rollout_tq", phase="submit")
            def generate(self):
                self._submit_batch_to_rollout(self._next_train_batch())

        generate = Trainer().generate
    else:
        monkeypatch.setattr(rollout_skip, "meet_precondition", lambda *args: True)
        monkeypatch.setattr(rollout_skip, "warp_function", lambda *args: training_batch)

        @SkipManager.annotate(role=rollout_role)
        async def generate_sequences(self, prompts):
            events.append((SkipManager.step, "generate"))
            return training_batch

        def generate():
            prompts = DataProto(non_tensor_batch={"uid": np.array([f"uid_sample_0_{SkipManager.step}"])})
            return asyncio.run(generate_sequences(None, prompts))

    for step in [1, 2]:
        SkipManager.set_step(step)
        generate()
        assert train(None, training_batch) is training_batch
        assert SkipManager.step == step  # async sample indexing must not change the trainer step

    assert [(step, event) for step, event in events if event != "dataloader"] == [(1, "train"), (2, "generate")]
    if rollout_role == "rollout_tq":
        assert [step for step, event in events if event == "dataloader"] == [1, 2]
