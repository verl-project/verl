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

from types import SimpleNamespace

import pytest
import transfer_queue as tq
from omegaconf import OmegaConf

import verl.trainer.main_ppo as main_ppo
import verl.trainer.ppo.v1 as trainer_v1
import verl.trainer.ppo.v1.streaming_dataloader as streaming_module
from verl.trainer.main_ppo import TaskRunnerV1


def _runner():
    runner_cls = TaskRunnerV1.__ray_metadata__.modified_class
    return runner_cls()


def test_task_runner_creates_node_affined_dataloader_with_trainer_dependencies(monkeypatch):
    runner = _runner()
    runner.config = OmegaConf.create({"trainer": {"v1": {"trainer_mode": "sync"}}})
    reward_handles = ["reward-worker"]
    runner.trainer = SimpleNamespace(
        tokenizer="tokenizer",
        processor="processor",
        filter_groups_metric="acc",
        get_llm_client=lambda: "llm-client",
        get_teacher_client=lambda: "teacher-client",
        get_reward_handles=lambda: reward_handles,
    )
    calls = {}

    class _ActorOptions:
        def remote(self, **kwargs):
            calls["remote"] = kwargs
            return "streaming-actor"

    class _ActorClass:
        def options(self, **kwargs):
            calls["options"] = kwargs
            return _ActorOptions()

    monkeypatch.setattr(streaming_module, "create_streaming_dataloader_actor_cls", lambda _config: _ActorClass())
    monkeypatch.setattr(main_ppo.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "node-1"))
    monkeypatch.setattr(
        main_ppo,
        "NodeAffinitySchedulingStrategy",
        lambda **kwargs: ("node-affinity", kwargs),
    )

    runner.init_streaming_dataloader()

    assert runner.streaming_dataloader == "streaming-actor"
    assert calls["options"] == {"scheduling_strategy": ("node-affinity", {"node_id": "node-1", "soft": False})}
    assert calls["remote"] == {
        "config": runner.config,
        "tokenizer": "tokenizer",
        "processor": "processor",
        "llm_client": "llm-client",
        "teacher_client": "teacher-client",
        "reward_loop_worker_handles": reward_handles,
        "filter_groups_metric": "acc",
    }


def test_task_runner_failure_finishes_tracking_then_kills_actor_before_tq_close(monkeypatch):
    runner = _runner()
    config = OmegaConf.create(
        {
            "transfer_queue": {"enable": False},
            "trainer": {"v1": {"trainer_mode": "sync"}},
        }
    )
    events = []

    class _Tracking:
        def finish(self, exit_code):
            events.append(("finish", exit_code))

    class _Trainer:
        def __init__(self, config):
            self.config = config
            self.logger = _Tracking()

        def init(self):
            events.append(("trainer_init",))

        def fit(self, *, streaming_dataloader):
            events.append(("fit", streaming_dataloader))
            raise RuntimeError("fit failed")

    def init_streaming_dataloader():
        events.append(("actor_init",))
        runner.streaming_dataloader = "streaming-actor"

    runner.init_streaming_dataloader = init_streaming_dataloader
    monkeypatch.setattr(main_ppo, "configure_verl_logging", lambda: None)
    monkeypatch.setattr(main_ppo, "pprint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer_v1, "get_trainer_cls", lambda _mode: _Trainer)
    monkeypatch.setattr(tq, "init", lambda _config: events.append(("tq_init",)))
    monkeypatch.setattr(tq, "close", lambda: events.append(("tq_close",)))
    monkeypatch.setattr(main_ppo.ray, "kill", lambda actor: events.append(("ray_kill", actor)))

    with pytest.raises(RuntimeError, match="fit failed"):
        runner.run(config)

    assert config.transfer_queue.enable is True
    assert events == [
        ("tq_init",),
        ("trainer_init",),
        ("actor_init",),
        ("fit", "streaming-actor"),
        ("finish", 1),
        ("ray_kill", "streaming-actor"),
        ("tq_close",),
    ]
