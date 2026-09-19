# Copyright 2026 Individual Contributor: Lirui Luo
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

"""Exercise the real resume/fit counters without importing Ray or model backends.

Compile the checked-in methods, not a copy of their counter logic. Checkpoint
I/O, queue termination and model updates are doubles; step advancement, actor
warmup, publication counters and JSONL serialization use the real methods.
"""

import ast
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_methods(relative_path, class_name, names, namespace):
    path = REPO_ROOT / relative_path
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [
        node for node in cls.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    extracted = ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[]))
    exec(compile(extracted, str(path), "exec"), namespace)


class _TrainingStopException(Exception):
    pass


async def _run_resume(tmp_path, *, mode, trigger, use_critic=True, warmup=0):
    version = 100
    checkpoint = tmp_path / f"global_step_{version}"
    find_latest = Mock(return_value=None if mode == "auto-empty" else str(checkpoint))
    namespace = dict(
        os=os,
        time=time,
        datetime=datetime,
        asyncio=asyncio,
        json=json,
        ThreadPoolExecutor=ThreadPoolExecutor,
        find_latest_ckpt_path=find_latest,
        Role=SimpleNamespace(Critic="critic"),
        TrainingStopException=_TrainingStopException,
        marked_timer=lambda *args, **kwargs: nullcontext(),
        reduce_metrics=lambda metrics: metrics,
    )
    trainer_methods = ("load_checkpoint", "fit", "_fit_update_local_step", "_fit_postprocess_step")
    _load_methods(
        "verl/experimental/fully_async_policy/fully_async_trainer.py", "FullyAsyncTrainer", trainer_methods, namespace
    )
    _load_methods(
        "verl/experimental/separation/ray_trainer.py", "SeparateRayPPOTrainer", ("_fit_update_actor",), namespace
    )
    dump_methods = ("_init_dump_executor", "_dump_generations", "_write_generations", "_shutdown_dump_executor")
    _load_methods("verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer", dump_methods, namespace)

    trainer = SimpleNamespace(
        config=SimpleNamespace(
            trainer=SimpleNamespace(
                resume_mode="auto" if mode == "auto-empty" else mode,
                default_hdfs_dir=None,
                default_local_dir=str(tmp_path),
                resume_from_path=str(checkpoint),
                del_local_ckpt_after_load=False,
                test_freq=-1,
                critic_warmup=warmup,
            )
        ),
        actor_rollout_wg=SimpleNamespace(load_checkpoint=Mock()),
        critic_wg=SimpleNamespace(load_checkpoint=Mock()),
        use_critic=use_critic,
        global_steps=0,
        current_param_version=0,
        last_ckpt_version=0,
        trigger_parameter_sync_step=trigger,
        local_trigger_step=1,
        required_samples=1,
        message_queue_client=SimpleNamespace(get_queue_size_sync=lambda: 0),
        rollouter=object(),
        metrics_aggregator=SimpleNamespace(add_step_metrics=Mock()),
        progress_bar=SimpleNamespace(close=Mock(), update=Mock()),
        timing_raw={},
        _fit_save_checkpoint=Mock(),
        _fit_log_aggregated_training_metrics=Mock(),
    )
    for name in (*trainer_methods, "_fit_update_actor", *dump_methods):
        method = namespace[name]
        setattr(trainer, name, method.__func__ if isinstance(method, staticmethod) else MethodType(method, trainer))

    async def noop():
        return None

    trainer._fit_update_weights = noop
    trainer._fit_validate = noop
    labels, versions, actor_updates = [], [], []

    def update_actor(batch):
        actor_updates.append(trainer.global_steps)
        return SimpleNamespace(meta_info={"metrics": {}})

    async def fit_step():
        if len(labels) == 2 * trigger:
            raise _TrainingStopException
        labels.append(trainer.global_steps)
        trainer.metrics = {"training/global_step": trainer.global_steps}
        trainer._fit_update_actor(None)
        trainer._fit_update_local_step()
        versions.append(trainer.current_param_version)
        trainer._dump_generations(["input"], ["output"], [None], [0.0], {}, str(tmp_path / "dumps"))
        trainer._fit_postprocess_step()

    trainer._update_actor = update_actor
    trainer.fit_step = fit_step
    trainer._init_dump_executor()
    try:
        restored_version = await trainer.load_checkpoint()
        restored_step = trainer.global_steps
        await trainer.fit()
    finally:
        # This test concerns resume; drain explicitly even before the separate
        # fully-async shutdown fix is applied.
        await asyncio.to_thread(trainer._shutdown_dump_executor)

    return trainer, restored_version, restored_step, labels, versions, actor_updates, checkpoint


@pytest.mark.parametrize("mode", ["disable", "auto-empty", "auto", "resume_path"])
@pytest.mark.parametrize("trigger", [1, 2, 4])
@pytest.mark.parametrize("use_critic", [False, True])
def test_resume_advances_once_before_first_update(tmp_path, mode, trigger, use_critic):
    trainer, restored_version, restored_step, labels, versions, actor_updates, checkpoint = asyncio.run(
        _run_resume(tmp_path, mode=mode, trigger=trigger, use_critic=use_critic)
    )
    initial_version = 0 if mode in ("disable", "auto-empty") else 100
    completed_steps = initial_version * trigger
    expected_labels = list(range(completed_steps + 1, completed_steps + 2 * trigger + 1))

    assert restored_version == initial_version
    assert restored_step == completed_steps
    assert labels == expected_labels
    assert actor_updates == expected_labels
    assert versions == [initial_version + (step + 1) // trigger for step in range(2 * trigger)]
    assert trainer.current_param_version == initial_version + 2
    assert trainer.last_ckpt_version == initial_version
    assert trainer.global_steps == expected_labels[-1] + 1
    assert trainer.local_trigger_step == 1
    assert trainer.progress_bar.update.call_count == 2
    trainer._fit_save_checkpoint.assert_called_once_with(force=True)
    logged_steps = [
        call.kwargs["metrics"]["training/global_step"]
        for call in trainer.metrics_aggregator.add_step_metrics.call_args_list
    ]
    assert logged_steps == expected_labels
    for step in expected_labels:
        rows = [json.loads(line) for line in (tmp_path / "dumps" / f"{step}.jsonl").read_text().splitlines()]
        assert [row["step"] for row in rows] == [step]

    if initial_version:
        trainer.actor_rollout_wg.load_checkpoint.assert_called_once_with(
            str(checkpoint / "actor"), del_local_after_load=False
        )
    else:
        trainer.actor_rollout_wg.load_checkpoint.assert_not_called()
    if initial_version and use_critic:
        trainer.critic_wg.load_checkpoint.assert_called_once_with(
            str(checkpoint / "critic"), del_local_after_load=False
        )
    else:
        trainer.critic_wg.load_checkpoint.assert_not_called()


@pytest.mark.parametrize("trigger", [1, 2, 4])
def test_resume_preserves_critic_warmup_boundary(tmp_path, trigger):
    first_step = 100 * trigger + 1
    warmup = first_step + 1
    _, _, _, labels, _, actor_updates, _ = asyncio.run(
        _run_resume(tmp_path, mode="resume_path", trigger=trigger, warmup=warmup)
    )
    assert labels[0] == first_step
    assert actor_updates == list(range(warmup, first_step + 2 * trigger))
