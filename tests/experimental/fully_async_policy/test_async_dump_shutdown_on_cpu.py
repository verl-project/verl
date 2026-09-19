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

"""Test fully-async fit's dump barrier with real threads and local JSONL writes.

Compile the checked-in fit and inherited dump methods to avoid importing Ray
or model backends. Only training, validation and checkpoint RPCs are doubled.
Only trainer-owned dumps are covered; the rollouter has a separate executor
for validation output.
"""

import ast
import asyncio
import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

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


def _make_trainer(*, test_freq=-1, local_trigger_step=1):
    namespace = dict(
        os=os,
        json=json,
        asyncio=asyncio,
        ThreadPoolExecutor=ThreadPoolExecutor,
        TrainingStopException=_TrainingStopException,
    )
    _load_methods(
        "verl/experimental/fully_async_policy/fully_async_trainer.py", "FullyAsyncTrainer", ("fit",), namespace
    )
    dump_methods = ("_init_dump_executor", "_dump_generations", "_write_generations", "_shutdown_dump_executor")
    _load_methods("verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer", dump_methods, namespace)
    trainer = SimpleNamespace(
        message_queue_client=object(),
        rollouter=object(),
        global_steps=0,
        current_param_version=1,
        local_trigger_step=local_trigger_step,
        config=SimpleNamespace(trainer=SimpleNamespace(test_freq=test_freq)),
        progress_bar=SimpleNamespace(close=Mock()),
        _fit_save_checkpoint=Mock(),
        _fit_log_aggregated_training_metrics=Mock(),
    )
    for name in ("fit", *dump_methods):
        method = namespace[name]
        setattr(trainer, name, method.__func__ if isinstance(method, staticmethod) else MethodType(method, trainer))

    async def stop():
        raise _TrainingStopException

    async def noop():
        return None

    trainer.fit_step = stop
    trainer._fit_update_weights = noop
    trainer._fit_validate = AsyncMock(return_value=None)
    trainer._init_dump_executor()
    return trainer


@pytest.mark.parametrize("exit_branch", ["no-final-validation", "validation-check", "partial-sync"])
@pytest.mark.parametrize("write_fails", [False, True])
def test_fit_waits_for_tail_dump_and_propagates_io_error(tmp_path, exit_branch, write_fails):
    async def run():
        trainer = _make_trainer(
            test_freq=2 if exit_branch == "validation-check" else -1,
            local_trigger_step=2 if exit_branch == "partial-sync" else 1,
        )
        entered, release = threading.Event(), threading.Event()
        real_write = trainer._write_generations

        def blocked_write(*args):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("fit blocked the event loop while waiting for the dump")
            if write_fails:
                raise OSError("injected tail dump failure")
            real_write(*args)

        trainer._write_generations = blocked_write

        async def fit_step():
            trainer._dump_generations(["input"], ["output"], [None], [1.0], {}, str(tmp_path))
            trainer.fit_step = stop

        async def stop():
            raise _TrainingStopException

        trainer.fit_step = fit_step

        task = asyncio.create_task(trainer.fit())
        try:
            assert await asyncio.to_thread(entered.wait, 5), "the dump worker never started"
            await asyncio.sleep(0)
            # The event loop must still run while fit waits for the final write.
            # Unpatched fit has already returned successfully at this point.
            assert not task.done(), "fit returned before its last dump completed"
            release.set()
            if write_fails:
                with pytest.raises(OSError, match="injected tail dump failure"):
                    await asyncio.wait_for(task, timeout=5)
            else:
                await asyncio.wait_for(task, timeout=5)
                assert trainer._dump_futures == []
                assert json.loads((tmp_path / "1.jsonl").read_text()) == {
                    "input": "input",
                    "output": "output",
                    "gts": None,
                    "score": 1.0,
                    "step": 1,
                }
                with pytest.raises(RuntimeError, match="shutdown"):
                    trainer._dump_executor.submit(lambda: None)
            trainer._fit_save_checkpoint.assert_called_once_with(force=True)
            if exit_branch == "no-final-validation":
                trainer._fit_validate.assert_not_awaited()
            else:
                trainer._fit_validate.assert_awaited_once()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            trainer._dump_executor.shutdown(wait=True)

    asyncio.run(run())


@pytest.mark.parametrize("write_fails", [False, True])
def test_fit_checks_completed_but_unobserved_futures(write_fails):
    trainer = _make_trainer()
    future = Future()
    if write_fails:
        future.set_exception(OSError("completed dump failure"))
    else:
        future.set_result(None)
    trainer._dump_futures.append(future)
    try:
        if write_fails:
            with pytest.raises(OSError, match="completed dump failure"):
                asyncio.run(trainer.fit())
        else:
            asyncio.run(trainer.fit())
            assert trainer._dump_futures == []
            with pytest.raises(RuntimeError, match="shutdown"):
                trainer._dump_executor.submit(lambda: None)
    finally:
        trainer._dump_executor.shutdown(wait=True)


def test_fit_closes_executor_when_dumping_is_disabled():
    trainer = _make_trainer()
    try:
        asyncio.run(trainer.fit())
        assert trainer._dump_futures == []
        with pytest.raises(RuntimeError, match="shutdown"):
            trainer._dump_executor.submit(lambda: None)
    finally:
        trainer._dump_executor.shutdown(wait=True)


@pytest.mark.parametrize("failure_source", ["training", "validation"])
def test_dump_failure_does_not_mask_primary_training_error(failure_source):
    trainer = _make_trainer(test_freq=2)
    future = Future()
    future.set_exception(OSError("secondary dump failure"))
    trainer._dump_futures.append(future)

    async def fail():
        raise RuntimeError("primary training failure")

    if failure_source == "training":
        trainer.fit_step = fail
    else:
        trainer._fit_validate = fail
    try:
        with pytest.raises(RuntimeError, match="primary training failure"):
            asyncio.run(trainer.fit())
        trainer._fit_save_checkpoint.assert_not_called()
    finally:
        trainer._dump_executor.shutdown(wait=True)
