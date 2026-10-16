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

"""Exercise checkpoint control flow without importing GPU-only Megatron kernels.

Compile the production methods with a fake training operation and async writer.
This tests queue progress and publication ordering, not distributed GPU I/O.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import get_checkpoint_tracker_filename
from verl.utils.fs import local_mkdir_safe
from verl.workers.engine_workers import TrainingWorker

ROOT = Path(__file__).resolve().parents[2]


def _load_methods(path, class_name, methods, namespace, base=object):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.bases = [ast.Name(id="TestBase", ctx=ast.Load())]
    cls.decorator_list = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in cls.body} == set(methods)
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    scope = dict(namespace, TestBase=base)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[class_name]


class _Request:
    def __init__(self):
        self.finalize_fns = []
        self.ready = False

    def add_finalize_fn(self, fn):
        self.finalize_fns.append(fn)


class _Queue:
    def __init__(self):
        self.pending = []
        self.scheduled = 0

    def schedule_async_request(self, request):
        self.pending.append(request)
        self.scheduled += 1

    def maybe_finalize_async_calls(self, blocking=False):
        while self.pending and (blocking or self.pending[0].ready):
            for fn in self.pending.pop(0).finalize_fns:
                fn()


def _manager():
    namespace = dict(
        os=os,
        logger=None,
        log_with_rank=lambda *args, **kwargs: None,
        local_mkdir_safe=local_mkdir_safe,
        get_checkpoint_tracker_filename=get_checkpoint_tracker_filename,
    )
    for content in ("model", "optimizer", "extra"):
        namespace[f"get_{content}_dist_checkpoint_path"] = lambda root, content=content: os.path.join(
            root, content, "dist_ckpt"
        )
    cls = _load_methods(
        "verl/utils/checkpoint/megatron_checkpoint_manager.py",
        "MegatronCheckpointManager",
        [
            "save_checkpoint",
            "_schedule_dist_checkpoint_saves",
            "_dispatch_finalize",
            "_maybe_finalize_async_save",
            "_finalize_save",
        ],
        namespace,
    )
    manager = cls()
    manager.checkpoint_config = SimpleNamespace(async_save=True)
    manager.rank = 0
    manager.use_megatron_fsdp = False
    manager.should_save_optimizer = False
    manager.should_save_dist_ckpt_model = False
    manager.should_save_hf_model = False
    manager.should_save_extra = True
    manager._build_sharded_state_dict_metadata = lambda: {}
    manager._build_extra_state_dict = lambda: {"rng_state": 1}
    manager._save_transformer_config = lambda path: None
    manager._save_dist_checkpoint = lambda path, state: _Request()
    manager._write_checkpoint_manifest = lambda path, *args: (Path(path) / "ckpt_contents.json").write_text("ready")
    manager.register_checkpoint = MagicMock()
    manager._async_calls_queue = _Queue()
    return manager


def _engine(manager, output=None):
    class BaseEngine:
        def train_batch(self, data, loss_function):
            loss_function()
            return output

    cls = _load_methods(
        "verl/workers/engine/megatron/transformer_impl.py",
        "MegatronEngine",
        ["train_batch", "save_checkpoint"],
        dict(
            tu=tu,
            torch=SimpleNamespace(distributed=SimpleNamespace(barrier=lambda: None)),
            get_megatron_module_device=lambda module: "cuda",
            load_megatron_model_to_gpu=lambda *args, **kwargs: None,
            offload_megatron_model_to_cpu=lambda module: None,
        ),
        base=BaseEngine,
    )
    engine = cls()
    engine.checkpoint_config = manager.checkpoint_config
    engine.checkpoint_mananager = manager
    engine._is_offload_param = False
    engine.module = []
    return engine


@pytest.mark.parametrize("relative_path", ["global_step_3", "global_step_3/actor"])
def test_blocking_save_waits_for_older_and_current_publication(tmp_path, relative_path):
    manager = _manager()
    previous = str(tmp_path / relative_path.replace("step_3", "step_2"))
    current = str(tmp_path / relative_path)
    manager.save_checkpoint(previous, global_step=2)
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    assert not tracker.exists()
    worker = SimpleNamespace(engine=_engine(manager))

    TrainingWorker.save_checkpoint(worker, current, global_step=3, blocking=True)

    assert not manager._async_calls_queue.pending
    assert manager._async_calls_queue.scheduled == 2
    assert (Path(previous) / "ckpt_contents.json").exists()
    assert (Path(current) / "ckpt_contents.json").exists()
    assert tracker.read_text() == "3"
    assert manager.register_checkpoint.call_count == 2


@pytest.mark.parametrize("has_output", [False, True])
@pytest.mark.parametrize("opt_in", [False, True])
def test_training_progresses_ready_save_without_scheduling_another(tmp_path, has_output, opt_in):
    manager = _manager()
    manager.save_checkpoint(str(tmp_path / "global_step_2"), global_step=2)
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    output = {"metrics": {}} if has_output else None
    engine = _engine(manager, output)
    data = TensorDict({}, batch_size=[])
    if opt_in:
        tu.assign_non_tensor(data, finalize_async_checkpoint=True)

    # An unfinished write must not hold up the training step or be published.
    assert engine.train_batch(data, lambda: None) is output
    assert not tracker.exists()

    def finish_writes_during_training():
        assert not tracker.exists()
        for request in manager._async_calls_queue.pending:
            request.ready = True

    assert engine.train_batch(data, finish_writes_during_training) is output
    assert tracker.exists() is opt_in
    if opt_in:
        assert tracker.read_text() == "2"
    # RL requests without the SFT opt-in leave completion to their coordinator.
    assert manager._async_calls_queue.scheduled == 1


def test_failed_training_does_not_enter_checkpoint_collectives():
    manager = _manager()
    manager._maybe_finalize_async_save = MagicMock()
    engine = _engine(manager)
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, finalize_async_checkpoint=True)

    def fail():
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        engine.train_batch(data, fail)
    manager._maybe_finalize_async_save.assert_not_called()


def test_sync_training_does_not_progress_async_queue():
    manager = _manager()
    manager.checkpoint_config.async_save = False
    manager._maybe_finalize_async_save = MagicMock()
    engine = _engine(manager)
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, finalize_async_checkpoint=True)

    engine.train_batch(data, lambda: None)
    manager._maybe_finalize_async_save.assert_not_called()


@pytest.mark.parametrize("blocking", [False, True])
def test_async_save_without_distributed_requests_publishes_immediately(tmp_path, blocking):
    manager = _manager()
    manager._build_extra_state_dict = lambda: {}
    manager.save_checkpoint(str(tmp_path / "global_step_2"), global_step=2, blocking=blocking)

    assert manager._async_calls_queue.scheduled == 0
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "2"


def test_non_megatron_async_save_rejected_before_engine_construction():
    config = SimpleNamespace(
        model_config={},
        engine_config=SimpleNamespace(strategy="fsdp"),
        optimizer_config=None,
        checkpoint_config=SimpleNamespace(async_save=True),
    )
    with (
        patch("verl.workers.engine_workers.Worker.__init__", return_value=None),
        patch("verl.workers.engine_workers.initialize_global_process_group_ray"),
        patch("verl.workers.engine_workers.set_numa_affinity"),
        pytest.raises(NotImplementedError, match="fsdp does not support checkpoint.async_save"),
    ):
        TrainingWorker(config)
