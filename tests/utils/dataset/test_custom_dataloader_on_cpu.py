# Copyright 2026 Individual Contributor
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

"""Exercise the bridge and actual trainer method bodies without Ray/GPU imports."""

import ast
import contextlib
import importlib.util
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import ReadonlyConfigError

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("loader_bridge", ROOT / "verl/utils/dataset/dataloader.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class FixtureLoader:
    instances = []

    def __init__(self, *, arg1, arg2, _data_config, _split, _batch_size, _tokenizer, _processor, _collate_fn):
        self.arg1, self.arg2 = arg1, arg2
        self.context, self.split, self.batch_size = _data_config, _split, _batch_size
        self.runtime = (_tokenizer, _processor, _collate_fn)
        self.cursor = 0
        self.closed = 0
        self.instances.append(self)
        if arg2.get("fail_val") and _split == "val":
            raise RuntimeError("source unavailable")

    def __iter__(self):
        if self.split == "val":
            self.cursor = 0
        while self.cursor < self.arg2.get("batches", 6):
            position = self.cursor
            self.cursor += 1
            yield {
                "raw_prompt": [[{"role": "user", "content": str(position)}] for _ in range(self.batch_size)],
                "extra_info": [{"position": position} for _ in range(self.batch_size)],
            }

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]

    def close(self):
        self.closed += 1


class SizedLoader(FixtureLoader):
    def __len__(self):
        return self.arg2.get("batches", 6)

    def __iter__(self):
        if self.cursor == len(self):
            self.cursor = 0
        yield from super().__iter__()


class IterableOnly(FixtureLoader):
    state_dict = None
    load_state_dict = None


class SaveOnly(FixtureLoader):
    load_state_dict = None


def not_a_class(**kwargs):
    return FixtureLoader(**kwargs)


def config(loader=FixtureLoader):
    return OmegaConf.create(
        {
            "data": {
                "custom_dataloader": {
                    "_target_": f"{__name__}.{loader.__name__}",
                    "arg1": "${data.prompt_key}",
                    "arg2": {"arg2_1": "value2"},
                },
                "custom_cls": {"path": None, "name": None},
                "prompt_key": "prompt",
                "train_batch_size": 4,
                "gen_batch_size": 2,
                "val_batch_size": None,
            },
            "algorithm": {"filter_groups": {"enable": False}},
            "trainer": {
                "use_v1": True,
                "v1": {"trainer_mode": "sync", "sampler": {"sync_refill_failed_groups": False}},
                "total_training_steps": 3,
                "total_epochs": 1,
                "val_before_train": False,
                "val_only": False,
                "test_freq": -1,
                "save_freq": -1,
                "project_name": "test",
                "experiment_name": "test",
                "logger": [],
            },
            "actor_rollout_ref": {"actor": {"optim": {"lr": 1e-4}}},
            "global_profiler": {"steps": None},
        }
    )


def make_loader(cfg, split="train", batch_size=2):
    return bridge.create_custom_dataloader(
        cfg.data, split=split, batch_size=batch_size, tokenizer=None, processor=None, collate_fn=None
    )


def test_constructor_kwargs_context_and_nested_targets():
    cfg = config()
    cfg.data.custom_dataloader.arg2["_target_"] = "not.imported.NestedClass"
    train, val = make_loader(cfg), make_loader(cfg, split="val")
    assert train is not val
    assert train.arg1 == "prompt"
    assert train.arg2.arg2_1 == "value2"
    assert train.arg2._target_ == "not.imported.NestedClass"
    assert train.split == "train" and val.split == "val"
    assert train.context.custom_dataloader._target_ == cfg.data.custom_dataloader._target_
    assert OmegaConf.is_readonly(train.context)
    with pytest.raises(ReadonlyConfigError):
        train.context.train_batch_size = 100
    assert cfg.data.train_batch_size == 4
    assert not OmegaConf.is_readonly(cfg.data)


def test_runtime_inputs_are_not_reinterpreted_as_hydra_configs():
    @dataclass
    class RuntimeProcessor:
        name: str

    tokenizer, processor = object(), RuntimeProcessor("live")
    collate = lambda batch: batch
    loader = bridge.create_custom_dataloader(
        config().data, split="train", batch_size=2, tokenizer=tokenizer, processor=processor, collate_fn=collate
    )
    assert loader.runtime[0] is tokenizer
    assert loader.runtime[1] is processor
    assert loader.runtime[2] is collate


@pytest.mark.parametrize("name", sorted(bridge.RUNTIME_ARGUMENTS | {"_args_", "_partial_", "_convert_", "_recursive_"}))
def test_reserved_arguments_are_rejected(name):
    cfg = config()
    cfg.data.custom_dataloader[name] = None
    with pytest.raises(ValueError, match="reserved"):
        make_loader(cfg)


def test_custom_dataset_conflict_and_class_requirement():
    cfg = config()
    cfg.data.custom_cls.path = "custom_dataset"
    with pytest.raises(ValueError, match="cannot be combined"):
        make_loader(cfg)
    cfg.data.custom_cls.path = None
    cfg.data.custom_dataloader._target_ = f"{__name__}.not_a_class"
    with pytest.raises(ValueError, match="custom train dataloader"):
        make_loader(cfg)


def test_unknown_user_argument_is_not_silently_discarded():
    cfg = config()
    cfg.data.custom_dataloader.typo = True
    with pytest.raises(ValueError, match="typo"):
        make_loader(cfg)


def test_training_and_validation_accept_iteration_without_checkpoint_methods():
    cfg = config(IterableOnly)
    train, val = make_loader(cfg), make_loader(cfg, split="val")
    assert next(iter(train)) == next(iter(val))
    assert train.closed == val.closed == 0


def test_partial_checkpoint_interface_is_rejected_and_closed():
    cfg = config(SaveOnly)
    with pytest.raises(TypeError, match="state_dict"):
        make_loader(cfg)
    assert FixtureLoader.instances[-1].closed == 1


def test_resume_preserves_next_batch_and_validation_restarts():
    cfg = config()
    train = make_loader(cfg)
    iterator = iter(train)
    next(iterator)
    state = train.state_dict()
    expected = next(iterator)
    restored = make_loader(cfg)
    restored.load_state_dict(state)
    assert next(iter(restored)) == expected
    val = make_loader(cfg, split="val")
    assert list(val) == list(val)


@pytest.mark.parametrize(
    "batch",
    [
        None,
        {},
        {"raw_prompt": []},
        {"raw_prompt": [[], []], "extra_info": [{}]},
        {"raw_prompt": [[], []], "scalar": 1},
        {"raw_prompt": [[], []], "text": "ab"},
    ],
)
def test_invalid_batches_fail_without_an_extra_read(batch):
    with pytest.raises((TypeError, ValueError)):
        bridge.validate_custom_batch(batch, split="train", batch_size=2)


def test_validation_allows_a_short_tail_but_training_does_not():
    batch = {"raw_prompt": [[{"role": "user", "content": "hello"}]]}
    bridge.validate_custom_batch(batch, split="val", batch_size=2)
    with pytest.raises(ValueError, match="expected batch size"):
        bridge.validate_custom_batch(batch, split="train", batch_size=2)


def test_cleanup_attempts_both_loaders_and_surfaces_errors():
    first = SimpleNamespace(close=Mock(side_effect=RuntimeError("close failed")))
    second = SimpleNamespace(close=Mock())
    with pytest.raises(RuntimeError, match="close failed"):
        bridge.close_dataloaders(first, second, first)
    first.close.assert_called_once()
    second.close.assert_called_once()


def test_cleanup_failure_does_not_replace_the_training_error(caplog):
    loader = SimpleNamespace(close=Mock(side_effect=RuntimeError("close failed")))
    with pytest.raises(ValueError, match="training failed"):
        try:
            raise ValueError("training failed")
        finally:
            bridge.close_dataloaders(loader)
    assert "Custom dataloader cleanup failed" in caplog.text


@pytest.fixture
def trainer_class():
    # Compile unchanged method bodies; isolate Ray/worker imports rather than mocking the bridge.
    path = ROOT / "verl/trainer/ppo/v1/trainer_base.py"
    source = ast.parse(path.read_text())
    trainer = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "PPOTrainer")
    names = {
        "_init_dataloader",
        "_init_custom_dataloaders",
        "_init_builtin_dataloaders",
        "_resolve_gen_batch_size",
        "_fetch_one_gen_batch",
        "close_dataloaders",
        "fit",
        "_fit",
        "init",
        "step",
        "_load_checkpoint",
        "_save_checkpoint",
    }
    methods = [node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            ast.ClassDef(name="Trainer", bases=[], keywords=[], body=methods, decorator_list=[]),
        ],
        type_ignores=[],
    )
    namespace = {
        "OmegaConf": OmegaConf,
        "open_dict": open_dict,
        "np": np,
        "uuid": uuid,
        "os": os,
        "torch": torch,
        "find_latest_ckpt_path": Mock(return_value=None),
        "Role": SimpleNamespace(Critic="critic"),
        "logger": logging.getLogger(__name__),
        "collate_fn": lambda batch: batch,
        "create_custom_dataloader": bridge.create_custom_dataloader,
        "validate_custom_batch": bridge.validate_custom_batch,
        "close_dataloaders": bridge.close_dataloaders,
        "DataLoaderExhausted": bridge.DataLoaderExhausted,
        "create_rl_dataset": Mock(side_effect=[list(range(16)), list(range(3))]),
        "create_rl_sampler": Mock(),
        "StatefulDataLoader": Mock(),
        "tu": SimpleNamespace(get_tensordict=lambda batch: batch),
        "Tracking": Mock(),
        "ValidationGenerationsLogger": Mock(),
        "DapoFilteredRewardTableLogger": Mock(),
        "SkipManager": Mock(),
        "tq": Mock(),
        "tqdm": Mock(),
        "pprint": Mock(),
        "marked_timer": lambda *args, **kwargs: contextlib.nullcontext(),
        "MetricsAggregator": Mock(),
        "DAPO_FILTERED_REWARD_COUNTS_KEY": "filtered_counts",
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace["Trainer"], namespace


def trainer_instance(trainer_class, cfg=None):
    cls, namespace = trainer_class
    trainer = cls()
    trainer.config = cfg if cfg is not None else config()
    trainer.trainer_mode = trainer.config.trainer.v1.trainer_mode
    trainer.parameter_sync_step = 2
    trainer.tokenizer = trainer.processor = None
    return trainer, namespace


def test_custom_path_needs_no_files_or_dataset_and_sets_optimizer_horizon(trainer_class):
    trainer, namespace = trainer_instance(trainer_class)
    trainer.config.trainer.val_before_train = True
    trainer._init_dataloader()
    namespace["create_rl_dataset"].assert_not_called()
    namespace["StatefulDataLoader"].assert_not_called()
    assert trainer.steps_per_epoch is None
    assert trainer.total_training_steps == 3
    assert trainer.config.actor_rollout_ref.actor.optim.total_training_steps == 6
    train, val = trainer.train_dataloader, trainer.val_dataloader
    assert train.batch_size == 2 and val.batch_size == 4
    trainer.close_dataloaders()
    trainer.close_dataloaders()
    assert train.closed == val.closed == 1


def test_unsized_loader_requires_step_limit(trainer_class):
    trainer, _ = trainer_instance(trainer_class)
    trainer.config.trainer.total_training_steps = None
    with pytest.raises(ValueError, match="Unsized"):
        trainer._init_dataloader()
    trainer.close_dataloaders()


def test_sized_loader_epoch_budget_uses_fetch_size_not_fetch_count(trainer_class):
    trainer, _ = trainer_instance(trainer_class, config(SizedLoader))
    trainer.config.trainer.total_training_steps = None
    trainer.config.trainer.total_epochs = 4
    trainer._init_dataloader()
    assert len(trainer.train_dataloader) == 6
    assert trainer.steps_per_epoch == 3
    assert trainer.total_training_steps == 12


@pytest.mark.parametrize("total_steps", [None, 10])
def test_sized_source_rewinds_only_in_epoch_mode(trainer_class, total_steps):
    trainer, _ = trainer_instance(trainer_class, config(SizedLoader))
    trainer.config.trainer.total_training_steps = total_steps
    trainer._init_dataloader()
    for _ in range(6):
        trainer._fetch_one_gen_batch()
    if total_steps is None:
        assert trainer._fetch_one_gen_batch()["extra_info"][0]["position"] == 0
    else:
        with pytest.raises(bridge.DataLoaderExhausted):
            trainer._fetch_one_gen_batch()


def test_forced_fetch_size_reaches_custom_constructor(trainer_class):
    trainer, _ = trainer_instance(trainer_class)
    trainer.config.algorithm.filter_groups.enable = True
    trainer._init_dataloader()
    assert trainer.train_dataloader.batch_size == 1
    assert trainer.train_dataloader.context.gen_batch_size == 1


def test_default_loader_path_is_unchanged(trainer_class):
    trainer, namespace = trainer_instance(trainer_class)
    trainer.config.data.custom_dataloader = None
    trainer.config.data.update({"train_files": ["train"], "val_files": ["val"], "dataloader_num_workers": 0})
    trainer.config.trainer.total_training_steps = None
    trainer._init_dataloader()
    assert namespace["create_rl_dataset"].call_count == 2
    calls = namespace["StatefulDataLoader"].call_args_list
    assert calls[0].kwargs["batch_size"] == 2
    assert calls[0].kwargs["drop_last"] is True
    assert calls[1].kwargs["batch_size"] == 3
    assert calls[1].kwargs["drop_last"] is False
    assert trainer.total_training_steps == 4


def test_partial_initialization_can_close_the_training_loader(trainer_class):
    trainer, _ = trainer_instance(trainer_class)
    trainer.config.trainer.val_before_train = True
    trainer.config.data.custom_dataloader.arg2.fail_val = True
    with pytest.raises(RuntimeError, match="source unavailable"):
        trainer._init_dataloader()
    train = trainer.train_dataloader
    trainer.close_dataloaders()
    assert train.closed == 1


def test_init_failure_closes_the_successfully_constructed_loader(trainer_class):
    trainer, _ = trainer_instance(trainer_class)
    trainer.config.trainer.val_before_train = True
    trainer.config.data.custom_dataloader.arg2.fail_val = True
    trainer._setup = trainer._init_dataloader
    with pytest.raises(RuntimeError, match="source unavailable"):
        trainer.init()
    assert trainer.train_dataloader is None
    assert FixtureLoader.instances[-2].closed == 1


@pytest.mark.parametrize("mode", ["colocate_async", "separate_async"])
def test_unsupported_modes_fail_before_loader_construction(trainer_class, mode):
    trainer, _ = trainer_instance(trainer_class)
    trainer.trainer_mode = mode
    with pytest.raises(ValueError, match="synchronous"):
        trainer._init_dataloader()


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_fetch_size_does_not_silently_fall_back(trainer_class, size):
    trainer, _ = trainer_instance(trainer_class)
    trainer.config.data.gen_batch_size = size
    with pytest.raises(ValueError, match="positive integer"):
        trainer._init_dataloader()


def test_default_loaders_are_not_closed_by_custom_lifecycle(trainer_class):
    trainer, _ = trainer_instance(trainer_class)
    trainer.config.data.custom_dataloader = None
    loader = SimpleNamespace(close=Mock())
    trainer.train_dataloader = loader
    trainer._fit = Mock()
    trainer.fit(None)
    assert trainer.train_dataloader is loader
    loader.close.assert_not_called()


@pytest.mark.parametrize("batches,expected_steps", [(6, 3), (2, 1)])
def test_step_mode_ignores_epochs_and_stops_cleanly_without_rewinding(trainer_class, batches, expected_steps):
    trainer, _ = trainer_instance(trainer_class)
    trainer.config.trainer.total_epochs = 0
    trainer.config.data.custom_dataloader.arg2.batches = batches
    trainer._init_dataloader()
    loader = trainer.train_dataloader
    trainer.global_steps = 0
    for name in (
        "_reissue_inflight_prompts",
        "on_train_begin",
        "on_train_end",
        "on_step_begin",
        "on_step_end",
        "_start_profiling",
        "_stop_profiling",
        "_compute_metrics",
        "_shutdown_dump_executor",
    ):
        setattr(trainer, name, Mock())
    trainer._consume_sync_metrics = Mock(return_value={})

    def step(*args):
        for _ in range(trainer.config.data.train_batch_size // loader.batch_size):
            trainer._fetch_one_gen_batch()
        return SimpleNamespace(keys=[], partition_id="train")

    trainer.step = step
    trainer.fit(None)
    assert trainer._compute_metrics.call_count == expected_steps
    assert all(call.kwargs["epoch"] is None for call in trainer._compute_metrics.call_args_list)
    assert loader.cursor == batches
    assert loader.closed == 1


def test_fit_failure_closes_custom_loaders(trainer_class):
    trainer, _ = trainer_instance(trainer_class)
    trainer._init_dataloader()
    loader = trainer.train_dataloader
    trainer._fit = Mock(side_effect=RuntimeError("training failed"))
    with pytest.raises(RuntimeError, match="training failed"):
        trainer.fit(None)
    assert loader.closed == 1


def test_exhaustion_during_a_partial_update_is_not_success(trainer_class):
    trainer, _ = trainer_instance(trainer_class)
    trainer.prepare_step = Mock(return_value={})
    trainer._step_once = Mock(
        side_effect=[
            SimpleNamespace(keys=["first"], tags=[{}], partition_id="train"),
            bridge.DataLoaderExhausted(),
        ]
    )
    with pytest.raises(RuntimeError, match="during PPO update/refill"):
        trainer.step({}, {})


def checkpoint_trainer(trainer_class, tmp_path, monkeypatch, loader_class):
    trainer, namespace = trainer_instance(trainer_class, config(loader_class))
    trainer.config.trainer.update(
        default_local_dir=str(tmp_path),
        default_hdfs_dir=None,
        resume_mode="auto",
        resume_from_path=None,
        del_local_ckpt_after_load=False,
    )
    trainer._init_dataloader()
    trainer.global_steps = 1
    trainer.use_critic = True
    trainer.actor_rollout_wg = Mock()
    trainer.critic_wg = Mock()
    trainer.checkpoint_callback = Mock()
    monkeypatch.setitem(
        sys.modules,
        "verl.utils.fs",
        SimpleNamespace(local_mkdir_safe=lambda path: Path(path).mkdir(parents=True, exist_ok=True)),
    )
    return trainer, namespace


def test_nonresumable_loader_saves_model_and_removes_stale_data_state(trainer_class, tmp_path, monkeypatch):
    trainer, _ = checkpoint_trainer(trainer_class, tmp_path, monkeypatch, IterableOnly)
    checkpoint = tmp_path / "global_step_1"
    checkpoint.mkdir()
    (checkpoint / "data.pt").write_text("stale state from an earlier run")
    trainer._save_checkpoint()
    trainer.actor_rollout_wg.save_checkpoint.assert_called_once()
    trainer.critic_wg.save_checkpoint.assert_called_once()
    trainer.checkpoint_callback.on_save.assert_called_once()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "1"
    assert not (checkpoint / "data.pt").exists()


@pytest.mark.parametrize("resume_mode", ["auto", "resume_path"])
def test_nonresumable_loader_rejects_restore_before_loading_weights(trainer_class, tmp_path, monkeypatch, resume_mode):
    trainer, namespace = checkpoint_trainer(trainer_class, tmp_path, monkeypatch, IterableOnly)
    checkpoint = str(tmp_path / "global_step_1")
    trainer.config.trainer.resume_mode = resume_mode
    trainer.config.trainer.resume_from_path = checkpoint
    namespace["find_latest_ckpt_path"].return_value = checkpoint
    with pytest.raises(ValueError, match="does not support checkpoint restoration"):
        trainer._load_checkpoint()
    trainer.actor_rollout_wg.load_checkpoint.assert_not_called()
    trainer.critic_wg.load_checkpoint.assert_not_called()


@pytest.mark.parametrize("resume_mode", ["auto", "disable"])
def test_nonresumable_loader_allows_fresh_training(trainer_class, tmp_path, monkeypatch, resume_mode):
    trainer, _ = checkpoint_trainer(trainer_class, tmp_path, monkeypatch, IterableOnly)
    trainer.config.trainer.resume_mode = resume_mode
    trainer._load_checkpoint()
    assert trainer.global_steps == 0
    trainer.actor_rollout_wg.load_checkpoint.assert_not_called()


def test_stateful_loader_still_saves_and_restores_data_state(trainer_class, tmp_path, monkeypatch):
    trainer, namespace = checkpoint_trainer(trainer_class, tmp_path, monkeypatch, FixtureLoader)
    iterator = iter(trainer.train_dataloader)
    next(iterator)
    trainer._save_checkpoint()
    expected = next(iterator)
    namespace["find_latest_ckpt_path"].return_value = str(tmp_path / "global_step_1")
    trainer._load_checkpoint()
    assert next(iter(trainer.train_dataloader)) == expected
    assert trainer.global_steps == 1
    trainer.actor_rollout_wg.load_checkpoint.assert_called_once()
    trainer.critic_wg.load_checkpoint.assert_called_once()


def test_stateful_loader_still_rejects_missing_state(trainer_class, tmp_path, monkeypatch):
    trainer, namespace = checkpoint_trainer(trainer_class, tmp_path, monkeypatch, FixtureLoader)
    namespace["find_latest_ckpt_path"].return_value = str(tmp_path / "global_step_1")
    with pytest.raises(FileNotFoundError, match="missing"):
        trainer._load_checkpoint()
