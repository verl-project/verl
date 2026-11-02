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
import math
import os
import sys
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
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


class RepeatingLoader(IterableOnly):
    def __iter__(self):
        while True:
            self.cursor = 0
            yield from super().__iter__()


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


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_config_validation_accepts_all_v1_modes_but_not_legacy(mode, monkeypatch):
    module_spec = importlib.util.spec_from_file_location("config_validation", ROOT / "verl/utils/config.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.setattr(module, "omega_conf_to_dataclass", lambda cfg: SimpleNamespace(validate=lambda *args: None))
    cfg = config()
    cfg.trainer.update(n_gpus_per_node=1, nnodes=1)
    cfg.trainer.v1.trainer_mode = mode
    cfg.actor_rollout_ref.actor.update(use_dynamic_bsz=True, policy_loss={})
    cfg.actor_rollout_ref.update(model={}, rollout={"val_kwargs": {"do_sample": False}})
    module.validate_config(cfg, use_reference_policy=False, use_critic=False)

    cfg.trainer.use_v1 = False
    with pytest.raises(ValueError, match="requires the V1"):
        module.validate_config(cfg, use_reference_policy=False, use_critic=False)


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


class Batch(dict):
    def __len__(self):
        return len(self["raw_prompt"])

    def select(self, *keys):
        return Batch({key: self[key] for key in keys})


class MemoryQueue:
    """Only the TransferQueue I/O boundary is faked; replay decisions use production code."""

    def __init__(self):
        self.tags = defaultdict(dict)
        self.fields = defaultdict(dict)
        self.checkpoints = {}

    def kv_list(self, partition_id=None):
        return deepcopy(self.tags if partition_id is None else {partition_id: self.tags[partition_id]})

    def kv_batch_put(self, *, keys, partition_id, tags, fields=None):
        for index, (key, tag) in enumerate(zip(keys, tags, strict=True)):
            self.tags[partition_id][key] = tag
            if fields is not None:
                self.fields[partition_id][key] = {name: values[index] for name, values in fields.items()}

    def kv_clear(self, *, keys, partition_id):
        for key in keys:
            self.tags[partition_id].pop(key, None)
            self.fields[partition_id].pop(key, None)

    def kv_batch_get(self, *, keys, partition_id, select_fields=None):
        rows = [self.fields[partition_id][key] for key in keys]
        return Batch({name: [row[name] for row in rows] for name in (select_fields or rows[0])})

    def save_checkpoint(self, path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.checkpoints[path] = deepcopy((self.tags, self.fields))

    def load_checkpoint(self, path):
        self.tags, self.fields = deepcopy(self.checkpoints[path])


@pytest.fixture
def replay_types():
    path = ROOT / "verl/trainer/ppo/v1/replay_buffer.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.Import | ast.ImportFrom)]
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    queue = MemoryQueue()
    namespace = {
        "__name__": __name__,
        "logging": logging,
        "os": os,
        "time": SimpleNamespace(
            time=lambda: 0,
            sleep=Mock(side_effect=AssertionError("Unexpected polling without a producer")),
        ),
        "Counter": Counter,
        "defaultdict": defaultdict,
        "np": np,
        "tq": queue,
        "KVBatchMeta": SimpleNamespace,
        "DataLoaderExhausted": bridge.DataLoaderExhausted,
        "SkipManager": SimpleNamespace(annotate_tq=lambda **kwargs: lambda fn: fn),
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def trainer_class(replay_types):
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
        "_next_train_batch",
        "_submit_batch_to_rollout",
        "_add_prompts_to_generate",
        "_add_batch_to_generate",
        "_add_async_warmup_batches",
        "_reissue_inflight_prompts",
        "prepare_step",
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
        "math": math,
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
        "ReplayBufferAsync": replay_types["ReplayBufferAsync"],
        "NonTensorData": type("NonTensorData", (), {}),
        "KVBatchMeta": SimpleNamespace,
        "create_rl_dataset": Mock(side_effect=[list(range(16)), list(range(3))]),
        "create_rl_sampler": Mock(),
        "StatefulDataLoader": Mock(),
        "tu": SimpleNamespace(
            get_tensordict=Batch,
            concat_tensordict=lambda chunks: Batch(
                {key: [item for chunk in chunks for item in chunk[key]] for key in chunks[0]}
            ),
            assign_non_tensor_data=lambda *args: None,
        ),
        "Tracking": Mock(),
        "ValidationGenerationsLogger": Mock(),
        "DapoFilteredRewardTableLogger": Mock(),
        "SkipManager": Mock(annotate_tq=lambda **kwargs: lambda fn: fn),
        "tq": replay_types["tq"],
        "_count_tq_prompt_groups": lambda: sum(
            tag.get("is_prompt", False) for tag in replay_types["tq"].tags["train"].values()
        ),
        "tqdm": Mock(),
        "pprint": Mock(),
        "marked_timer": lambda *args, **kwargs: contextlib.nullcontext(),
        "Counter": Counter,
        "defaultdict": defaultdict,
        "DAPO_FILTERED_REWARD_COUNTS_KEY": replay_types["DAPO_FILTERED_REWARD_COUNTS_KEY"],
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    metrics_path = ROOT / "verl/trainer/ppo/v1/utils.py"
    metrics_class = next(
        node
        for node in ast.parse(metrics_path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "MetricsAggregator"
    )
    metrics_tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            metrics_class,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(metrics_tree), str(metrics_path), "exec"), namespace)
    namespace["trainer_classes"] = {}
    namespace["PPOTrainer"] = namespace["Trainer"]
    namespace["HybridEngineMode"] = SimpleNamespace(ROLLOUT="rollout", TRAINER="trainer")
    for mode, name in (
        ("colocate_async", "PPOTrainerColocateAsync"),
        ("separate_async", "PPOTrainerSeparateAsync"),
    ):
        subclass_path = ROOT / f"verl/trainer/ppo/v1/trainer_{mode}.py"
        subclass_tree = ast.parse(subclass_path.read_text())
        subclass = next(node for node in subclass_tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        subclass.decorator_list = []
        subclass.body = [
            node
            for node in subclass.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("on_train_begin", "prepare_step", "_wait_for_sampleable_and_switch")
        ]
        exec(compile(ast.Module(body=[subclass], type_ignores=[]), str(subclass_path), "exec"), namespace)
        namespace["trainer_classes"][mode] = namespace[name]
    return namespace["Trainer"], namespace


def trainer_instance(trainer_class, cfg=None):
    cls, namespace = trainer_class
    trainer = cls()
    trainer.config = cfg if cfg is not None else config()
    trainer.trainer_mode = trainer.config.trainer.v1.trainer_mode
    trainer.parameter_sync_step = 2
    trainer.tokenizer = trainer.processor = None
    trainer.replay_buffer = namespace["ReplayBufferAsync"](
        trainer_mode=trainer.trainer_mode,
        trainer_config={},
        max_off_policy_threshold=None,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        refill_fn=trainer._add_prompts_to_generate,
    )
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
def test_sized_source_never_rewinds_in_either_scheduling_mode(trainer_class, total_steps):
    trainer, _ = trainer_instance(trainer_class, config(SizedLoader))
    trainer.config.trainer.total_training_steps = total_steps
    trainer._init_dataloader()
    for _ in range(6):
        trainer._fetch_one_gen_batch()
    with pytest.raises(bridge.DataLoaderExhausted):
        trainer._fetch_one_gen_batch()
    with pytest.raises(bridge.DataLoaderExhausted):
        trainer._fetch_one_gen_batch()
    assert trainer.train_dataloader.cursor == 6


def test_custom_loader_can_repeat_internally(trainer_class):
    trainer, _ = trainer_instance(trainer_class, config(RepeatingLoader))
    trainer._init_dataloader()
    positions = [trainer._fetch_one_gen_batch()["extra_info"][0]["position"] for _ in range(14)]
    assert positions == list(range(6)) * 2 + [0, 1]
    assert not trainer.replay_buffer.input_exhausted


def test_builtin_loader_still_restarts_at_eof(trainer_class):
    trainer, _ = trainer_instance(trainer_class)
    trainer.config.data.custom_dataloader = None
    trainer.train_dataloader = [{"raw_prompt": [["hello"]]}]
    trainer.train_dataloader_it = None
    for _ in range(3):
        assert trainer._fetch_one_gen_batch()["raw_prompt"] == [["hello"]]
    assert not trainer.replay_buffer.input_exhausted


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
def test_async_modes_use_the_effective_fetch_size(trainer_class, mode):
    trainer, _ = trainer_instance(trainer_class)
    trainer.trainer_mode = mode
    trainer._init_dataloader()
    assert trainer.train_dataloader.batch_size == 1
    assert trainer.train_dataloader.context.gen_batch_size == 1


def test_async_custom_sampler_must_support_draining(trainer_class):
    trainer, _ = trainer_instance(trainer_class)
    trainer.trainer_mode = "separate_async"
    trainer.replay_buffer = object()
    with pytest.raises(ValueError, match="ReplayBufferAsync"):
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


def streaming_fit_trainer(
    trainer_class,
    *,
    batches,
    initial_step=0,
    save_freq=-1,
    test_freq=-1,
    val_before_train=False,
    loader_class=FixtureLoader,
):
    trainer, _ = trainer_instance(trainer_class, config(loader_class))
    trainer.config.trainer.total_epochs = 0
    trainer.config.trainer.total_training_steps = initial_step + 3
    trainer.config.trainer.save_freq = save_freq
    trainer.config.trainer.test_freq = test_freq
    trainer.config.trainer.val_before_train = val_before_train
    trainer.config.data.custom_dataloader.arg2.batches = batches
    trainer._init_dataloader()
    loader = trainer.train_dataloader
    trainer.global_steps = initial_step
    events = []
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
        "on_validate_begin",
        "on_validate_end",
    ):
        setattr(trainer, name, Mock())
    trainer._stop_profiling.side_effect = lambda: events.append(("profile-stop", trainer.global_steps))
    trainer._consume_sync_metrics = Mock(return_value={})
    trainer._save_checkpoint = Mock(side_effect=lambda: events.append(("save", trainer.global_steps)))

    def validate():
        events.append(("validate", trainer.global_steps))
        return {"validation/score": 1.0}

    trainer._validate = Mock(side_effect=validate)

    def step(*args):
        for _ in range(trainer.config.data.train_batch_size // loader.batch_size):
            trainer._fetch_one_gen_batch()
        return SimpleNamespace(keys=[], partition_id="train")

    trainer.step = step
    return trainer, loader, events


@pytest.mark.parametrize("batches,expected_steps", [(6, 3), (2, 1)])
def test_step_mode_ignores_epochs_and_stops_cleanly_without_rewinding(trainer_class, batches, expected_steps):
    trainer, loader, _ = streaming_fit_trainer(trainer_class, batches=batches)
    trainer.fit(None)
    assert trainer._compute_metrics.call_count == expected_steps
    assert all(call.kwargs["epoch"] is None for call in trainer._compute_metrics.call_args_list)
    assert loader.cursor == batches
    assert loader.closed == 1


@pytest.mark.parametrize(
    "save_freq,test_freq,initial_step,saved,validated",
    [
        (10, 10, 0, [2], [2]),
        (2, 2, 0, [2], [2]),
        (1, 1, 0, [1, 2], [1, 2]),
        (1, 10, 0, [1, 2], [2]),
        (10, 1, 0, [2], [1, 2]),
        (-1, -1, 0, [], []),
        (0, 0, 0, [], []),
        (10, 10, 5, [7], [7]),
        (3, 3, 5, [6, 7], [6, 7]),
    ],
)
def test_early_exhaustion_finalizes_only_unhandled_completed_steps(
    trainer_class, save_freq, test_freq, initial_step, saved, validated
):
    trainer, loader, events = streaming_fit_trainer(
        trainer_class, batches=4, initial_step=initial_step, save_freq=save_freq, test_freq=test_freq
    )
    trainer.fit(None)
    assert [step for event, step in events if event == "save"] == saved
    assert [step for event, step in events if event == "validate"] == validated
    assert trainer.global_steps == initial_step + 3
    assert loader.cursor == 4 and loader.closed == 1
    assert trainer._compute_metrics.call_count == 2
    trainer.on_train_end.assert_called_once()
    if save_freq == 10:
        assert events.index(("profile-stop", initial_step + 3)) < events.index(("save", initial_step + 2))
    if test_freq == 10:
        assert events.index(("profile-stop", initial_step + 3)) < events.index(("validate", initial_step + 2))
        assert trainer.logger.log.call_args.kwargs == {
            "data": {"validation/score": 1.0},
            "step": initial_step + 2,
        }


@pytest.mark.parametrize("initial_step", [0, 5])
@pytest.mark.parametrize("val_before_train", [False, True])
def test_empty_source_does_not_finalize_an_unperformed_update(trainer_class, initial_step, val_before_train):
    trainer, loader, events = streaming_fit_trainer(
        trainer_class,
        batches=0,
        initial_step=initial_step,
        save_freq=10,
        test_freq=10,
        val_before_train=val_before_train,
    )
    trainer.fit(None)
    trainer._save_checkpoint.assert_not_called()
    assert [step for event, step in events if event == "validate"] == ([initial_step] if val_before_train else [])
    trainer._compute_metrics.assert_not_called()
    assert loader.closed == 1


@pytest.mark.parametrize("operation", ["save", "validate"])
def test_early_finalization_failure_restores_step_and_closes_loaders(trainer_class, operation):
    trainer, loader, events = streaming_fit_trainer(trainer_class, batches=2, save_freq=10, test_freq=10)
    val_loader = trainer.val_dataloader

    def fail():
        assert trainer.global_steps == 1
        raise RuntimeError(f"{operation} failed")

    getattr(trainer, "_save_checkpoint" if operation == "save" else "_validate").side_effect = fail
    with pytest.raises(RuntimeError, match=f"{operation} failed"):
        trainer.fit(None)
    assert trainer.global_steps == 2
    assert trainer_class[1]["SkipManager"].set_step.call_args.args == (2,)
    assert loader.closed == val_loader.closed == 1
    trainer.on_train_end.assert_not_called()


def test_partial_ppo_update_failure_does_not_trigger_final_checkpoint(trainer_class):
    trainer, loader, _ = streaming_fit_trainer(trainer_class, batches=2, save_freq=10, test_freq=10)
    trainer.step = type(trainer).step.__get__(trainer)
    trainer.prepare_step = Mock(return_value={})
    trainer._step_once = Mock(
        side_effect=[
            SimpleNamespace(keys=["first"], tags=[{}], partition_id="train"),
            bridge.DataLoaderExhausted(),
        ]
    )
    with pytest.raises(RuntimeError, match="during PPO update/refill"):
        trainer.fit(None)
    trainer._save_checkpoint.assert_not_called()
    trainer._validate.assert_not_called()
    assert loader.closed == 1


def test_step_budget_finalization_does_not_probe_or_repeat(trainer_class):
    trainer, loader, events = streaming_fit_trainer(trainer_class, batches=8, save_freq=10, test_freq=10)
    trainer.fit(None)
    assert [step for event, step in events if event == "save"] == [3]
    assert [step for event, step in events if event == "validate"] == [3]
    assert loader.cursor == 6 and loader.closed == 1


@pytest.mark.parametrize("loader_class", [FixtureLoader, IterableOnly])
def test_early_exhaustion_writes_the_completed_checkpoint_directory(trainer_class, tmp_path, monkeypatch, loader_class):
    trainer, _, _ = streaming_fit_trainer(trainer_class, batches=2, save_freq=10, loader_class=loader_class)
    with open_dict(trainer.config.trainer):
        trainer.config.trainer.update(default_local_dir=str(tmp_path), default_hdfs_dir=None)
    trainer.use_critic = False
    trainer.actor_rollout_wg = Mock()
    trainer.checkpoint_callback = Mock()
    monkeypatch.setitem(
        sys.modules,
        "verl.utils.fs",
        SimpleNamespace(local_mkdir_safe=lambda path: Path(path).mkdir(parents=True, exist_ok=True)),
    )
    trainer._save_checkpoint = type(trainer)._save_checkpoint.__get__(trainer)
    trainer.fit(None)
    assert (tmp_path / "global_step_1").is_dir()
    assert (tmp_path / "global_step_1" / "data.pt").is_file() == (loader_class is FixtureLoader)
    assert not (tmp_path / "global_step_2").exists()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "1"
    assert trainer.actor_rollout_wg.save_checkpoint.call_args.args[2] == 1


def test_documentation_links_to_the_existing_configuration_guide():
    config_doc = (ROOT / "docs/examples/config.rst").read_text()
    assert ".. _custom-ppo-dataloader:" in config_doc
    assert "_target_: my_project.StreamingLoader" in config_doc
    assert (
        ":ref:`custom dataloader <custom-ppo-dataloader>`" in (ROOT / "docs/preparation/prepare_data.rst").read_text()
    )
    assert "advance/custom_dataloader" not in (ROOT / "docs/index.rst").read_text()


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


def finish_prompt(queue, uid, *, status="finished"):
    tag = queue.tags["train"][uid]
    tag["status"] = status
    if status == "finished":
        queue.tags["train"][f"{uid}_0_0"] = {"is_prompt": False, "global_steps": tag["global_steps"]}


def async_fit_trainer(trainer_class, *, mode, batches, warmup=1, sync_steps=2):
    subclass = trainer_class[1]["trainer_classes"][mode]
    trainer, loader, events = streaming_fit_trainer(
        (subclass, trainer_class[1]), batches=batches, save_freq=10, test_freq=10
    )
    trainer.trainer_mode = mode
    trainer.parameter_sync_step = sync_steps
    with open_dict(trainer.config):
        trainer.config.trainer.v1.trainer_mode = mode
        trainer.config.trainer.v1[mode] = {"num_warmup_batches": warmup}
        trainer.config.skip = {"rollout_tq": {"enable": False}}
    # Construct with the async effective fetch size, before any reads.
    trainer.close_dataloaders()
    trainer._init_dataloader()
    loader = trainer.train_dataloader
    trainer._restored_tq_prompt_count = 0
    trainer.timing_raw = {}
    trainer.step = type(trainer).step.__get__(trainer)
    trainer.on_train_begin = subclass.on_train_begin.__get__(trainer)
    trainer.current_mode = "rollout"
    trainer._step_threshold = 2
    trainer._timed_switch_to_trainer = lambda: setattr(trainer, "current_mode", "trainer")
    queue = trainer_class[1]["tq"]
    submitted = []
    trained = []

    def generate(batch):
        submitted.extend(batch["uid"])
        for uid in batch["uid"]:
            finish_prompt(queue, uid)

    def update(metrics, timing, batch_size):
        batch, replay_metrics = trainer.replay_buffer.sample(trainer.global_steps, "train", batch_size)
        metrics.update(replay_metrics)
        trained.extend(key.split("_")[0] for key in batch.keys)
        return batch

    trainer._step_once = update
    manager = SimpleNamespace(generate_sequences=generate)
    return trainer, loader, manager, submitted, trained, events


@pytest.mark.parametrize("mode", ["colocate_async", "separate_async"])
@pytest.mark.parametrize("warmup", [0, 2])
@pytest.mark.parametrize("sync_steps", [1, 4])
@pytest.mark.parametrize("batches", [0, 3, 4, 5, 9, 20])
def test_async_drains_full_cycles_without_partial_updates(trainer_class, mode, warmup, sync_steps, batches, caplog):
    trainer, loader, manager, submitted, trained, events = async_fit_trainer(
        trainer_class, mode=mode, batches=batches, warmup=warmup, sync_steps=sync_steps
    )
    with caplog.at_level(logging.INFO):
        trainer.fit(manager)

    expected_steps = min(3, batches // 4)
    assert trainer._compute_metrics.call_count == expected_steps
    assert len(trained) == len(set(trained)) == expected_steps * 4
    assert set(trained) <= set(submitted)
    assert len(submitted) == loader.cursor
    assert loader.closed == 1
    assert [step for event, step in events if event == "save"] == ([expected_steps] if expected_steps else [])
    assert [step for event, step in events if event == "validate"] == ([expected_steps] if expected_steps else [])
    if batches < 12:
        assert not trainer_class[1]["tq"].tags["train"]
        assert f"discarding {batches % 4} remaining prompt groups" in caplog.text


@pytest.mark.parametrize("status", ["finished", "failure"])
@pytest.mark.parametrize("strategy", ["drop", "wait"])
def test_async_eof_waits_for_inflight_groups_before_deciding_to_stop(trainer_class, replay_types, status, strategy):
    trainer, _, manager, _, trained, _ = async_fit_trainer(trainer_class, mode="colocate_async", batches=4, warmup=2)
    trainer.agent_loop_manager = manager
    trainer.global_steps = 1
    trainer._add_async_warmup_batches(2)
    assert trainer.replay_buffer.input_exhausted
    queue = trainer_class[1]["tq"]
    uid = next(key for key, tag in queue.tags["train"].items() if tag["is_prompt"])
    queue.kv_clear(keys=[f"{uid}_0_0"], partition_id="train")
    queue.tags["train"][uid]["status"] = "running"
    trainer.replay_buffer.max_off_policy_threshold = 1
    trainer.replay_buffer.max_off_policy_strategy = strategy
    replay_types["time"].sleep.side_effect = lambda _: finish_prompt(queue, uid, status=status)

    if status == "finished":
        trainer.step({}, {})
        assert len(trained) == 4
    else:
        with pytest.raises(bridge.DataLoaderExhausted):
            trainer.step({}, {})
        assert trained == []
        assert queue.tags["train"] == {}
    replay_types["time"].sleep.assert_called_once()


@pytest.mark.parametrize("reason", ["stale", "failure", "filtered"])
def test_async_eof_evictions_do_not_refetch_or_wait_forever(trainer_class, reason):
    trainer, loader, manager, _, trained, _ = async_fit_trainer(
        trainer_class, mode="separate_async", batches=5, warmup=2
    )
    trainer.agent_loop_manager = manager
    trainer.global_steps = 1
    trainer._add_async_warmup_batches(2)
    assert trainer.replay_buffer.input_exhausted
    queue = trainer_class[1]["tq"]
    uids = [key for key, tag in queue.tags["train"].items() if tag["is_prompt"]]
    for uid in uids[:2]:
        if reason == "stale":
            queue.tags["train"][uid]["global_steps"] = 0
            trainer.replay_buffer.max_off_policy_threshold = 1
        elif reason == "failure":
            queue.tags["train"][uid]["status"] = "failure"
    if reason == "filtered":
        trainer.replay_buffer.filter_groups_metric = "reward"
        for index, uid in enumerate(uids):
            for session, reward in enumerate([1.0, 1.0 if index < 2 else 0.0]):
                key = f"{uid}_{session}_0"
                queue.tags["train"][key] = {"is_prompt": False}
                queue.fields["train"][key] = {"rm_scores": np.array([reward])}

    with pytest.raises(bridge.DataLoaderExhausted):
        trainer.step({}, {})
    assert trained == []
    assert loader.cursor == 5
    assert queue.tags["train"] == {}


def test_async_partial_refill_submits_the_tail_and_trains_a_full_cycle(trainer_class):
    trainer, loader, manager, submitted, trained, _ = async_fit_trainer(
        trainer_class, mode="separate_async", batches=6, warmup=0
    )
    trainer.agent_loop_manager = manager
    trainer.global_steps = 1
    trainer._add_prompts_to_generate(5)
    queue = trainer_class[1]["tq"]
    for uid in submitted[:2]:
        queue.tags["train"][uid]["status"] = "failure"
    sampleable, metrics = trainer.replay_buffer.wait_for_sampleable(1, "train", 4)
    assert len(sampleable) == 4
    assert metrics["training/rollout_failure/evicted_samples"] == 2
    assert trainer.replay_buffer.input_exhausted
    assert loader.cursor == 6
    assert len(submitted) == 6

    trainer.step({}, {})
    assert len(trained) == 4
    assert submitted[-1] in trained
    assert not set(submitted[:2]) & set(trained)


def test_training_eof_does_not_end_validation_sampling(trainer_class, replay_types):
    trainer, _ = trainer_instance(trainer_class)
    trainer.replay_buffer.input_exhausted = True
    queue = trainer_class[1]["tq"]
    queue.tags["val"]["first"] = {"is_prompt": True, "status": "finished", "global_steps": 1}
    replay_types["time"].sleep.side_effect = lambda _: queue.tags["val"].update(
        second={"is_prompt": True, "status": "finished", "global_steps": 1}
    )
    sampleable, _ = trainer.replay_buffer.wait_for_sampleable(1, "val", 2)
    assert sampleable == {"first", "second"}


def test_async_loader_errors_propagate_and_close(trainer_class):
    trainer, loader, manager, _, _, _ = async_fit_trainer(trainer_class, mode="separate_async", batches=10, warmup=0)

    def failing_source():
        yield next(iter(loader))
        raise ValueError("source disconnected")

    trainer.train_dataloader_it = failing_source()
    with pytest.raises(ValueError, match="source disconnected"):
        trainer.fit(manager)
    assert not trainer.replay_buffer.input_exhausted
    assert loader.closed == 1


def checkpoint_trainer(trainer_class, tmp_path, monkeypatch, loader_class, mode="sync"):
    cfg = config(loader_class)
    cfg.trainer.v1.trainer_mode = mode
    trainer, namespace = trainer_instance(trainer_class, cfg)
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


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_nonresumable_loader_saves_model_and_removes_stale_data_state(trainer_class, tmp_path, monkeypatch, mode):
    trainer, _ = checkpoint_trainer(trainer_class, tmp_path, monkeypatch, IterableOnly, mode)
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
@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_nonresumable_loader_rejects_restore_before_loading_weights(
    trainer_class, tmp_path, monkeypatch, resume_mode, mode
):
    trainer, namespace = checkpoint_trainer(trainer_class, tmp_path, monkeypatch, IterableOnly, mode)
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


@pytest.mark.parametrize("mode", ["colocate_async", "separate_async"])
def test_async_resume_restores_cursor_and_inflight_prompts_without_duplicate_warmup(
    trainer_class, tmp_path, monkeypatch, mode
):
    trainer, namespace = checkpoint_trainer(trainer_class, tmp_path, monkeypatch, FixtureLoader, mode)
    with open_dict(trainer.config):
        trainer.config.skip = {"rollout_tq": {"enable": False}}
    trainer.agent_loop_manager = Mock()
    trainer._add_prompts_to_generate(3)
    queue = namespace["tq"]
    uids = set(queue.tags["train"])
    finish_prompt(queue, next(iter(uids)))
    trainer._save_checkpoint()
    checkpoint = str(tmp_path / "global_step_1")

    # A fresh trainer restores both components before reissuing queued work.
    restored, _ = trainer_instance(trainer_class, trainer.config)
    restored._init_dataloader()
    restored.use_critic = False
    restored.actor_rollout_wg = Mock()
    restored.agent_loop_manager = Mock()
    namespace["find_latest_ckpt_path"].return_value = checkpoint
    restored._load_checkpoint()
    assert restored._restored_tq_prompt_count == 3
    assert restored.train_dataloader.cursor == 3
    assert restored._reissue_inflight_prompts() == 2
    restored._add_async_warmup_batches(1)
    assert restored.train_dataloader.cursor == 4
    assert sum(tag.get("is_prompt", False) for tag in queue.tags["train"].values()) == 4
    calls = restored.agent_loop_manager.generate_sequences.call_args_list
    assert len(calls) == 2
    assert set(calls[0].args[0]["uid"]) < uids
    assert calls[1].args[0]["extra_info"][0]["position"] == 3
    assert restored._fetch_one_gen_batch()["extra_info"][0]["position"] == 4
