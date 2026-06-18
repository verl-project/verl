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

"""Unit tests for ``verl.utils.skip`` (SkipManager, RolloutSkip, config, registry)."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.skip.base_skip import SKIP_REGISTRY, BaseSkip, SkipAction, register_skip
from verl.utils.skip.config import AsyncRolloutSkipConfig, RolloutSkipConfig, SkipManagerConfig
from verl.utils.skip.rollout_skip import AsyncRolloutSkip, RolloutSkip, parse_async_rollout_sample_step
from verl.utils.skip.skip_manager import SkipManager


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def _reset_skip_manager_class_state() -> None:
    SkipManager.config = None  # type: ignore[attr-defined]
    SkipManager.step = -1  # type: ignore[attr-defined]
    SkipManager.skip_instances = {}  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_skip_manager():
    _reset_skip_manager_class_state()
    yield
    _reset_skip_manager_class_state()


def _minimal_skip_cfg(
    dump_dir: str,
    *,
    enable: bool = True,
    steps: list[int] | None = None,
    action: str = "cache",
    async_enable: bool = False,
) -> OmegaConf:
    steps = steps if steps is not None else [1]
    return OmegaConf.create(
        {
            "skip": {
                "rollout": {
                    "enable": enable,
                    "dump_dir": dump_dir,
                    "steps": steps,
                    "action": action,
                },
                "async_rollout": {
                    "enable": async_enable,
                    "dump_dir": dump_dir,
                    "steps": steps,
                    "action": action,
                },
            },
            "actor_rollout_ref": {"rollout": {"skip": {"enable": False}, "n": 2}},
            "trainer": {"experiment_name": "ut_exp", "project_name": "ut_proj"},
            "data": {"gen_batch_size": 4, "max_prompt_length": 8, "max_response_length": 16},
        }
    )


def _local_rollout_config(cfg: OmegaConf) -> RolloutSkipConfig:
    return omega_conf_to_dataclass(cfg.skip.rollout, RolloutSkipConfig)


def _project_dump_root(dump_dir: Path, cfg: OmegaConf) -> Path:
    exp = cfg.trainer.experiment_name
    proj = cfg.trainer.project_name
    gbs = cfg.data.gen_batch_size
    n = int(OmegaConf.select(cfg, "actor_rollout_ref.rollout.n", default=0))
    inp = cfg.data.max_prompt_length
    out = cfg.data.max_response_length
    sub = f"{exp}_{proj}/GBS{gbs}_N{n}_in{inp}_out{out}"
    return dump_dir.joinpath(sub).resolve()


def _write_valid_step_dump(project_root: Path, step: int, proto: DataProto) -> Path:
    step_dir = project_root.joinpath(str(step))
    step_dir.mkdir(parents=True, exist_ok=True)
    proto.save_to_disk(step_dir.joinpath("gen_batch.dp"))
    step_dir.joinpath("meta.json").write_text(json.dumps({"global_steps": step}), encoding="utf-8")
    return step_dir


class TestRolloutSkipConfig:
    def test_defaults(self):
        c = RolloutSkipConfig()
        assert c.enable is False
        assert c.action == "cache"
        assert c.steps == []

    def test_async_rollout_config_defaults(self):
        c = AsyncRolloutSkipConfig()
        assert c.enable is False
        assert c.action == "cache"

    def test_invalid_action(self):
        with pytest.raises(AssertionError, match="action"):
            RolloutSkipConfig(action="not_an_action")

    def test_steps_must_be_int(self):
        with pytest.raises(AssertionError, match="steps"):
            RolloutSkipConfig(steps=[1, "x"])  # type: ignore[list-item]


class TestSkipRegistryAndBaseSkip:
    def test_rollout_and_async_rollout_registered(self):
        assert "rollout" in SKIP_REGISTRY
        assert "async_rollout" in SKIP_REGISTRY
        assert SKIP_REGISTRY["async_rollout"] is AsyncRolloutSkip

    def test_register_skip_adds_class(self):
        name = f"ut_dummy_skip_{uuid.uuid4().hex[:8]}"

        @register_skip(name)
        class _UtSkip(BaseSkip):
            support_actions = [SkipAction.EMPTY]

            def meet_precondition(self, step: int, func, *args, **kwargs) -> bool:
                return True

            def warp_function(self, step: int, func, *args, **kwargs):
                return "warped"

            def prepare_data(self, step: int, result, *args, **kwargs):
                pass

        assert SKIP_REGISTRY[name] is _UtSkip
        del SKIP_REGISTRY[name]

    def test_base_skip_rejects_unsupported_action(self):
        class _Bad(BaseSkip):
            support_actions = [SkipAction.CACHE]

        with pytest.raises(ValueError, match="Unsupported action"):
            _Bad(
                RolloutSkipConfig(enable=True, action="repeat", steps=[1]),
                OmegaConf.create({}),
            )


class TestParseAsyncRolloutSampleStep:
    def test_valid_sample_id(self):
        assert parse_async_rollout_sample_step("sample_0_42") == 42
        assert parse_async_rollout_sample_step("sample_3_1") == 1

    def test_valid_uid_prefixed_sample_id(self):
        assert parse_async_rollout_sample_step("uid_sample_0_42") == 42
        assert parse_async_rollout_sample_step("uid_sample_3_1") == 1

    def test_invalid_sample_id(self):
        with pytest.raises(ValueError, match="Invalid async rollout sample_id"):
            parse_async_rollout_sample_step("bad_id")


class TestRolloutSkipPaths:
    def test_check_valid_step_path(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), enable=True, steps=[1], action="cache")
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        root = _project_dump_root(tmp_path, cfg)

        assert rs._check_valid_step_path(root.joinpath("99")) is False

        proto = DataProto.from_dict(tensors={"x": torch.zeros(1)})
        _write_valid_step_dump(root, 7, proto)
        assert rs._check_valid_step_path(root.joinpath("7")) is True

    def test_get_available_steps_filters_invalid_dirs(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path))
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        root = _project_dump_root(tmp_path, cfg)
        root.mkdir(parents=True)

        (root / "not_int").mkdir()
        (root / "2").mkdir()
        proto = DataProto.from_dict(tensors={"x": torch.ones(1)})
        _write_valid_step_dump(root, 1, proto)
        _write_valid_step_dump(root, 10, proto)

        assert rs._get_available_steps() == [1, 10]

    def test_find_latest_step_exact_then_smaller_then_larger(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path))
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        root = _project_dump_root(tmp_path, cfg)
        proto = DataProto.from_dict(tensors={"x": torch.tensor([2.0])})

        _write_valid_step_dump(root, 5, proto)
        _write_valid_step_dump(root, 20, proto)

        assert rs._find_latest_step(5) == 5
        assert rs._find_latest_step(12) == 5
        assert rs._find_latest_step(3) == 5

        shutil.rmtree(root)
        root.mkdir(parents=True)
        assert rs._find_latest_step(100) == -1


class TestRolloutSkipMeetWarpPrepare:
    def test_meet_precondition_cache_miss_and_hit(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), action="cache")
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        root = _project_dump_root(tmp_path, cfg)

        assert rs.meet_precondition(1, _noop) is False

        proto = DataProto.from_dict(tensors={"t": torch.arange(3)})
        _write_valid_step_dump(root, 1, proto)
        assert rs.meet_precondition(1, _noop) is True

    def test_meet_precondition_repeat(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), action="repeat")
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        root = _project_dump_root(tmp_path, cfg)
        proto = DataProto.from_dict(tensors={"t": torch.tensor([1.0])})

        assert rs.meet_precondition(2, _noop) is False

        _write_valid_step_dump(root, 1, proto)
        assert rs.meet_precondition(2, _noop) is True

    def test_prepare_data_and_warp_roundtrip(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), action="cache")
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        original = DataProto.from_dict(tensors={"k": torch.tensor([[1.0, 2.0]])})
        rs.prepare_data(3, original)

        loaded = rs.warp_function(3, _noop)
        assert torch.allclose(loaded.batch["k"], original.batch["k"])

    def test_warp_function_repeat_resolves_step_per_call(self, tmp_path: Path):
        """Each repeat warp call must resolve its own substitute step (no shared instance cache)."""
        cfg = _minimal_skip_cfg(str(tmp_path), action="repeat")
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        root = _project_dump_root(tmp_path, cfg)
        _write_valid_step_dump(root, 1, DataProto.from_dict(tensors={"t": torch.tensor([1.0])}))
        _write_valid_step_dump(root, 5, DataProto.from_dict(tensors={"t": torch.tensor([5.0])}))

        assert rs.meet_precondition(12, _noop) is True
        loaded_small = rs.warp_function(3, _noop)
        loaded_large = rs.warp_function(12, _noop)
        assert loaded_small.batch["t"].item() == 1.0
        assert loaded_large.batch["t"].item() == 5.0

    def test_warp_function_repeat_without_meet_precondition(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), action="repeat")
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        root = _project_dump_root(tmp_path, cfg)
        proto = DataProto.from_dict(tensors={"t": torch.tensor([7.0])})
        _write_valid_step_dump(root, 7, proto)

        loaded = rs.warp_function(99, _noop)
        assert loaded.batch["t"].item() == 7.0

    def test_warp_function_repeat_raises_when_no_dump(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), action="repeat")
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        with pytest.raises(RuntimeError, match="repeat action expected dumped data"):
            rs.warp_function(1, _noop)


class TestAsyncRolloutSkipExtractStep:
    def test_extract_step_from_prompts_uid(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), async_enable=True)
        local = omega_conf_to_dataclass(cfg.skip.async_rollout, AsyncRolloutSkipConfig)
        ars = AsyncRolloutSkip(local, cfg)
        prompts = DataProto.from_dict(tensors={"x": torch.zeros(1)})
        prompts.non_tensor_batch["uid"] = np.array(["uid_sample_1_7"], dtype=object)
        assert ars.extract_step(object(), prompts) == 7

    def test_extract_step_from_kwargs_prompts(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), async_enable=True)
        local = omega_conf_to_dataclass(cfg.skip.async_rollout, AsyncRolloutSkipConfig)
        ars = AsyncRolloutSkip(local, cfg)
        prompts = DataProto.from_dict(tensors={"x": torch.zeros(1)})
        prompts.non_tensor_batch["uid"] = np.array(["uid_sample_0_1"], dtype=object)
        assert ars.extract_step(prompts=prompts) == 1

    def test_extract_step_missing_uid(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), async_enable=True)
        local = omega_conf_to_dataclass(cfg.skip.async_rollout, AsyncRolloutSkipConfig)
        ars = AsyncRolloutSkip(local, cfg)
        prompts = DataProto.from_dict(tensors={"x": torch.zeros(1)})
        with pytest.raises(ValueError, match="uid"):
            ars.extract_step(object(), prompts)


class TestSkipManagerInitAndAnnotate:
    def test_init_builds_rollout_and_async_instances(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), steps=[1, 2], async_enable=True)
        SkipManager.init(cfg)
        assert SkipManager.config is not None
        assert "rollout" in SkipManager.skip_instances
        assert "async_rollout" in SkipManager.skip_instances
        rollout = SkipManager.skip_instances["rollout"]
        assert rollout.is_enabled() is True
        assert rollout.steps == [1, 2]
        async_inst = SkipManager.skip_instances["async_rollout"]
        assert async_inst.support_online_step is True

    def test_annotate_sync_bypass_when_step_not_in_steps(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), enable=True, steps=[99])
        SkipManager.init(cfg)

        @SkipManager.annotate(role="rollout")
        def work(x: int) -> int:
            return x + 1

        SkipManager.set_step(1)
        assert work(40) == 41

    def test_should_bypass_for_validation(self):
        batch = DataProto.from_dict(tensors={"x": torch.zeros(1)})
        batch.meta_info = {"validate": True}
        assert SkipManager._should_bypass_for_validation((object(), batch), {}) is True
        batch.meta_info = {"validate": False}
        assert SkipManager._should_bypass_for_validation((object(), batch), {}) is False

    def test_annotate_sync_validation_bypasses_cache_and_dump(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), enable=True, steps=[1], action="cache")
        SkipManager.init(cfg)
        root = _project_dump_root(tmp_path, cfg)
        train_proto = DataProto.from_dict(tensors={"z": torch.tensor([99.0])})
        _write_valid_step_dump(root, 1, train_proto)

        @SkipManager.annotate(role="rollout")
        def gen(_self: Any, prompts: DataProto) -> DataProto:
            return DataProto.from_dict(tensors={"z": torch.tensor([42.0])})

        SkipManager.set_step(1)
        val_batch = DataProto.from_dict(tensors={"z": torch.tensor([0.0])})
        val_batch.meta_info = {"validate": True}
        out = gen(None, val_batch)
        assert out.batch["z"].item() == 42.0
        loaded = DataProto.load_from_disk(root / "1" / "gen_batch.dp")
        assert loaded.batch["z"].item() == 99.0

    def test_annotate_async_validation_bypasses_cache_and_dump(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), enable=True, steps=[1], action="cache")
        SkipManager.init(cfg)
        root = _project_dump_root(tmp_path, cfg)
        train_proto = DataProto.from_dict(tensors={"z": torch.tensor([99.0])})
        _write_valid_step_dump(root, 1, train_proto)

        @SkipManager.annotate(role="rollout")
        async def gen(_self: Any, prompts: DataProto) -> DataProto:
            return DataProto.from_dict(tensors={"z": torch.tensor([42.0])})

        async def _run():
            SkipManager.set_step(1)
            val_batch = DataProto.from_dict(tensors={"z": torch.tensor([0.0])})
            val_batch.meta_info = {"validate": True}
            out = await gen(None, val_batch)
            assert out.batch["z"].item() == 42.0
            loaded = DataProto.load_from_disk(root / "1" / "gen_batch.dp")
            assert loaded.batch["z"].item() == 99.0

        asyncio.run(_run())

    def test_annotate_sync_cache_step_one(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), enable=True, steps=[1], action="cache")
        SkipManager.init(cfg)
        root = _project_dump_root(tmp_path, cfg)

        @SkipManager.annotate(role="rollout")
        def gen(_: Any = None) -> DataProto:
            return DataProto.from_dict(tensors={"z": torch.tensor([7.0])})

        SkipManager.set_step(1)
        out = gen()
        assert out.batch["z"].item() == 7.0
        assert (root / "1" / "gen_batch.dp").exists()

        SkipManager.set_step(1)

        @SkipManager.annotate(role="rollout")
        def gen_cached(_: Any = None) -> DataProto:
            raise AssertionError("should not run when cache hit")

        loaded = gen_cached()
        assert torch.allclose(loaded.batch["z"], torch.tensor([7.0]))

    def test_annotate_async_rollout_online_step(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), enable=False, async_enable=True, steps=[1], action="cache")
        SkipManager.init(cfg)
        root = _project_dump_root(tmp_path, cfg)

        def _make_prompts(sample_id: str) -> DataProto:
            p = DataProto.from_dict(tensors={"x": torch.zeros(1)})
            p.non_tensor_batch["uid"] = np.array([f"uid_{sample_id}"], dtype=object)
            return p

        @SkipManager.annotate(role="async_rollout")
        async def gen_single(_self: Any, prompts: DataProto) -> DataProto:
            # simulate reading sample_id: strip "uid_" prefix then parse
            raw = str(prompts.non_tensor_batch["uid"][0])
            sample_id = raw[4:] if raw.startswith("uid_") else raw
            return DataProto.from_dict(tensors={"a": torch.tensor([float(sample_id.split("_")[-1])])})

        async def _run():
            out = await gen_single(None, _make_prompts("sample_0_1"))
            assert out.batch["a"].item() == 1.0
            assert (root / "1" / "gen_batch.dp").exists()

            @SkipManager.annotate(role="async_rollout")
            async def gen_cached(_self: Any, prompts: DataProto) -> DataProto:
                raise AssertionError("cached path")

            loaded = await gen_cached(None, _make_prompts("sample_0_1"))
            assert loaded.batch["a"].item() == 1.0

        asyncio.run(_run())

    def test_annotate_async_bypass_when_step_not_in_list(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), async_enable=True, steps=[99], action="cache")
        SkipManager.init(cfg)
        calls = {"n": 0}

        @SkipManager.annotate(role="async_rollout")
        async def gen(_self: Any, prompts: DataProto) -> DataProto:
            calls["n"] += 1
            return prompts

        async def _run():
            p = DataProto.from_dict(tensors={"x": torch.zeros(1)})
            p.non_tensor_batch["uid"] = np.array(["uid_sample_0_1"], dtype=object)
            out = await gen(None, p)
            assert out.non_tensor_batch["uid"][0] == "uid_sample_0_1"
            assert calls["n"] == 1

        asyncio.run(_run())


class TestSkipManagerConfigDataclass:
    def test_skip_manager_config_merge(self):
        c = SkipManagerConfig()
        assert isinstance(c.rollout, RolloutSkipConfig)
        assert isinstance(c.async_rollout, AsyncRolloutSkipConfig)
        assert c.rollout.enable is False


class TestSkipManagerRuntimeScenarios:
    def test_annotate_unknown_role_is_noop(self, tmp_path: Path):
        cfg = _minimal_skip_cfg(str(tmp_path), enable=True, steps=[1])
        SkipManager.init(cfg)
        SkipManager.set_step(1)

        @SkipManager.annotate(role="unknown_role")
        def f(x: int) -> int:
            return x * 2

        assert f(3) == 6


class TestSkipDumpDiskScenarios:
    def test_dump_dirs_are_isolated_by_config(self, tmp_path: Path):
        dump_a = tmp_path / "disk_a"
        dump_b = tmp_path / "disk_b"
        cfg_a = _minimal_skip_cfg(str(dump_a), enable=True, steps=[1], action="cache")
        cfg_b = _minimal_skip_cfg(str(dump_b), enable=True, steps=[1], action="cache")
        rs_a = RolloutSkip(_local_rollout_config(cfg_a), cfg_a)
        rs_b = RolloutSkip(_local_rollout_config(cfg_b), cfg_b)

        rs_a.prepare_data(1, DataProto.from_dict(tensors={"x": torch.tensor([1.0])}))
        rs_b.prepare_data(1, DataProto.from_dict(tensors={"x": torch.tensor([2.0])}))

        loaded_a = rs_a.warp_function(1, _noop)
        loaded_b = rs_b.warp_function(1, _noop)
        assert loaded_a.batch["x"].item() == 1.0
        assert loaded_b.batch["x"].item() == 2.0

    def test_prepare_data_handles_disk_write_error_without_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cfg = _minimal_skip_cfg(str(tmp_path), enable=True, steps=[1], action="cache")
        rs = RolloutSkip(_local_rollout_config(cfg), cfg)

        def _raise_save(*args, **kwargs):
            raise OSError("simulated disk write failure")

        monkeypatch.setattr(DataProto, "save_to_disk", _raise_save)
        rs.prepare_data(1, DataProto.from_dict(tensors={"x": torch.tensor([1.0])}))
        dump_file = rs._get_step_dump_dir(1).joinpath("gen_batch.dp")
        assert dump_file.exists() is False
