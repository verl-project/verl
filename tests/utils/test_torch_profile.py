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

import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.profiler.config import ProfilerConfig, TorchProfilerScheduleConfig, TorchProfilerToolConfig
from verl.utils.profiler.profile import DistProfiler, _NoOpProfiler
from verl.utils.profiler.torch_profile import (
    Profiler,
    build_trace_basename,
    get_torch_profiler,
)


class TestTorchProfile(unittest.TestCase):
    def setUp(self):
        # Reset process-global Profiler class state so tests don't leak into each other.
        Profiler._define_count = 0
        Profiler._active_prof = None

    def tearDown(self):
        Profiler._define_count = 0
        Profiler._active_prof = None

    @patch("torch.profiler.profile")
    def test_get_torch_profiler(self, mock_profile):
        # Test wrapper function
        get_torch_profiler(contents=["cpu", "cuda", "stack"], save_path="/tmp/test", rank=0)
        mock_profile.assert_called_once()
        _, kwargs = mock_profile.call_args

        # Verify activities
        activities = kwargs["activities"]
        self.assertIn(torch.profiler.ProfilerActivity.CPU, activities)
        self.assertIn(torch.profiler.ProfilerActivity.CUDA, activities)

        # Verify options
        self.assertTrue(kwargs["with_stack"])
        self.assertFalse(kwargs["record_shapes"])
        self.assertFalse(kwargs["profile_memory"])

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_profiler_lifecycle(self, mock_get_profiler):
        # Mock the underlying torch profiler object
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        # Initialize
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        # Test Start
        profiler.start()
        mock_get_profiler.assert_called_once()
        mock_prof_instance.start.assert_called_once()

        # Test Step
        profiler.step()
        mock_prof_instance.step.assert_called_once()

        # Test Stop
        profiler.stop()
        mock_prof_instance.stop.assert_called_once()

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_discrete_mode(self, mock_get_profiler):
        # Mock for discrete mode
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        # In discrete mode, start/stop shouldn't trigger global profiler immediately
        profiler.start()
        mock_get_profiler.assert_not_called()

        profiler.stop()
        mock_prof_instance.stop.assert_not_called()

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_discrete_annotate_stops_profiler_on_exception(self, mock_get_profiler):
        # A stage raising inside a discrete-mode annotate must still stop the
        # (process-global) torch profiler; otherwise it leaks, the next stage's
        # start() fails with "Profiler is already enabled" and the process aborts.
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        calls = {"n": 0}

        @profiler.annotate(role="boom")
        def boom():
            calls["n"] += 1
            raise RuntimeError("stage failed on purpose")

        with self.assertRaises(RuntimeError):
            boom()

        # Profiler must be started and, crucially, stopped despite the exception,
        # and the stage body must run exactly once (no re-execution).
        mock_prof_instance.start.assert_called_once()
        mock_prof_instance.stop.assert_called_once()
        self.assertEqual(calls["n"], 1)

    def test_dist_annotate_propagates_and_runs_func_once(self):
        # DistProfiler.annotate must not swallow errors from the wrapped function
        # nor re-run it (which would execute the stage twice). It only falls back
        # when *setting up* backend profiling fails.
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )

        class PassthroughImpl:
            def annotate(self, **kwargs):
                def decorator(fn):
                    return fn

                return decorator

        calls = {"n": 0}

        class FakeWorker:
            def __init__(self, profiler):
                self.profiler = profiler
                self.rank = 0

            @DistProfiler.annotate(role="boom")
            def boom(self):
                calls["n"] += 1
                raise RuntimeError("stage failed on purpose")

        dp = DistProfiler(rank=0, config=config, tool_config=tool_config)
        dp._impl = PassthroughImpl()
        dp._this_step = True  # simulate an active profiled step

        worker = FakeWorker(dp)
        with self.assertRaises(RuntimeError):
            worker.boom()
        self.assertEqual(calls["n"], 1)

    def test_dist_annotate_falls_back_when_setup_fails(self):
        # If backend annotate setup raises, the function still runs (once), unprofiled.
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )

        class BadImpl:
            def annotate(self, **kwargs):
                raise RuntimeError("cannot set up profiling")

        calls = {"n": 0}

        class FakeWorker:
            def __init__(self, profiler):
                self.profiler = profiler
                self.rank = 0

            @DistProfiler.annotate(role="x")
            def do_work(self):
                calls["n"] += 1
                return "ok"

        dp = DistProfiler(rank=0, config=config, tool_config=tool_config)
        dp._impl = BadImpl()
        dp._this_step = True

        worker = FakeWorker(dp)
        self.assertEqual(worker.do_work(), "ok")
        self.assertEqual(calls["n"], 1)

    @patch("torch.profiler.schedule")
    @patch("torch.profiler.profile")
    def test_get_torch_profiler_with_schedule(self, mock_profile, mock_schedule):
        # When a schedule dict is provided, torch.profiler.schedule must be built and forwarded.
        sentinel_schedule = object()
        mock_schedule.return_value = sentinel_schedule
        schedule = {"skip_first": 1, "wait": 2, "warmup": 1, "active": 3, "repeat": 2}

        get_torch_profiler(contents=["cpu"], save_path="/tmp/test", rank=0, schedule=schedule)

        mock_schedule.assert_called_once_with(**schedule)
        _, kwargs = mock_profile.call_args
        self.assertIs(kwargs["schedule"], sentinel_schedule)

    @patch("torch.profiler.schedule")
    @patch("torch.profiler.profile")
    def test_get_torch_profiler_without_schedule(self, mock_profile, mock_schedule):
        # Without a schedule, the profiler runs in continuous mode (no schedule kwarg).
        get_torch_profiler(contents=["cpu"], save_path="/tmp/test", rank=0)

        mock_schedule.assert_not_called()
        _, kwargs = mock_profile.call_args
        self.assertNotIn("schedule", kwargs)

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_scheduled_profiler_lifecycle(self, mock_get_profiler):
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        schedule_cfg = TorchProfilerScheduleConfig(wait=1, warmup=1, active=2, repeat=1)
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False, schedule=schedule_cfg)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        # Start forwards the resolved schedule kwargs and records the active profiler.
        profiler.start()
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(
            kwargs["schedule"],
            {"skip_first": 0, "wait": 1, "warmup": 1, "active": 2, "repeat": 1},
        )
        self.assertIs(Profiler._active_prof, mock_prof_instance)

        # Each step advances the active torch profiler.
        profiler.step()
        profiler.step()
        self.assertEqual(mock_prof_instance.step.call_count, 2)

        # With a schedule, stop must NOT emit an extra implicit step (stepping is per mini-batch).
        profiler.stop()
        self.assertEqual(mock_prof_instance.step.call_count, 2)
        mock_prof_instance.stop.assert_called_once()
        self.assertIsNone(Profiler._active_prof)

    def test_build_trace_basename_encodes_role_rank_and_parallelism(self):
        # Filename stem must embed the worker role, scope role, rank/world size and the
        # tp/pp/dp/cp parallel ranks so per-process traces are self-describing.
        name = build_trace_basename(
            rank=5,
            role="e2e",
            save_file_prefix="actor",
            topology={"rank": 5, "world_size": 16, "tp": 1, "pp": 0, "dp": 2, "cp": 0},
        )
        self.assertTrue(name.startswith("actor_e2e_"))
        self.assertIn("rank5-of-16", name)
        self.assertIn("tp1-pp0-dp2-cp0", name)
        self.assertIn(f"pid{os.getpid()}", name)

    def test_build_trace_basename_distinguishes_roles_same_rank(self):
        # The original scheme collided ref/critic at the same rank; the role prefix fixes it.
        topo = {"rank": 5, "world_size": 16}
        ref_name = build_trace_basename(rank=5, save_file_prefix="ref", topology=topo)
        critic_name = build_trace_basename(rank=5, save_file_prefix="value_model", topology=topo)
        self.assertTrue(ref_name.startswith("ref_rank5-of-16_"))
        # Underscores in labels are normalized to hyphens (underscore is the field separator).
        self.assertTrue(critic_name.startswith("value-model_rank5-of-16_"))
        self.assertNotEqual(ref_name, critic_name)

    def test_build_trace_basename_minimal_topology(self):
        # With no distributed topology, fall back to the passed rank and omit parallel dims.
        name = build_trace_basename(rank=3, topology={})
        self.assertTrue(name.startswith("rank3_"))
        self.assertNotIn("-of-", name)
        for dim in ("tp", "pp", "dp", "cp"):
            self.assertNotIn(f"{dim}0", name)

    def test_build_trace_basename_sanitizes_labels(self):
        # Slashes/spaces in labels must not leak into the filename.
        name = build_trace_basename(rank=0, role="update actor", save_file_prefix="actor/rollout", topology={})
        self.assertNotIn("/", name)
        self.assertNotIn(" ", name)
        self.assertIn("actor-rollout", name)
        self.assertIn("update-actor", name)

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_dist_profiler_forwards_save_file_prefix(self, mock_get_profiler):
        # DistProfiler must forward save_file_prefix down to the torch backend so it
        # ends up in the trace filename.
        mock_get_profiler.return_value = MagicMock()
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )
        dist_profiler = DistProfiler(rank=0, config=config, tool_config=tool_config, save_file_prefix="actor")
        self.assertEqual(dist_profiler._impl.save_file_prefix, "actor")

        dist_profiler.start()
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(kwargs["save_file_prefix"], "actor")
        dist_profiler.stop()

    def test_dist_profiler_step_noop_backend(self):
        # A backend without scheduling support (no-op impl) must make step() a safe no-op.
        config = ProfilerConfig(tool=None, enable=True, all_ranks=True, save_path="/tmp/test", tool_config=None)
        dist_profiler = DistProfiler(rank=0, config=config)
        self.assertIsNone(dist_profiler.step())

    def test_dist_profiler_step_disabled(self):
        # When disabled, step() must not touch the backend at all.
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(
            tool="torch", enable=False, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )
        dist_profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)
        dist_profiler._impl = MagicMock()
        self.assertIsNone(dist_profiler.step())
        dist_profiler._impl.step.assert_not_called()

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_dist_profiler_step_torch_delegates(self, mock_get_profiler):
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )
        dist_profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)

        dist_profiler.start()
        dist_profiler.step()
        mock_prof_instance.step.assert_called_once()

        dist_profiler.stop()


def _role_profiler_omegaconf(tool="torch", enable=True, discrete=False, contents=("cpu", "cuda")):
    """Mimic a per-role ``profiler`` OmegaConf sub-tree (identical across ref/ref.yaml and
    critic/critic.yaml).

    The nested ``_target_`` entries are what the hydra instantiation path (omega_conf_to_dataclass
    without an explicit dataclass_type) uses to build real dataclass tool configs, as opposed to the
    plain dicts the torch profiler cannot consume.
    """
    return OmegaConf.create(
        {
            "_target_": "verl.utils.profiler.ProfilerConfig",
            "tool": tool,
            "enable": enable,
            "all_ranks": False,
            "ranks": [0],
            "save_path": "/tmp/test_role_profile",
            "tool_config": {
                "torch": {
                    "_target_": "verl.utils.profiler.config.TorchProfilerToolConfig",
                    "contents": list(contents),
                    "discrete": discrete,
                    "schedule": {
                        "_target_": "verl.utils.profiler.config.TorchProfilerScheduleConfig",
                        "skip_first": 0,
                        "wait": 0,
                        "warmup": 0,
                        "active": 0,
                        "repeat": 0,
                    },
                },
            },
        }
    )


class TestRefWorkerProfilerConfig(unittest.TestCase):
    """The reference model's inner TrainingWorker must receive a real, torch-consumable profiler
    config (mirroring the actor), instead of silently running with a disabled no-op profiler.

    ``ActorRolloutRefWorker.init_model`` now forwards the ref's own ``profiler`` config to the ref
    ``TrainingWorkerConfig`` via ``omega_conf_to_dataclass(self.config.ref.get("profiler", {}))``.
    These lock in that conversion path so a torch profiler config actually yields a torch backend on
    the ref worker, while an absent config degrades to a no-op (the previous ref behavior).
    """

    def test_ref_profiler_config_builds_torch_backend(self):
        # Exercise the exact conversion init_model performs on actor_rollout_ref.ref.profiler:
        # omega_conf_to_dataclass(...) (no dataclass_type) must resolve the _target_ entries into
        # real nested dataclasses the torch Profiler can consume via attribute access.
        omega_cfg = _role_profiler_omegaconf(tool="torch", enable=True)
        ref_profiler_config = omega_conf_to_dataclass(omega_cfg)

        self.assertIsInstance(ref_profiler_config, ProfilerConfig)
        self.assertTrue(ref_profiler_config.enable)
        self.assertEqual(ref_profiler_config.tool, "torch")

        # TrainingWorker.__init__ extracts the tool-specific config exactly like this; it must be a
        # real dataclass (not a plain dict) for the torch Profiler to read .contents/.schedule.
        tool_config = ref_profiler_config.tool_config.get(ref_profiler_config.tool)
        self.assertIsInstance(tool_config, TorchProfilerToolConfig)
        self.assertEqual(tool_config.contents, ["cpu", "cuda"])

        dist_profiler = DistProfiler(
            rank=0, config=ref_profiler_config, tool_config=tool_config, save_file_prefix="ref"
        )
        self.assertIsInstance(dist_profiler._impl, Profiler)
        self.assertTrue(dist_profiler.check_enable())
        self.assertTrue(dist_profiler.check_this_rank())

    def test_absent_ref_profiler_config_is_disabled_noop(self):
        # Contrast with the previous behavior: without a profiler_config the ref worker built a
        # disabled no-op profiler, so the reference model was never profiled by its own worker.
        dist_profiler = DistProfiler(rank=0, config=None)
        self.assertIsInstance(dist_profiler._impl, _NoOpProfiler)
        self.assertFalse(dist_profiler.check_enable())


class TestCriticWorkerProfilerConfig(unittest.TestCase):
    """The critic is a standalone TrainingWorker (no outer ActorRolloutRefWorker wrapper): the
    trainer drives start_profile()/stop_profile() and the ``train_batch`` annotation directly on it.

    ``RayPPOTrainer._init_workers`` (and the v1 / separation trainer variants) now forward
    ``omega_conf_to_dataclass(self.config.critic.get("profiler", {}))`` into the critic
    ``TrainingWorkerConfig``. Without it the critic's DistProfiler silently degraded to a no-op, so
    the critic was never profiled by any backend. These lock in that wiring.
    """

    def test_critic_profiler_config_builds_torch_backend(self):
        # critic/critic.yaml's profiler block is structurally identical to ref/ref.yaml; the trainer
        # converts it the same way. It must yield a real torch backend on the standalone critic worker.
        omega_cfg = _role_profiler_omegaconf(tool="torch", enable=True)
        critic_profiler_config = omega_conf_to_dataclass(omega_cfg)

        self.assertIsInstance(critic_profiler_config, ProfilerConfig)
        tool_config = critic_profiler_config.tool_config.get(critic_profiler_config.tool)
        self.assertIsInstance(tool_config, TorchProfilerToolConfig)

        # The critic TrainingWorker uses model_type="value_model" as the trace filename prefix.
        dist_profiler = DistProfiler(
            rank=0, config=critic_profiler_config, tool_config=tool_config, save_file_prefix="value_model"
        )
        self.assertIsInstance(dist_profiler._impl, Profiler)
        self.assertTrue(dist_profiler.check_enable())
        self.assertTrue(dist_profiler.check_this_rank())
        self.assertEqual(dist_profiler._impl.save_file_prefix, "value_model")

    def test_absent_critic_profiler_config_is_disabled_noop(self):
        # The previous behavior: the trainer built the critic TrainingWorkerConfig without a
        # profiler_config, so DistProfiler(config=None) degraded to a disabled no-op.
        dist_profiler = DistProfiler(rank=0, config=None)
        self.assertIsInstance(dist_profiler._impl, _NoOpProfiler)
        self.assertFalse(dist_profiler.check_enable())


if __name__ == "__main__":
    unittest.main()
