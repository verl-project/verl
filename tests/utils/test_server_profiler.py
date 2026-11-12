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
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from verl.utils.profiler.config import (
    NsightToolConfig,
    ProfilerConfig,
    TorchProfilerToolConfig,
    relocate_rollout_traces,
    rollout_profiler_global_ranks,
    rollout_trace_dir,
    rollout_trace_local_rank,
)
from verl.utils.profiler.profile import DistProfiler


class TestRolloutTraceRelocation(unittest.TestCase):
    """Engines write into a sub-directory; relocation brings the traces to the configured path."""

    def _config(self, save_path: str, **kwargs) -> ProfilerConfig:
        return ProfilerConfig(
            tool="torch",
            enable=True,
            ranks=[0],
            save_path=save_path,
            tool_config=TorchProfilerToolConfig(contents=["cpu"], discrete=True),
            **kwargs,
        )

    def _write_engine_traces(self, config: ProfilerConfig, rank: int, *names: str) -> str:
        src_dir = rollout_trace_dir(config, rank)
        os.makedirs(src_dir, exist_ok=True)
        for name in names:
            with open(os.path.join(src_dir, name), "w") as f:
                f.write(name)
        return src_dir

    def test_traces_end_up_next_to_the_training_ones(self):
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path, relocate_results=True)
            src_dir = self._write_engine_traces(config, 2, "host_123.pt.trace.json.gz")

            relocated = relocate_rollout_traces(config, rank=2)

            expected = os.path.join(save_path, "rollout-replica2_host_123.pt.trace.json.gz")
            self.assertEqual(relocated, [expected])
            # The replica is in the name now that it is no longer in the path.
            self.assertTrue(os.path.exists(expected))
            self.assertEqual(os.listdir(src_dir), [])

    def test_traces_stay_where_the_engine_wrote_them_by_default(self):
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path)
            src_dir = self._write_engine_traces(config, 0, "host_123.pt.trace.json.gz")

            self.assertEqual(relocate_rollout_traces(config, rank=0), [])

            self.assertEqual(os.listdir(src_dir), ["host_123.pt.trace.json.gz"])

    def test_an_engine_that_wrote_nothing_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path, relocate_results=True)

            self.assertEqual(relocate_rollout_traces(config, rank=1), [])

    def test_global_gpu_rank_is_added_when_the_engine_encodes_the_tp_rank(self):
        # The name should line up with the global `ranks` the user configured: with tp=2, replica 4
        # drives the engine spanning global GPUs 8 and 9, and the engine names its per-GPU traces by
        # tensor-parallel rank, so rank0 -> global 8 and rank1 -> global 9.
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path, relocate_results=True)
            self._write_engine_traces(config, 4, "rank0.pt.trace.json.gz", "rank1.pt.trace.json.gz")

            relocated = relocate_rollout_traces(config, rank=4, world_size=2)

            self.assertEqual(
                sorted(os.path.basename(p) for p in relocated),
                [
                    "rollout-replica4-globalrank8_rank0.pt.trace.json.gz",
                    "rollout-replica4-globalrank9_rank1.pt.trace.json.gz",
                ],
            )

    def test_replica_index_is_used_when_the_tp_rank_is_not_in_the_name(self):
        # Engines that name traces by host/pid instead of tp rank cannot be mapped to a single GPU,
        # so the whole (tp-sharded) engine keeps the replica-only name.
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path, relocate_results=True)
            self._write_engine_traces(config, 4, "host_123.pt.trace.json.gz")

            relocated = relocate_rollout_traces(config, rank=4, world_size=2)

            self.assertEqual(
                [os.path.basename(p) for p in relocated],
                ["rollout-replica4_host_123.pt.trace.json.gz"],
            )

    def test_world_size_one_keeps_the_replica_index_which_already_is_the_global_rank(self):
        # tp=dp=pp=1: the replica index equals the global GPU rank, so no extra suffix is needed.
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path, relocate_results=True)
            self._write_engine_traces(config, 8, "rank0.pt.trace.json.gz")

            relocated = relocate_rollout_traces(config, rank=8, world_size=1)

            self.assertEqual(
                [os.path.basename(p) for p in relocated],
                ["rollout-replica8_rank0.pt.trace.json.gz"],
            )

    def test_ranks_0_and_8_yield_exactly_gpu_0_and_8_not_their_tp_mates(self):
        # The whole point: tp=2 forces the engine to trace its whole replica, but ranks=[0, 8] must
        # surface exactly global GPUs 0 and 8. Replica 0 owns GPUs 0-1 and replica 4 owns GPUs 8-9,
        # so tp rank 1 (GPUs 1 and 9) is dropped, not relocated into save_path.
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path, relocate_results=True)
            self._write_engine_traces(config, 0, "rank0.pt.trace.json.gz", "rank1.pt.trace.json.gz")
            self._write_engine_traces(config, 4, "rank0.pt.trace.json.gz", "rank1.pt.trace.json.gz")

            kept0 = relocate_rollout_traces(config, rank=0, world_size=2, keep_global_ranks={0, 8})
            kept4 = relocate_rollout_traces(config, rank=4, world_size=2, keep_global_ranks={0, 8})

            self.assertEqual(
                [os.path.basename(p) for p in kept0],
                ["rollout-replica0-globalrank0_rank0.pt.trace.json.gz"],
            )
            self.assertEqual(
                [os.path.basename(p) for p in kept4],
                ["rollout-replica4-globalrank8_rank0.pt.trace.json.gz"],
            )
            # The flat traces in save_path (ignoring the per-replica sub-directories) are only the
            # two GPUs that were asked for.
            top_level_files = sorted(
                name for name in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, name))
            )
            self.assertEqual(
                top_level_files,
                [
                    "rollout-replica0-globalrank0_rank0.pt.trace.json.gz",
                    "rollout-replica4-globalrank8_rank0.pt.trace.json.gz",
                ],
            )
            # The unrequested tp-mates are left behind in the sub-directory, not deleted.
            self.assertEqual(os.listdir(rollout_trace_dir(config, 0)), ["rank1.pt.trace.json.gz"])
            self.assertEqual(os.listdir(rollout_trace_dir(config, 4)), ["rank1.pt.trace.json.gz"])

    def test_filtering_never_drops_a_file_whose_global_rank_is_unknown(self):
        # If the tp rank cannot be read from the engine's name we cannot prove the file is unwanted,
        # so it is kept rather than risk dropping a rank the user asked for.
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path, relocate_results=True)
            self._write_engine_traces(config, 0, "host_123.pt.trace.json.gz")

            relocated = relocate_rollout_traces(config, rank=0, world_size=2, keep_global_ranks={0})

            self.assertEqual(
                [os.path.basename(p) for p in relocated],
                ["rollout-replica0_host_123.pt.trace.json.gz"],
            )

    def test_sglang_tp_named_traces_are_filtered_to_the_requested_ranks(self):
        # SGLang names per-GPU traces "{profile_id}-TP-{tp}.trace.json.gz". ranks=[0, 8] with tp=2
        # must keep only TP-0 of replica 0 (global 0) and TP-0 of replica 4 (global 8).
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path, relocate_results=True)
            self._write_engine_traces(config, 0, "42-TP-0.trace.json.gz", "42-TP-1.trace.json.gz")
            self._write_engine_traces(config, 4, "42-TP-0.trace.json.gz", "42-TP-1.trace.json.gz")

            kept0 = relocate_rollout_traces(config, rank=0, world_size=2, keep_global_ranks={0, 8})
            kept4 = relocate_rollout_traces(config, rank=4, world_size=2, keep_global_ranks={0, 8})

            self.assertEqual(
                [os.path.basename(p) for p in kept0],
                ["rollout-replica0-globalrank0_42-TP-0.trace.json.gz"],
            )
            self.assertEqual(
                [os.path.basename(p) for p in kept4],
                ["rollout-replica4-globalrank8_42-TP-0.trace.json.gz"],
            )

    def test_vllm_instance_rank_named_traces_are_filtered_to_the_requested_ranks(self):
        # vLLM names per-GPU traces "{instance_id}-rank-{tp}.{ts}.pt.trace.json.gz".
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path, relocate_results=True)
            self._write_engine_traces(
                config, 4, "inst9-rank-0.1700.pt.trace.json.gz", "inst9-rank-1.1700.pt.trace.json.gz"
            )

            kept = relocate_rollout_traces(config, rank=4, world_size=2, keep_global_ranks={0, 8})

            self.assertEqual(
                [os.path.basename(p) for p in kept],
                ["rollout-replica4-globalrank8_inst9-rank-0.1700.pt.trace.json.gz"],
            )

    def test_multi_dim_parallel_names_are_kept_rather_than_mismapped(self):
        # With DP/PP/EP in the name the GPU's linear offset is layout-dependent; do not guess, keep all.
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path, relocate_results=True)
            self._write_engine_traces(config, 0, "42-TP-0-DP-0.trace.json.gz", "42-TP-1-DP-1.trace.json.gz")

            relocated = relocate_rollout_traces(config, rank=0, world_size=2, keep_global_ranks={0})

            self.assertEqual(
                sorted(os.path.basename(p) for p in relocated),
                ["rollout-replica0_42-TP-0-DP-0.trace.json.gz", "rollout-replica0_42-TP-1-DP-1.trace.json.gz"],
            )


class TestRolloutTraceLocalRank(unittest.TestCase):
    """`rollout_trace_local_rank` reads a GPU's offset within its replica from the engine file name."""

    def test_reads_sglang_tp_rank(self):
        self.assertEqual(rollout_trace_local_rank("42-TP-0.trace.json.gz", world_size=2), 0)
        self.assertEqual(rollout_trace_local_rank("42-TP-1.trace.json.gz", world_size=2), 1)

    def test_reads_vllm_instance_rank(self):
        self.assertEqual(rollout_trace_local_rank("inst-rank-0.99.pt.trace.json.gz", world_size=4), 0)
        self.assertEqual(rollout_trace_local_rank("inst-rank-3.99.pt.trace.json.gz", world_size=4), 3)

    def test_reads_bare_rank_prefix(self):
        self.assertEqual(rollout_trace_local_rank("rank0.pt.trace.json.gz", world_size=2), 0)

    def test_multi_dim_layout_is_ambiguous(self):
        self.assertIsNone(rollout_trace_local_rank("42-TP-0-DP-1.trace.json.gz", world_size=4))

    def test_rank_out_of_range_is_rejected(self):
        # A parsed value that cannot be a valid offset in this replica is not trusted.
        self.assertIsNone(rollout_trace_local_rank("42-TP-9.trace.json.gz", world_size=2))

    def test_host_pid_names_are_unknown(self):
        self.assertIsNone(rollout_trace_local_rank("host_12345.170.pt.trace.json.gz", world_size=2))

    def test_world_size_one_needs_no_offset(self):
        self.assertIsNone(rollout_trace_local_rank("42-TP-0.trace.json.gz", world_size=1))


class TestRolloutProfilerGlobalRanks(unittest.TestCase):
    """`rollout_profiler_global_ranks` derives the "keep" set relocation uses from the config."""

    def _config(self, **kwargs) -> ProfilerConfig:
        return ProfilerConfig(tool="torch", enable=True, save_path="/tmp/test", tool_config=None, **kwargs)

    def test_explicit_ranks_become_the_keep_set(self):
        self.assertEqual(rollout_profiler_global_ranks(self._config(ranks=[0, 8])), {0, 8})

    def test_all_ranks_keeps_everything(self):
        self.assertIsNone(rollout_profiler_global_ranks(self._config(all_ranks=True, ranks=[0, 8])))

    def test_empty_ranks_keeps_everything(self):
        self.assertIsNone(rollout_profiler_global_ranks(self._config(ranks=[])))

    def test_missing_profiler_config_keeps_everything(self):
        self.assertIsNone(rollout_profiler_global_ranks(None))


class TestRolloutFinishHook(unittest.TestCase):
    """The finish hook runs the user's command once, on the last profiled step (run_command=True)."""

    def _config(self, save_path: str, cmd: str, **kwargs) -> ProfilerConfig:
        return ProfilerConfig(
            tool="torch",
            enable=True,
            ranks=[0],
            save_path=save_path,
            tool_config=TorchProfilerToolConfig(contents=["cpu"], discrete=True),
            finish_hook_cmd=cmd,
            **kwargs,
        )

    def test_run_finish_hook_runs_command_without_going_through_stop(self):
        with tempfile.TemporaryDirectory() as save_path:
            marker = os.path.join(save_path, "hook_ran")
            profiler = DistProfiler(rank=0, config=self._config(save_path, f"touch {marker}"))

            profiler.run_finish_hook()

            self.assertTrue(os.path.exists(marker))

    def test_run_finish_hook_exports_save_path_to_the_command(self):
        with tempfile.TemporaryDirectory() as save_path:
            out = os.path.join(save_path, "env")
            profiler = DistProfiler(rank=0, config=self._config(save_path, f'echo "$VERL_PROFILE_SAVE_PATH" > {out}'))

            profiler.run_finish_hook()

            with open(out) as f:
                self.assertEqual(f.read().strip(), save_path)

    def test_run_finish_hook_skips_unselected_ranks(self):
        with tempfile.TemporaryDirectory() as save_path:
            marker = os.path.join(save_path, "hook_ran")
            config = self._config(save_path, f"touch {marker}", finish_hook_ranks=[3])
            profiler = DistProfiler(rank=0, config=config)

            profiler.run_finish_hook()

            self.assertFalse(os.path.exists(marker))

    def test_run_finish_hook_defers_the_command_until_the_final_step(self):
        # The command runs once, on the last profiled step: earlier steps pass run_command=False, so
        # nothing is uploaded yet (backend stop + relocation still happen elsewhere every step), and
        # the last step passes run_command=True to fire the single upload of the whole save_path.
        with tempfile.TemporaryDirectory() as save_path:
            marker = os.path.join(save_path, "hook_ran")
            profiler = DistProfiler(rank=0, config=self._config(save_path, f"touch {marker}"))

            profiler.run_finish_hook(run_command=False)
            self.assertFalse(os.path.exists(marker))  # not the last profiled step yet

            profiler.run_finish_hook(run_command=True)
            self.assertTrue(os.path.exists(marker))  # last step: command runs exactly once

    def test_run_finish_hook_uploads_the_whole_save_path_once(self):
        # A single end-of-run upload of the directory sends each accumulated trace exactly once --
        # this is what replaces the per-step upload (which re-sent earlier steps every step).
        with tempfile.TemporaryDirectory() as save_path:
            uploaded = os.path.join(save_path, "uploaded.log")
            cmd = f'for f in "$VERL_PROFILE_SAVE_PATH"/*.json.gz; do echo "$(basename "$f")" >> {uploaded}; done'
            profiler = DistProfiler(rank=0, config=self._config(save_path, cmd))

            # Both steps' traces have accumulated in save_path by the time the command runs.
            for name in ("step1.json.gz", "step2.json.gz"):
                with open(os.path.join(save_path, name), "w") as f:
                    f.write("x")

            profiler.run_finish_hook(run_command=True)

            sent = open(uploaded).read().split()
            self.assertEqual(sorted(sent), ["step1.json.gz", "step2.json.gz"])  # each exactly once


class TestVllmProfiler:
    @pytest.fixture
    def backend(self):
        return pytest.importorskip("verl.workers.rollout.vllm_rollout.vllm_async_server")

    @pytest.fixture
    def server(self, backend):
        server = backend.vLLMHttpServer.__new__(backend.vLLMHttpServer)
        server.node_rank = 0
        server.profiler_controller = Mock(_tool="nsys")
        server.profiler_controller.check_enable.return_value = True
        server.profiler_controller.check_this_rank.return_value = True
        server.profiler_controller.is_discrete_mode.return_value = False
        server.engine = AsyncMock()
        return server

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool, discrete, expected_calls, collect_traces",
        [
            ("torch", True, [call.engine_start(profile_step=3), call.engine_stop()], True),
            (
                "nsys",
                False,
                [
                    call.server_start(profile_step=3),
                    call.engine_start(),
                    call.engine_stop(),
                    call.server_stop(run_command=False),
                ],
                False,
            ),
        ],
        ids=["torch", "nsys"],
    )
    async def test_start_stop_profile(self, backend, server, tool, discrete, expected_calls, collect_traces):
        server.profiler_controller._tool = tool
        server.profiler_controller.is_discrete_mode.return_value = discrete
        server.profiler_controller.config = ProfilerConfig(save_path="/tmp/test", tool_config=None)
        server.replica_rank = 3
        server.replica_world_size = 2
        server.profiler_keep_global_ranks = {6}
        events = Mock()
        events.attach_mock(server.profiler_controller.start, "server_start")
        events.attach_mock(server.profiler_controller.stop, "server_stop")
        events.attach_mock(server.engine.start_profile, "engine_start")
        events.attach_mock(server.engine.stop_profile, "engine_stop")

        with patch.object(backend, "relocate_rollout_traces") as relocate:
            await server.start_profile(profile_step=3)
            await server.stop_profile()

        assert events.mock_calls == expected_calls
        if collect_traces:
            relocate.assert_called_once_with(server.profiler_controller.config, 3, 2, {6})
        else:
            relocate.assert_not_called()
        # The training worker owns the shared directory's end-of-run upload.
        server.profiler_controller.run_finish_hook.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool, server_calls", [("torch", 0), ("nsys", 1)])
    @pytest.mark.parametrize(
        "node_rank, enabled, selected",
        [(1, True, True), (0, False, True), (0, True, False)],
        ids=["secondary-node", "disabled", "unselected"],
    )
    async def test_skipped_engine_does_not_collect_traces(
        self, backend, server, tool, server_calls, node_rank, enabled, selected
    ):
        server.profiler_controller._tool = tool
        server.profiler_controller.is_discrete_mode.return_value = True
        server.node_rank = node_rank
        server.profiler_controller.check_enable.return_value = enabled
        server.profiler_controller.check_this_rank.return_value = selected

        with patch.object(backend, "relocate_rollout_traces") as relocate:
            await server.start_profile(profile_step=3)
            await server.stop_profile()

        server.engine.start_profile.assert_not_awaited()
        server.engine.stop_profile.assert_not_awaited()
        relocate.assert_not_called()
        assert server.profiler_controller.start.call_count == server_calls
        assert server.profiler_controller.stop.call_count == server_calls

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["start", "stop"])
    async def test_engine_failure_still_stops_server_profiler(self, backend, server, action):
        error = RuntimeError("engine failed")
        getattr(server.engine, f"{action}_profile").side_effect = error

        with patch.object(backend, "relocate_rollout_traces") as relocate:
            with pytest.raises(RuntimeError, match="engine") as exc:
                await getattr(server, f"{action}_profile")()

        assert exc.value is error
        server.profiler_controller.stop.assert_called_once_with(run_command=False)
        relocate.assert_not_called()

    def test_init_accepts_nsight_and_sets_report_role(self, backend):
        profiler_config = ProfilerConfig(tool="nsys", tool_config={"nsys": NsightToolConfig()})
        server = backend.vLLMHttpServer.__new__(backend.vLLMHttpServer)
        server.config = SimpleNamespace(
            profiler=profiler_config,
            tensor_model_parallel_size=2,
            data_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        server.replica_rank = 3
        server.node_rank = 1
        with patch.object(backend, "build_rollout_dist_profiler") as build_profiler:
            server._init_profiler()

        build_profiler.assert_called_once_with(3, 2, config=profiler_config, tool_config=NsightToolConfig())
        assert server.profiler_controller is build_profiler.return_value
        assert server.profiler_controller.save_file_prefix == "rollout_server_node_1"


class TestSGLangProfiler(unittest.IsolatedAsyncioTestCase):
    async def test_sglang_start_stop_profile(self):
        try:
            # Import strictly inside test to avoid import errors if dependencies missing
            from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
        except ImportError:
            self.skipTest("sglang or dependencies not installed")
            return

        # Mock dependencies
        mock_profiler = MagicMock()
        mock_profiler.check_enable.return_value = True
        mock_profiler.check_this_rank.return_value = True
        mock_profiler.is_discrete_mode.return_value = True
        mock_profiler.config = ProfilerConfig(save_path="/tmp/test", tool_config=None)
        mock_profiler.tool_config = MagicMock()

        mock_tokenizer_manager = AsyncMock()

        mock_self = MagicMock()
        mock_self.profiler_controller = mock_profiler
        mock_self.tokenizer_manager = mock_tokenizer_manager
        mock_self.replica_rank = 0

        # Mock build_sglang_profiler_args to return known dict
        with patch("verl.workers.rollout.sglang_rollout.async_sglang_server.build_sglang_profiler_args") as mock_build:
            mock_args = {"arg1": "val1"}
            mock_build.return_value = mock_args

            # Test start_profile
            await SGLangHttpServer.start_profile(mock_self)

            mock_build.assert_called_once_with(mock_profiler.config, mock_profiler.tool_config, mock_self.replica_rank)
            mock_tokenizer_manager.start_profile.assert_called_once_with(**mock_args)

            # Test stop_profile
            with patch(
                "verl.workers.rollout.sglang_rollout.async_sglang_server.relocate_rollout_traces"
            ) as mock_relocate:
                await SGLangHttpServer.stop_profile(mock_self)
            mock_tokenizer_manager.stop_profile.assert_called_once()
            # Relocation runs every profiled step so traces accumulate in save_path; the engine does
            # not run the finish command itself (the colocated training worker's single end-of-run
            # upload covers them), so running it here too would upload the shared directory twice.
            mock_relocate.assert_called_once_with(
                mock_profiler.config,
                mock_self.replica_rank,
                mock_self.replica_world_size,
                mock_self.profiler_keep_global_ranks,
            )
            mock_profiler.run_finish_hook.assert_not_called()


@pytest.mark.parametrize(
    "options, expected_env",
    [
        (
            {"o": "rendered_%p"},
            {"env_vars": {"NOSET_DEVICE": "1", "EXTRA": "value"}, "nsight": {"o": "rendered_%p"}},
        ),
        (None, {"env_vars": {"NOSET_DEVICE": "1", "EXTRA": "value"}}),
    ],
    ids=["nsight", "no-nsight"],
)
def test_vllm_replica_build_server_runtime_env(options, expected_env):
    backend = pytest.importorskip("verl.workers.rollout.vllm_rollout.vllm_async_server")
    replica = SimpleNamespace(
        world_size=4,
        replica_rank=3,
        config=SimpleNamespace(profiler={}),
    )

    platform = Mock()
    platform.ray_noset_envvars.return_value = ["NOSET_DEVICE"]
    platform.rollout_env_vars.return_value = {"EXTRA": "value"}
    with (
        patch.object(backend, "get_platform", return_value=platform),
        patch.object(backend, "_build_rollout_ray_actor_nsight_launch_options", return_value=options) as build_options,
    ):
        runtime_env = backend.vLLMReplica._build_server_runtime_env(replica, node_rank=1)

    build_options.assert_called_once_with(
        replica.config.profiler, replica_rank=3, replica_world_size=4, role="rollout_server_node_1"
    )
    assert runtime_env == expected_env


if __name__ == "__main__":
    unittest.main()
