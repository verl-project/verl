# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import unittest
from unittest.mock import MagicMock, patch

from verl.utils.profiler.config import NPUToolConfig, ProfilerConfig
from verl.utils.profiler.mstx_profile import NPUProfiler
from verl.utils.profiler.profile import DistProfiler


class TestNPUProfilerInitialization(unittest.TestCase):
    def setUp(self):
        NPUProfiler._define_count = 0

    def test_init_with_default_config(self):
        tool_config = NPUToolConfig()
        config = ProfilerConfig(tool="npu")
        profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)
        self.assertFalse(profiler.check_enable())

    def test_init_with_disabled_config(self):
        config = ProfilerConfig(enable=False, tool="npu")
        tool_config = NPUToolConfig()
        profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)
        self.assertFalse(profiler.check_enable())

    def test_init_with_all_ranks_true(self):
        config = ProfilerConfig(enable=True, all_ranks=True, tool="npu")
        tool_config = NPUToolConfig()
        profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)
        self.assertTrue(profiler.check_this_rank())

    def test_init_with_ranks_list(self):
        config = ProfilerConfig(enable=True, ranks=[1, 2], tool="npu")
        tool_config = NPUToolConfig()
        profiler = DistProfiler(rank=1, config=config, tool_config=tool_config)
        self.assertTrue(profiler.check_this_rank())

    def test_init_with_rank_not_in_ranks(self):
        config = ProfilerConfig(enable=True, ranks=[1, 2], tool="npu")
        tool_config = NPUToolConfig()
        profiler = DistProfiler(rank=3, config=config, tool_config=tool_config)
        self.assertFalse(profiler.check_this_rank())


class TestNPUProfilerStart(unittest.TestCase):
    def setUp(self):
        NPUProfiler._define_count = 0
        self.config = ProfilerConfig(enable=True, ranks=[0], tool="npu")
        self.tool_config = NPUToolConfig(discrete=False)

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_start_when_enabled_and_this_rank(self, mock_get_profiler):
        profiler = DistProfiler(rank=0, config=self.config, tool_config=self.tool_config)
        profiler.start(role="worker", profile_step="1")
        self.assertTrue(profiler.check_this_step())
        self.assertEqual(NPUProfiler._define_count, 1)
        mock_get_profiler.assert_called_once()

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_start_when_not_this_rank(self, mock_get_profiler):
        profiler = DistProfiler(rank=1, config=self.config, tool_config=self.tool_config)
        profiler.start()
        self.assertFalse(profiler.check_this_step())
        self.assertEqual(NPUProfiler._define_count, 0)
        mock_get_profiler.assert_not_called()

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_start_discrete_mode_does_not_increase_count(self, mock_get_profiler):
        tool_config = NPUToolConfig(discrete=True)
        profiler = DistProfiler(rank=0, config=self.config, tool_config=tool_config)
        profiler.start()
        self.assertEqual(NPUProfiler._define_count, 0)
        mock_get_profiler.assert_not_called()

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_multiple_start_calls_do_not_increase_count(self, mock_get_profiler):
        profiler = DistProfiler(rank=0, config=self.config, tool_config=self.tool_config)
        profiler.start()
        profiler.start()
        self.assertEqual(NPUProfiler._define_count, 1)
        mock_get_profiler.assert_called_once()


class TestNPUProfilerStartStopInteraction(unittest.TestCase):
    def setUp(self):
        NPUProfiler._define_count = 0
        self.config = ProfilerConfig(enable=True, ranks=[0], tool="npu")
        self.tool_config = NPUToolConfig(discrete=False)

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_start_stop_cycle(self, mock_get_profiler):
        mock_profile_npu = MagicMock()
        mock_get_profiler.return_value = mock_profile_npu

        profiler = DistProfiler(rank=0, config=self.config, tool_config=self.tool_config)
        profiler.start()
        self.assertEqual(NPUProfiler._define_count, 1)
        self.assertEqual(mock_profile_npu.start.call_count, 1)
        profiler.stop()
        self.assertEqual(NPUProfiler._define_count, 0)
        self.assertEqual(mock_profile_npu.step.call_count, 1)
        self.assertEqual(mock_profile_npu.stop.call_count, 1)

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_multiple_instances_share_define_count(self, mock_get_profiler):
        mock_profile_npu = MagicMock()
        mock_get_profiler.return_value = mock_profile_npu

        profiler1 = DistProfiler(rank=0, config=self.config, tool_config=self.tool_config)
        profiler2 = DistProfiler(rank=0, config=self.config, tool_config=self.tool_config)
        profiler1.start()
        profiler2.start()
        self.assertEqual(NPUProfiler._define_count, 1)
        self.assertEqual(mock_profile_npu.start.call_count, 1)
        profiler1.stop()
        self.assertEqual(NPUProfiler._define_count, 0)


class TestNPUProfilerAnnotate(unittest.TestCase):
    def setUp(self):
        self.config = ProfilerConfig(enable=True, all_ranks=True, tool="npu")
        self.tool_config = NPUToolConfig(discrete=False)
        self.rank = 0

    def test_annotate_decorator_applied_correctly(self):
        mock_worker = MagicMock()
        mock_worker.profiler = DistProfiler(rank=self.rank, config=self.config, tool_config=self.tool_config)
        # Manually set private attribute for testing annotation in active step
        mock_worker.profiler._this_step = True

        mock_mark_range = "mocked_range_handle"

        with (
            patch("verl.utils.profiler.mstx_profile.mark_start_range") as mock_start_patch,
            patch("verl.utils.profiler.mstx_profile.mark_end_range") as mock_end_patch,
        ):
            mock_start_patch.return_value = mock_mark_range

            with patch("verl.utils.profiler.mstx_profile.get_npu_profiler") as mock_get_profiler:
                decorator = mock_worker.profiler.annotate(message="test")

                @decorator
                def test_func(self, *args, **kwargs):
                    return "result"

                result = test_func(mock_worker)

                self.assertEqual(result, "result")
                mock_start_patch.assert_called_once_with(message="test")
                mock_end_patch.assert_called_once_with(mock_mark_range)
                mock_get_profiler.assert_not_called()

    def test_annotate_when_profiler_disabled(self):
        disabled_config = ProfilerConfig(enable=False, tool="npu")
        mock_worker = MagicMock()
        mock_worker.profiler = DistProfiler(rank=self.rank, config=disabled_config, tool_config=self.tool_config)

        with (
            patch("verl.utils.profiler.mstx_profile.mark_start_range") as mock_start_patch,
            patch("verl.utils.profiler.mstx_profile.mark_end_range") as mock_end_patch,
            patch("verl.utils.profiler.mstx_profile.get_npu_profiler") as mock_get_profiler,
        ):
            decorator = mock_worker.profiler.annotate(message="test")

            @decorator
            def test_func(self, *args, **kwargs):
                return "result"

            result = test_func(mock_worker)

            self.assertEqual(result, "result")
            mock_start_patch.assert_not_called()
            mock_end_patch.assert_not_called()
            mock_get_profiler.assert_not_called()

    def test_annotate_when_this_step_disabled(self):
        mock_worker = MagicMock()
        mock_worker.profiler = DistProfiler(rank=self.rank, config=self.config, tool_config=self.tool_config)
        mock_worker.profiler._this_step = False

        with (
            patch("verl.utils.profiler.mstx_profile.mark_start_range") as mock_start_patch,
            patch("verl.utils.profiler.mstx_profile.mark_end_range") as mock_end_patch,
            patch("verl.utils.profiler.mstx_profile.get_npu_profiler") as mock_get_profiler,
        ):
            decorator = mock_worker.profiler.annotate(message="test")

            @decorator
            def test_func(self, *args, **kwargs):
                return "result"

            result = test_func(mock_worker)

            self.assertEqual(result, "result")
            mock_start_patch.assert_not_called()
            mock_end_patch.assert_not_called()
            mock_get_profiler.assert_not_called()

    def test_annotate_discrete_mode_enabled(self):
        discrete_tool_config = NPUToolConfig(discrete=True)
        mock_worker = MagicMock()
        mock_worker.profiler = DistProfiler(rank=self.rank, config=self.config, tool_config=discrete_tool_config)
        mock_worker.profiler._this_step = True

        mock_mark_range = "mocked_range_handle"
        mock_profile_npu = MagicMock()

        with (
            patch("verl.utils.profiler.mstx_profile.mark_start_range") as mock_start_patch,
            patch("verl.utils.profiler.mstx_profile.mark_end_range") as mock_end_patch,
            patch("verl.utils.profiler.mstx_profile.get_npu_profiler") as mock_get_profiler,
        ):
            mock_start_patch.return_value = mock_mark_range
            mock_get_profiler.return_value = mock_profile_npu
            decorator = mock_worker.profiler.annotate(message="test", role="test_role")

            @decorator
            def test_func(self, *args, **kwargs):
                return "result"

            result = test_func(mock_worker)

            self.assertEqual(result, "result")
            mock_start_patch.assert_called_once_with(message="test")
            mock_end_patch.assert_called_once_with(mock_mark_range)
            mock_get_profiler.assert_called_once_with(
                contents=mock_worker.profiler._impl.profile_contents,
                profile_level=mock_worker.profiler._impl.profile_level,
                profile_save_path=mock_worker.profiler._impl.profile_save_path,
                analysis=mock_worker.profiler._impl.analysis,
                role="test_role",
            )
            mock_profile_npu.start.assert_called_once()
            mock_profile_npu.step.assert_called_once()
            mock_profile_npu.stop.assert_called_once()

    def test_annotate_with_default_message(self):
        mock_worker = MagicMock()
        mock_worker.profiler = DistProfiler(rank=self.rank, config=self.config, tool_config=self.tool_config)
        mock_worker.profiler._this_step = True

        mock_mark_range = "mocked_range_handle"
        with (
            patch("verl.utils.profiler.mstx_profile.mark_start_range") as mock_start_patch,
            patch("verl.utils.profiler.mstx_profile.mark_end_range") as mock_end_patch,
        ):
            mock_start_patch.return_value = mock_mark_range
            decorator = mock_worker.profiler.annotate()

            @decorator
            def test_func(self, *args, **kwargs):
                return "result"

            test_func(mock_worker)

            mock_start_patch.assert_called_once_with(message="test_func")
            mock_end_patch.assert_called_once_with(mock_mark_range)


if __name__ == "__main__":
    unittest.main()
