"""
Integration tests for cluster_analysis module.

These tests verify the end-to-end functionality of parsing profiling data
and generating visualizations.
"""
import json
import os
import tempfile
import shutil
from pathlib import Path

import pytest

from verl.tools.cluster_analysis.mstx_parser import MstxClusterParser
from verl.tools.cluster_analysis.schema import Constant
from verl.tools.cluster_analysis.visualizer import generate_rl_timeline


class TestClusterAnalysis:
    """Integration tests for cluster analysis workflow."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_profiler_data(self, temp_dir):
        """Create mock MSTX profiler data directory structure."""
        # Create directory structure: /temp/actor/ascend_pt_rank_0_step_1/
        base_dir = Path(temp_dir) / "actor" / "ascend_pt_rank_0_step_1"
        base_dir.mkdir(parents=True)

        # Create ASCEND_PROFILER_OUTPUT directory
        profiler_output = base_dir / Constant.ASCEND_PROFILER_OUTPUT
        profiler_output.mkdir()

        # Create profiler_info_0.json
        profiler_info = base_dir / f"{Constant.ASCEND_PROFILER_INFO_HEAD}0{Constant.JSON_EXTENSION}"
        profiler_info.write_text("{}")

        # Create profiler_metadata.json
        profiler_metadata = base_dir / Constant.ASCEND_PROFILER_METADATA_JSON
        profiler_metadata.write_text(json.dumps({"roll": "actor"}))

        # Create trace_view.json with overlap analysis data
        trace_data = [
            {
                "ph": "M",
                "pid": 456,
                "args": {"name": "Overlap Analysis"}
            },
            {
                "ph": "X",
                "pid": 456,
                "ts": 1000000000,  # nanoseconds
                "dur": 500000000,  # nanoseconds
                "args": {"data": "test"}
            }
        ]
        trace_view = profiler_output / "trace_view.json"
        trace_view.write_text(json.dumps(trace_data))

        return temp_dir

    @pytest.fixture
    def mock_profiler_data_multiple_ranks(self, temp_dir):
        """Create mock MSTX profiler data with multiple ranks."""
        base_dir = Path(temp_dir)

        for rank_id in [0, 1, 2]:
            # Create directory for each rank
            rank_dir = base_dir / "actor" / f"ascend_pt_rank_{rank_id}_step_1"
            rank_dir.mkdir(parents=True)

            # Create ASCEND_PROFILER_OUTPUT directory
            profiler_output = rank_dir / Constant.ASCEND_PROFILER_OUTPUT
            profiler_output.mkdir()

            # Create profiler_info_X.json
            profiler_info = rank_dir / f"{Constant.ASCEND_PROFILER_INFO_HEAD}{rank_id}{Constant.JSON_EXTENSION}"
            profiler_info.write_text("{}")

            # Create profiler_metadata.json
            profiler_metadata = rank_dir / Constant.ASCEND_PROFILER_METADATA_JSON
            profiler_metadata.write_text(json.dumps({"roll": "actor"}))

            # Create trace_view.json with overlap analysis data
            # Each rank has different timing
            start_ts = 1000000000 + rank_id * 200000000  # Stagger start times
            trace_data = [
                {
                    "ph": "M",
                    "pid": 456 + rank_id,
                    "args": {"name": "Overlap Analysis"}
                },
                {
                    "ph": "X",
                    "pid": 456 + rank_id,
                    "ts": start_ts,
                    "dur": 500000000,
                    "args": {"data": f"rank_{rank_id}"}
                }
            ]
            trace_view = profiler_output / "trace_view.json"
            trace_view.write_text(json.dumps(trace_data))

        return temp_dir

    @pytest.fixture
    def mock_profiler_data_with_events(self, temp_dir):
        """Create mock MSTX profiler data with RL events."""
        base_dir = Path(temp_dir) / "actor" / "ascend_pt_rank_0_step_1"
        base_dir.mkdir(parents=True)

        # Create ASCEND_PROFILER_OUTPUT directory
        profiler_output = base_dir / Constant.ASCEND_PROFILER_OUTPUT
        profiler_output.mkdir()

        # Create profiler_info_0.json
        profiler_info = base_dir / f"{Constant.ASCEND_PROFILER_INFO_HEAD}0{Constant.JSON_EXTENSION}"
        profiler_info.write_text("{}")

        # Create profiler_metadata.json
        profiler_metadata = base_dir / Constant.ASCEND_PROFILER_METADATA_JSON
        profiler_metadata.write_text(json.dumps({"roll": "actor"}))

        # Create trace_view.json with RL events
        trace_data = [
            {
                "name": "forward_pass",
                "ts": 100000,  # microseconds
                "dur": 50000,
                "tid": 123,
                "args": {
                    "event_type": "compute",
                    "domain": "default"
                }
            },
            {
                "name": "backward_pass",
                "ts": 150000,
                "dur": 30000,
                "tid": 123,
                "args": {
                    "event_type": "compute",
                    "domain": "default"
                }
            },
            {
                "name": "optimizer_step",
                "ts": 180000,
                "dur": 20000,
                "tid": 123,
                "args": {
                    "event_type": "update",
                    "domain": "default"
                }
            }
        ]
        trace_view = profiler_output / "trace_view.json"
        trace_view.write_text(json.dumps(trace_data))

        return temp_dir

    def test_end_to_end_single_rank(self, mock_profiler_data, temp_dir):
        """Test end-to-end workflow with single rank."""
        # Setup parser configuration
        params = {
            Constant.INPUT_PATH: mock_profiler_data,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        }

        # Parse data
        parser = MstxClusterParser(params)
        data = parser.parse()

        # Verify parsed data
        assert data is not None
        assert len(data) == 1
        assert data.iloc[0]["roll"] == "actor"
        assert data.iloc[0]["domain"] == "default"
        assert data.iloc[0]["rank_id"] == 0

    def test_end_to_end_multiple_ranks(self, mock_profiler_data_multiple_ranks, temp_dir):
        """Test end-to-end workflow with multiple ranks."""
        # Setup parser configuration
        params = {
            Constant.INPUT_PATH: mock_profiler_data_multiple_ranks,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        }

        # Parse data
        parser = MstxClusterParser(params)
        data = parser.parse()

        # Verify parsed data
        assert data is not None
        assert len(data) == 3

        # Verify all ranks are present
        rank_ids = sorted(data["rank_id"].unique())
        assert rank_ids == [0, 1, 2]

        # Verify data is sorted by start_time_ms
        for i in range(len(data) - 1):
            assert data.iloc[i]["start_time_ms"] <= data.iloc[i + 1]["start_time_ms"]

    def test_end_to_end_visualization(self, mock_profiler_data, temp_dir):
        """Test end-to-end workflow with visualization."""
        # Setup parser configuration
        params = {
            Constant.INPUT_PATH: mock_profiler_data,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        }

        # Parse data
        parser = MstxClusterParser(params)
        data = parser.parse()

        # Generate visualization
        output_dir = os.path.join(temp_dir, "output")
        fig = generate_rl_timeline(data, output_dir)

        # Verify figure was created
        assert fig is not None

        # Verify HTML file was created
        html_file = os.path.join(output_dir, "rl_timeline.html")
        assert os.path.exists(html_file)

    def test_end_to_end_with_custom_filename(self, mock_profiler_data, temp_dir):
        """Test end-to-end workflow with custom output filename."""
        # Setup parser configuration
        params = {
            Constant.INPUT_PATH: mock_profiler_data,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        }

        # Parse data
        parser = MstxClusterParser(params)
        data = parser.parse()

        # Generate visualization with custom filename
        output_dir = os.path.join(temp_dir, "output")
        custom_filename = "custom_timeline.html"
        fig = generate_rl_timeline(data, output_dir, output_filename=custom_filename)

        # Verify figure was created
        assert fig is not None

        # Verify custom HTML file was created
        html_file = os.path.join(output_dir, custom_filename)
        assert os.path.exists(html_file)

    def test_end_to_end_with_different_data_types(self, mock_profiler_data, temp_dir):
        """Test end-to-end workflow with different data type configurations."""
        # Test with TEXT type
        params_text = {
            Constant.INPUT_PATH: mock_profiler_data,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        }
        parser_text = MstxClusterParser(params_text)
        data_text = parser_text.parse()

        assert data_text is not None
        assert len(data_text) == 1

    def test_end_to_end_empty_directory(self, temp_dir):
        """Test end-to-end workflow with empty directory."""
        # Create empty directory
        empty_dir = os.path.join(temp_dir, "empty")
        os.makedirs(empty_dir)

        # Setup parser configuration
        params = {
            Constant.INPUT_PATH: empty_dir,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        }

        # Parse data
        parser = MstxClusterParser(params)
        data = parser.parse()

        # Verify no data was parsed
        assert data is not None
        assert len(data) == 0

    def test_end_to_end_missing_profiler_data(self, temp_dir):
        """Test end-to-end workflow when profiler data files are missing."""
        # Create directory structure without trace_view.json
        base_dir = Path(temp_dir) / "actor" / "ascend_pt_rank_0_step_1"
        base_dir.mkdir(parents=True)

        profiler_output = base_dir / Constant.ASCEND_PROFILER_OUTPUT
        profiler_output.mkdir()

        profiler_info = base_dir / f"{Constant.ASCEND_PROFILER_INFO_HEAD}0{Constant.JSON_EXTENSION}"
        profiler_info.write_text("{}")

        profiler_metadata = base_dir / Constant.ASCEND_PROFILER_METADATA_JSON
        profiler_metadata.write_text(json.dumps({"roll": "actor"}))

        # Setup parser configuration
        params = {
            Constant.INPUT_PATH: temp_dir,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        }

        # Parse data
        parser = MstxClusterParser(params)
        data = parser.parse()

        # Verify no data was parsed due to missing trace_view.json
        assert data is not None
        assert len(data) == 0

    def test_end_to_end_multiple_roles(self, temp_dir):
        """Test end-to-end workflow with multiple roles (actor, critic)."""
        base_dir = Path(temp_dir)

        for role in ["actor", "critic"]:
            # Create directory for each role
            role_dir = base_dir / role / f"ascend_pt_rank_0_step_1"
            role_dir.mkdir(parents=True)

            # Create profiler output directory
            profiler_output = role_dir / Constant.ASCEND_PROFILER_OUTPUT
            profiler_output.mkdir()

            # Create profiler_info_0.json
            profiler_info = role_dir / f"{Constant.ASCEND_PROFILER_INFO_HEAD}0{Constant.JSON_EXTENSION}"
            profiler_info.write_text("{}")

            # Create profiler_metadata.json
            profiler_metadata = role_dir / Constant.ASCEND_PROFILER_METADATA_JSON
            profiler_metadata.write_text(json.dumps({"roll": role}))

            # Create trace_view.json
            trace_data = [
                {
                    "ph": "M",
                    "pid": 456,
                    "args": {"name": "Overlap Analysis"}
                },
                {
                    "ph": "X",
                    "pid": 456,
                    "ts": 1000000000,
                    "dur": 500000000,
                    "args": {"data": role}
                }
            ]
            trace_view = profiler_output / "trace_view.json"
            trace_view.write_text(json.dumps(trace_data))

        # Setup parser configuration
        params = {
            Constant.INPUT_PATH: temp_dir,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        }

        # Parse data
        parser = MstxClusterParser(params)
        data = parser.parse()

        # Verify both roles are present
        assert data is not None
        assert len(data) == 2

        roles = sorted(data["roll"].unique())
        assert roles == ["actor", "critic"]

    def test_end_to_end_large_dataset(self, temp_dir):
        """Test end-to-end workflow with larger dataset."""
        base_dir = Path(temp_dir) / "actor" / "ascend_pt_rank_0_step_1"
        base_dir.mkdir(parents=True)

        profiler_output = base_dir / Constant.ASCEND_PROFILER_OUTPUT
        profiler_output.mkdir()

        profiler_info = base_dir / f"{Constant.ASCEND_PROFILER_INFO_HEAD}0{Constant.JSON_EXTENSION}"
        profiler_info.write_text("{}")

        profiler_metadata = base_dir / Constant.ASCEND_PROFILER_METADATA_JSON
        profiler_metadata.write_text(json.dumps({"roll": "actor"}))

        # Create trace data with many events
        trace_data = [
            {
                "ph": "M",
                "pid": 456,
                "args": {"name": "Overlap Analysis"}
            }
        ]

        # Add many time intervals
        for i in range(10):
            trace_data.append({
                "ph": "X",
                "pid": 456,
                "ts": 1000000000 + i * 100000000,
                "dur": 50000000,
                "args": {"data": f"event_{i}"}
            })

        trace_view = profiler_output / "trace_view.json"
        trace_view.write_text(json.dumps(trace_data))

        # Setup parser configuration
        params = {
            Constant.INPUT_PATH: temp_dir,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        }

        # Parse data
        parser = MstxClusterParser(params)
        data = parser.parse()

        # Verify data was parsed
        assert data is not None
        assert len(data) == 1

        # Generate visualization to ensure it handles larger datasets
        output_dir = os.path.join(temp_dir, "output")
        fig = generate_rl_timeline(data, output_dir)

        assert fig is not None
        assert os.path.exists(os.path.join(output_dir, "rl_timeline.html"))

    def test_end_to_end_invalid_json(self, temp_dir):
        """Test end-to-end workflow with invalid JSON data."""
        base_dir = Path(temp_dir) / "actor" / "ascend_pt_rank_0_step_1"
        base_dir.mkdir(parents=True)

        profiler_output = base_dir / Constant.ASCEND_PROFILER_OUTPUT
        profiler_output.mkdir()

        profiler_info = base_dir / f"{Constant.ASCEND_PROFILER_INFO_HEAD}0{Constant.JSON_EXTENSION}"
        profiler_info.write_text("{}")

        profiler_metadata = base_dir / Constant.ASCEND_PROFILER_METADATA_JSON
        profiler_metadata.write_text(json.dumps({"roll": "actor"}))

        # Create invalid JSON file
        trace_view = profiler_output / "trace_view.json"
        trace_view.write_text("{ invalid json }")

        # Setup parser configuration
        params = {
            Constant.INPUT_PATH: temp_dir,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        }

        # Parse data should handle error gracefully
        parser = MstxClusterParser(params)
        with pytest.raises(Exception):  # JSON parse error
            parser.parse()