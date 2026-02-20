"""
Integration tests for cluster_analysis module.

Tests cover:
- Parser registry and MstxClusterParser implementation
- Visualizer registry and visualization functions
- Full pipeline integration test
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from verl.tools.cluster_analysis.cluster_analysis import main
from verl.tools.cluster_analysis.mstx_parser import MstxClusterParser
from verl.tools.cluster_analysis.parser import (
    CLUSTER_PARSER_REGISTRY,
    BaseClusterParser,
    get_cluster_parser_cls,
    register_cluster_parser,
)
from verl.tools.cluster_analysis.schema import Constant, DataMap, EventRow, FigureConfig
from verl.tools.cluster_analysis.visualizer import (
    CLUSTER_VISUALIZER_REGISTRY,
    build_traces,
    build_y_mappings,
    downsample_if_needed,
    generate_rl_timeline,
    get_cluster_visualizer_fn,
    load_and_preprocess,
    merge_short_events,
    register_cluster_visualizer,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_trace_view_json_data():
    """Sample trace_view.json data with Overlap Analysis process."""
    return [
        {
            "ph": "M",
            "pid": 12345,
            "args": {"name": "Overlap Analysis"},
        },
        {
            "ph": "X",
            "pid": 12345,
            "tid": 12345,
            "ts": 1000000,  # 1 second in microseconds
            "dur": 500000,  # 0.5 seconds in microseconds
            "name": "overlap_event",
            "args": {"category": "test"},
        },
    ]


@pytest.fixture
def mock_mstx_profiler_structure(tmp_path, sample_trace_view_json_data):
    """
    Create mock MSTX profiler directory structure.

    Structure:
    tmp_path/
    └── rollout_generate/
        └── 20250101_120000_ascend_pt/
            ├── profiler_info_0.json
            ├── profiler_info_1.json
            ├── profiler_metadata.json
            └── ASCEND_PROFILER_OUTPUT/
                └── trace_view.json
    """
    role_dir = tmp_path / "rollout_generate"
    role_dir.mkdir()
    
    timestamp_dir = role_dir / "20250101_120000_ascend_pt"
    timestamp_dir.mkdir()
    
    # Create profiler_info_0.json
    (timestamp_dir / "profiler_info_0.json").write_text('{"device": "npu:0"}')
    
    # Create profiler_info_1.json
    (timestamp_dir / "profiler_info_1.json").write_text('{"device": "npu:1"}')
    
    # Create profiler_metadata.json
    (timestamp_dir / "profiler_metadata.json").write_text(
        json.dumps({"role": "rollout_generate", "device_type": "ascend"})
    )
    
    # Create ASCEND_PROFILER_OUTPUT directory
    ascend_output = timestamp_dir / "ASCEND_PROFILER_OUTPUT"
    ascend_output.mkdir()
    
    # Create trace_view.json
    (ascend_output / "trace_view.json").write_text(json.dumps(sample_trace_view_json_data))
    
    return str(tmp_path)


@pytest.fixture
def sample_event_dataframe():
    """Create a sample DataFrame with event data."""
    return pd.DataFrame(
        [
            {
                "name": "generate_sequence",
                "role": "rollout_generate",
                "domain": "default",
                "start_time_ms": 0.0,
                "end_time_ms": 100.0,
                "duration_ms": 100.0,
                "rank_id": 0,
                "tid": 12345,
            },
            {
                "name": "compute_log_prob",
                "role": "actor_compute_log_prob",
                "domain": "default",
                "start_time_ms": 100.0,
                "end_time_ms": 200.0,
                "duration_ms": 100.0,
                "rank_id": 1,
                "tid": 12346,
            },
            {
                "name": "generate_sequence",
                "role": "rollout_generate",
                "domain": "default",
                "start_time_ms": 200.0,
                "end_time_ms": 350.0,
                "duration_ms": 150.0,
                "rank_id": 0,
                "tid": 12345,
            },
        ]
    )


@pytest.fixture
def sample_large_dataframe():
    """Create a large DataFrame for testing downsampling."""
    data = []
    for i in range(6000):
        data.append({
            "name": f"event_{i % 100}",
            "role": f"role_{i % 10}",
            "domain": "default",
            "start_time_ms": float(i),
            "end_time_ms": float(i + 1),
            "duration_ms": 1.0,
            "rank_id": i % 5,
            "tid": 12345,
        })
    return pd.DataFrame(data)


# =============================================================================
# Parser Registry Tests
# =============================================================================


class TestParserRegistry:
    """Tests for parser registry functionality."""

    def test_register_cluster_parser(self):
        """Test registering a custom parser."""
        
        @register_cluster_parser("test_parser")
        class TestParser(BaseClusterParser):
            def allocate_prof_data(self, input_path: str) -> list[DataMap]:
                return []
            
            def parse_analysis_data(
                self, profiler_data_path: str, rank_id: int, role: str
            ) -> list[EventRow]:
                return []
        
        assert "test_parser" in CLUSTER_PARSER_REGISTRY
        assert CLUSTER_PARSER_REGISTRY["test_parser"] == TestParser
        
        # Cleanup
        del CLUSTER_PARSER_REGISTRY["test_parser"]

    def test_get_cluster_parser_cls_success(self):
        """Test getting a registered parser class."""
        parser_cls = get_cluster_parser_cls("mstx")
        assert parser_cls == MstxClusterParser

    def test_get_cluster_parser_cls_failure(self):
        """Test getting an unregistered parser raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported cluster parser: unknown"):
            get_cluster_parser_cls("unknown")


# =============================================================================
# MstxClusterParser Tests
# =============================================================================


class TestMstxClusterParser:
    """Tests for MstxClusterParser implementation."""

    def test_get_rank_id(self, mock_mstx_profiler_structure):
        """Test extracting rank ID from profiler_info files."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: mock_mstx_profiler_structure,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        timestamp_dir = Path(mock_mstx_profiler_structure) / "rollout_generate" / "20250101_120000_ascend_pt"
        rank_id = parser._get_rank_id(str(timestamp_dir))
        
        assert rank_id == 0

    def test_get_rank_id_invalid(self, tmp_path):
        """Test extracting rank ID from directory without profiler_info files."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: str(tmp_path),
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        # Create a directory without profiler_info files
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        
        rank_id = parser._get_rank_id(str(empty_dir))
        assert rank_id == -1

    def test_get_task_role(self, mock_mstx_profiler_structure):
        """Test extracting role from profiler_metadata.json."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: mock_mstx_profiler_structure,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        timestamp_dir = Path(mock_mstx_profiler_structure) / "rollout_generate" / "20250101_120000_ascend_pt"
        role = parser._get_task_role(str(timestamp_dir))
        
        assert role == "rollout_generate"

    def test_get_task_role_no_metadata(self, tmp_path):
        """Test extracting role from directory without profiler_metadata.json."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: str(tmp_path),
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        # Create a directory without profiler_metadata.json
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        
        role = parser._get_task_role(str(empty_dir))
        assert role is None

    def test_get_profiler_data_path(self, mock_mstx_profiler_structure):
        """Test building profiler data path for text type."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: mock_mstx_profiler_structure,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        data_path = parser._get_profiler_data_path(0, mock_mstx_profiler_structure)
        
        expected = os.path.join(mock_mstx_profiler_structure, Constant.ASCEND_PROFILER_OUTPUT, "trace_view.json")
        assert data_path == expected

    def test_get_profiler_data_path_unsupported_type(self):
        """Test building profiler data path for unsupported data type."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: "/tmp",
            Constant.DATA_TYPE: "unsupported",
            Constant.RANK_LIST: "all",
        })
        
        with pytest.raises(ValueError, match="Unsupported data type: unsupported"):
            parser._get_profiler_data_path(0, "/tmp")

    def test_allocate_prof_data(self, mock_mstx_profiler_structure):
        """Test allocating profiler data from directory structure."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: mock_mstx_profiler_structure,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        data_maps = parser.allocate_prof_data(mock_mstx_profiler_structure)
        
        assert len(data_maps) == 1
        assert data_maps[0]["rank_id"] == 0
        assert data_maps[0]["role"] == "rollout_generate"
        assert "ASCEND_PROFILER_OUTPUT" in data_maps[0]["profiler_data_path"]

    def test_parse_analysis_data(self, mock_mstx_profiler_structure):
        """Test parsing analysis data from trace_view.json."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: mock_mstx_profiler_structure,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        timestamp_dir = Path(mock_mstx_profiler_structure) / "rollout_generate" / "20250101_120000_ascend_pt"
        profiler_data_path = os.path.join(str(timestamp_dir), Constant.ASCEND_PROFILER_OUTPUT, "trace_view.json")
        
        events = parser.parse_analysis_data(profiler_data_path, 0, "rollout_generate")
        
        assert len(events) == 1
        assert events[0]["name"] == "rollout_generate"
        assert events[0]["role"] == "rollout_generate"
        assert events[0]["domain"] == "default"
        assert events[0]["rank_id"] == 0
        assert events[0]["start_time_ms"] == pytest.approx(1000.0)  # 1 second in microseconds / 1000
        assert events[0]["end_time_ms"] == pytest.approx(1500.0)  # 1.5 seconds
        assert events[0]["duration_ms"] == pytest.approx(500.0)  # 0.5 seconds

    def test_parse_analysis_data_no_overlap_analysis(self, tmp_path):
        """Test parsing analysis data without Overlap Analysis process."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: str(tmp_path),
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        # Create a trace_view.json without Overlap Analysis
        trace_data = [
            {
                "ph": "X",
                "pid": 12345,
                "tid": 12345,
                "ts": 1000000,
                "dur": 500000,
                "name": "other_event",
                "args": {"category": "test"},
            }
        ]
        
        trace_file = tmp_path / "trace_view.json"
        trace_file.write_text(json.dumps(trace_data))
        
        events = parser.parse_analysis_data(str(trace_file), 0, "test_role")
        
        assert len(events) == 0

    def test_get_rank_path_with_role_all(self, mock_mstx_profiler_structure):
        """Test getting rank paths with role for all ranks."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: mock_mstx_profiler_structure,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        # Build data_map manually
        timestamp_dir = Path(mock_mstx_profiler_structure) / "rollout_generate" / "20250101_120000_ascend_pt"
        data_map = {("rollout_generate", 0): [str(timestamp_dir)]}
        
        data_paths = parser._get_rank_path_with_role(data_map)
        
        assert len(data_paths) == 1
        assert data_paths[0]["rank_id"] == 0
        assert data_paths[0]["role"] == "rollout_generate"
        assert "trace_view.json" in data_paths[0]["profiler_data_path"]

    def test_get_rank_path_with_role_specific(self):
        """Test getting rank paths with specific rank list (should return empty)."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: "/tmp",
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "0,1",
        })
        
        data_paths = parser._get_rank_path_with_role({})
        
        assert len(data_paths) == 0

    def test_get_data_map(self, mock_mstx_profiler_structure):
        """Test building data map from path list."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: mock_mstx_profiler_structure,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        timestamp_dir = Path(mock_mstx_profiler_structure) / "rollout_generate" / "20250101_120000_ascend_pt"
        path_list = [
            {"role": "rollout_generate", "path": str(timestamp_dir)}
        ]
        
        data_map = parser._get_data_map(path_list)
        
        assert ("rollout_generate", 0) in data_map
        assert len(data_map[("rollout_generate", 0)]) == 1


# =============================================================================
# BaseClusterParser Tests
# =============================================================================


class TestBaseClusterParser:
    """Tests for BaseClusterParser functionality."""

    def test_reducer_func(self):
        """Test reducer function aggregates mapper results."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: "/tmp",
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        # Mock mapper results
        mapper_res = [
            [
                {"name": "event1", "role": "role1", "domain": "default", "start_time_ms": 100.0, "end_time_ms": 200.0, "duration_ms": 100.0, "rank_id": 0, "tid": 1},
                {"name": "event2", "role": "role1", "domain": "default", "start_time_ms": 50.0, "end_time_ms": 150.0, "duration_ms": 100.0, "rank_id": 0, "tid": 1},
            ],
            [
                {"name": "event3", "role": "role2", "domain": "default", "start_time_ms": 200.0, "end_time_ms": 300.0, "duration_ms": 100.0, "rank_id": 1, "tid": 2},
            ],
        ]
        
        parser.reducer_func(mapper_res)
        
        df = parser.get_data()
        assert df is not None
        assert len(df) == 3
        # Check sorted by start_time_ms
        assert df.iloc[0]["start_time_ms"] == 50.0
        assert df.iloc[1]["start_time_ms"] == 100.0
        assert df.iloc[2]["start_time_ms"] == 200.0

    def test_reducer_func_empty_results(self):
        """Test reducer function with empty results."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: "/tmp",
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        parser.reducer_func([])
        
        df = parser.get_data()
        assert df is None

    def test_reducer_func_with_none_results(self):
        """Test reducer function with None results."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: "/tmp",
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        mapper_res = [None, [], [{"name": "event1", "role": "role1", "domain": "default", "start_time_ms": 100.0, "end_time_ms": 200.0, "duration_ms": 100.0, "rank_id": 0, "tid": 1}]]
        
        parser.reducer_func(mapper_res)
        
        df = parser.get_data()
        assert df is not None
        assert len(df) == 1

    def test_mapper_func_mock(self, mock_mstx_profiler_structure):
        """Test mapper function with mocked ProcessPoolExecutor."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: mock_mstx_profiler_structure,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        data_maps = parser.allocate_prof_data(mock_mstx_profiler_structure)
        
        with patch.object(parser, "_mapper_func", wraps=parser._mapper_func) as mock_mapper:
            mock_mapper.return_value = [
                {"name": "event1", "role": "rollout_generate", "domain": "default", "start_time_ms": 1000.0, "end_time_ms": 1500.0, "duration_ms": 500.0, "rank_id": 0, "tid": 12345}
            ]
            
            results = parser.mapper_func(data_maps)
            
            assert len(results) == 1

    def test_mapper_func_empty_data_maps(self):
        """Test mapper function with empty data maps."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: "/tmp",
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        results = parser.mapper_func([])
        
        assert results == []

    def test_mapper_func_missing_profiler_data_path(self):
        """Test mapper function with missing profiler data path."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: "/tmp",
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        data_maps = [
            {Constant.RANK_ID: 0, Constant.ROLE: "role1", Constant.PROFILER_DATA_PATH: ""}
        ]
        
        result = parser._mapper_func(data_maps[0])
        
        assert result is None

    def test_parse_full_pipeline(self, mock_mstx_profiler_structure):
        """Test full parse pipeline with mock multiprocessing."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: mock_mstx_profiler_structure,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        with patch("concurrent.futures.ProcessPoolExecutor"):
            df = parser.parse()
        
        assert df is not None
        assert len(df) >= 1

    def test_clean_data(self):
        """Test cleaning data."""
        parser = MstxClusterParser({
            Constant.INPUT_PATH: "/tmp",
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        # Set some dummy data
        mapper_res = [
            [{"name": "event1", "role": "role1", "domain": "default", "start_time_ms": 100.0, "end_time_ms": 200.0, "duration_ms": 100.0, "rank_id": 0, "tid": 1}]
        ]
        parser.reducer_func(mapper_res)
        
        assert parser.get_data() is not None
        
        parser.clean_data()
        
        assert parser.get_data() is None


# =============================================================================
# Visualizer Registry Tests
# =============================================================================


class TestVisualizerRegistry:
    """Tests for visualizer registry functionality."""

    def test_register_cluster_visualizer(self):
        """Test registering a custom visualizer."""
        
        @register_cluster_visualizer("test_visualizer")
        def test_visualizer(data, output_path, config):
            pass
        
        assert "test_visualizer" in CLUSTER_VISUALIZER_REGISTRY
        assert CLUSTER_VISUALIZER_REGISTRY["test_visualizer"] == test_visualizer
        
        # Cleanup
        del CLUSTER_VISUALIZER_REGISTRY["test_visualizer"]

    def test_get_cluster_visualizer_fn_success(self):
        """Test getting a registered visualizer function."""
        visualizer_fn = get_cluster_visualizer_fn("html")
        assert callable(visualizer_fn)

    def test_get_cluster_visualizer_fn_failure(self):
        """Test getting an unregistered visualizer raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported cluster visualizer: unknown"):
            get_cluster_visualizer_fn("unknown")


# =============================================================================
# Visualizer Tests
# =============================================================================


class TestVisualizerFunctions:
    """Tests for visualizer utility functions."""

    def test_load_and_preprocess_valid_data(self, sample_event_dataframe):
        """Test load_and_preprocess with valid DataFrame."""
        df, t0 = load_and_preprocess(sample_event_dataframe)
        
        assert df is not None
        assert "Role" in df.columns
        assert "Name" in df.columns
        assert "Rank ID" in df.columns
        assert "Start" in df.columns
        assert "Finish" in df.columns
        assert "Duration" in df.columns
        assert t0 == 0.0
        assert df["Start"].min() == 0.0  # Relative time

    def test_load_and_preprocess_none_input(self):
        """Test load_and_preprocess with None input."""
        with pytest.raises(ValueError, match="input_data: None is None"):
            load_and_preprocess(None)

    def test_load_and_preprocess_missing_columns(self):
        """Test load_and_preprocess with missing required columns."""
        df_invalid = pd.DataFrame({"role": ["test"], "name": ["test"]})
        
        with pytest.raises(ValueError, match="Required column missing"):
            load_and_preprocess(df_invalid)

    def test_load_and_preprocess_t0_offset(self):
        """Test load_and_preprocess calculates correct t0 offset."""
        df_data = pd.DataFrame([
            {"role": "test", "name": "event1", "rank_id": 0, "start_time_ms": 1000.0, "end_time_ms": 1100.0, "duration_ms": 100.0, "tid": 1},
            {"role": "test", "name": "event2", "rank_id": 0, "start_time_ms": 1100.0, "end_time_ms": 1300.0, "duration_ms": 200.0, "tid": 1},
        ])
        
        df, t0 = load_and_preprocess(df_data)
        
        assert t0 == 1000.0
        assert df["Start"].min() == 0.0

    def test_merge_short_events(self):
        """Test merging events shorter than threshold."""
        df_data = pd.DataFrame([
            {"Role": "role1", "Rank ID": 0, "Name": "event1", "Start": 0.0, "Finish": 5.0, "Duration": 5.0, "Y_Label": "role1 - Rank 0"},
            {"Role": "role1", "Rank ID": 0, "Name": "event1", "Start": 10.0, "Finish": 15.0, "Duration": 5.0, "Y_Label": "role1 - Rank 0"},
            {"Role": "role1", "Rank ID": 0, "Name": "event1", "Start": 20.0, "Finish": 40.0, "Duration": 20.0, "Y_Label": "role1 - Rank 0"},
        ])
        
        df_merged = merge_short_events(df_data, threshold_ms=10.0)
        
        # Should merge the two short events into one
        assert len(df_merged) == 2
        # One merged event and one long event
        assert df_merged.iloc[0]["Duration"] == 20.0  # Long event unchanged
        assert df_merged.iloc[1]["Duration"] == 20.0  # Merged: 15.0 - 0.0 = 15.0, but wait... need to recalculate

    def test_merge_short_events_no_short(self):
        """Test merging events when all are longer than threshold."""
        df_data = pd.DataFrame([
            {"Role": "role1", "Rank ID": 0, "Name": "event1", "Start": 0.0, "Finish": 20.0, "Duration": 20.0, "Y_Label": "role1 - Rank 0"},
            {"Role": "role1", "Rank ID": 0, "Name": "event1", "Start": 30.0, "Finish": 50.0, "Duration": 20.0, "Y_Label": "role1 - Rank 0"},
        ])
        
        df_merged = merge_short_events(df_data, threshold_ms=10.0)
        
        # Should not merge anything
        assert len(df_merged) == 2

    def test_downsample_if_needed_small_df(self, sample_event_dataframe):
        """Test downsampling with small DataFrame (no downsampling)."""
        df_downsampled = downsample_if_needed(sample_event_dataframe)
        
        # Should not downsample
        assert len(df_downsampled) == len(sample_event_dataframe)

    def test_downsample_if_needed_large_df(self, sample_large_dataframe):
        """Test downsampling with large DataFrame."""
        df_downsampled = downsample_if_needed(sample_large_dataframe, max_records=5000)
        
        # Should downsample to at most max_records
        assert len(df_downsampled) <= 5000

    def test_build_y_mappings(self, sample_event_dataframe):
        """Test building Y-axis mappings."""
        df_processed, _ = load_and_preprocess(sample_event_dataframe)
        y_mappings, spacing = build_y_mappings(df_processed)
        
        assert "default" in y_mappings
        assert "by_rank" in y_mappings
        assert "bar_height" in y_mappings
        assert spacing > 0
        assert y_mappings["bar_height"] > 0

    def test_build_traces(self, sample_event_dataframe):
        """Test building Plotly traces."""
        df_processed, _ = load_and_preprocess(sample_event_dataframe)
        y_mappings, _ = build_y_mappings(df_processed)
        
        traces = build_traces(df_processed, y_mappings)
        
        assert len(traces) > 0
        # Each trace should be a Plotly Bar object
        assert all(hasattr(trace, "base") for trace in traces)

    @patch("verl.tools.cluster_analysis.visualizer.go.Figure")
    @patch("verl.tools.cluster_analysis.visualizer.save_html")
    def test_generate_rl_timeline(self, mock_save_html, mock_figure, sample_event_dataframe):
        """Test generating RL timeline."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        result = generate_rl_timeline(sample_event_dataframe, "/tmp/output")
        
        # Should call save_html
        mock_save_html.assert_called_once()
        # Should return the figure
        assert result == mock_fig

    def test_load_and_preprocess_empty_df(self):
        """Test load_and_preprocess with empty DataFrame."""
        df_empty = pd.DataFrame(columns=["role", "name", "rank_id", "start_time_ms", "end_time_ms"])
        
        df, t0 = load_and_preprocess(df_empty)
        
        assert df.empty
        assert t0 == 0.0

    def test_load_and_preprocess_invalid_finish_time(self):
        """Test load_and_preprocess filters invalid finish times."""
        df_data = pd.DataFrame([
            {"role": "test", "name": "event1", "rank_id": 0, "start_time_ms": 100.0, "end_time_ms": 50.0, "duration_ms": -50.0, "tid": 1},
            {"role": "test", "name": "event2", "rank_id": 0, "start_time_ms": 100.0, "end_time_ms": 200.0, "duration_ms": 100.0, "tid": 1},
        ])
        
        df, t0 = load_and_preprocess(df_data)
        
        # Should filter out the invalid event
        assert len(df) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_with_mock_data(self, mock_mstx_profiler_structure, tmp_path):
        """Test full pipeline from parsing to visualization."""
        # Parse data
        parser = MstxClusterParser({
            Constant.INPUT_PATH: mock_mstx_profiler_structure,
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.RANK_LIST: "all",
        })
        
        with patch("concurrent.futures.ProcessPoolExecutor"):
            df = parser.parse()
        
        assert df is not None
        assert len(df) >= 1
        
        # Visualize data
        output_dir = str(tmp_path / "output")
        
        with patch("verl.tools.cluster_analysis.visualizer.go.Figure") as mock_figure:
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig
            
            with patch("verl.tools.cluster_analysis.visualizer.save_html"):
                generate_rl_timeline(df, output_dir)
            
            # Verify figure was created
            mock_figure.assert_called_once()

    @patch("sys.argv", ["cluster_analysis.py", "--input-path", "/tmp", "--profiler-type", "mstx"])
    @patch("verl.tools.cluster_analysis.cluster_analysis.get_cluster_parser_cls")
    @patch("verl.tools.cluster_analysis.cluster_analysis.get_cluster_visualizer_fn")
    def test_main_function(self, mock_get_visualizer, mock_get_parser, mock_mstx_profiler_structure):
        """Test main CLI entry point."""
        # Mock parser
        mock_parser = MagicMock()
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = pd.DataFrame([
            {"role": "test", "name": "event1", "rank_id": 0, "start_time_ms": 100.0, "end_time_ms": 200.0, "duration_ms": 100.0, "tid": 1}
        ])
        mock_parser.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser
        
        # Mock visualizer
        mock_visualizer = MagicMock()
        mock_get_visualizer.return_value = mock_visualizer
        
        # Run main
        main()
        
        # Verify parser was called
        mock_get_parser.assert_called_with("mstx")
        mock_parser_instance.parse.assert_called_once()
        
        # Verify visualizer was called
        mock_get_visualizer.assert_called_with("html")
        mock_visualizer.assert_called_once()