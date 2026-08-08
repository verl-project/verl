# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from unittest.mock import Mock, patch

import pytest
import ray

from verl.single_controller.ray.topology_aware import (
    build_tree_topo,
    get_node_info,
    topology_aware_schedule,
)


@pytest.fixture
def init_ray():
    """Initialize Ray for testing."""
    ray.init(num_cpus=4)
    yield
    ray.shutdown()


class TestTreeTopology:
    """Test TreeTopology functionality."""

    def test_build_tree_topo_empty(self):
        """Test build_tree_topo with no node labels."""
        with patch("ray.nodes", return_value=[]):
            result = build_tree_topo()
            assert result is None

    def test_build_tree_topo_no_l0(self, init_ray):
        """Test build_tree_topo when nodes don't have L0 labels."""
        mock_nodes = [
            {"Labels": {}},
            {"Labels": {"L1": "A:1"}},
        ]
        with patch("ray.nodes", return_value=mock_nodes):
            result = build_tree_topo()
            assert result is None

    def test_build_tree_topo_basic(self, init_ray):
        """Test build_tree_topo with valid node labels."""
        mock_nodes = [
            {"Labels": {"L0": "node1:0", "L1": "A:1", "L2": "X:2"}},
            {"Labels": {"L0": "node2:0", "L1": "B:1", "L2": "Y:2"}},
        ]
        with patch("ray.nodes", return_value=mock_nodes):
            result = build_tree_topo()
            assert result is not None
            assert hasattr(result, "root")
            assert result.root.layer == 3


class TestGetNodeInfo:
    """Test get_node_info function."""

    def test_get_node_info_empty(self, init_ray):
        """Test get_node_info with no nodes."""
        with patch("ray.nodes", return_value=[]):
            node_mapping, height = get_node_info()
            assert node_mapping == {}
            assert height == 0

    def test_get_node_info_no_labels(self, init_ray):
        """Test get_node_info with nodes that have no labels."""
        mock_nodes = [{"Labels": {}}, {"Labels": {}}]
        with patch("ray.nodes", return_value=mock_nodes):
            node_mapping, height = get_node_info()
            assert node_mapping == {}
            assert height == 0

    def test_get_node_info_single_node(self, init_ray):
        """Test get_node_info with a single node."""
        mock_nodes = [
            {
                "Labels": {
                    "L0": "node1",
                    "L1": "A",
                    "L2": "X:2",
                }
            }
        ]
        with patch("ray.nodes", return_value=mock_nodes):
            node_mapping, height = get_node_info()
            assert len(node_mapping) == 1
            assert "node1" in node_mapping
            assert height == 3
            assert node_mapping["node1"]["Labels"] == ["X", "A", "node1"]
            assert node_mapping["node1"]["Blocks"] == [2, 1, 1]

    def test_get_node_info_multiple_nodes(self, init_ray):
        """Test get_node_info with multiple nodes."""
        mock_nodes = [
            {
                "Labels": {
                    "L0": "node1",
                    "L1": "A:1",
                    "L2": "X:2",
                }
            },
            {
                "Labels": {
                    "L0": "node2",
                    "L1": "B:1",
                    "L2": "Y:3",
                }
            },
        ]
        with patch("ray.nodes", return_value=mock_nodes):
            node_mapping, height = get_node_info()
            assert len(node_mapping) == 2
            assert height == 3

    def test_get_node_info_invalid_block_size(self, init_ray):
        """Test get_node_info with invalid block size values."""
        mock_nodes = [
            {
                "Labels": {
                    "L0": "node1",
                    "L1": "A:invalid",
                    "L2": "X:2",
                }
            }
        ]
        with patch("ray.nodes", return_value=mock_nodes):
            node_mapping, height = get_node_info()
            assert len(node_mapping) == 1
            assert height == 3


class TestTopologyAwareSchedule:
    """Test topology_aware_schedule function."""

    def test_schedule_with_none_pgs(self, init_ray):
        """Test scheduling with empty PG list."""
        result = topology_aware_schedule([], "STRICT_PACK", "test_pg", None)
        assert result == []

    def test_schedule_fallback_on_error(self, init_ray):
        """Test that scheduling falls back gracefully when tr.schedule() raises exception."""
        mock_pg = Mock()
        mock_pg.bundle_specs = [{"CPU": 1}]

        mock_tree = Mock()
        mock_tree.schedule.side_effect = Exception("Test error")

        with patch(
            "verl.single_controller.ray.topology_aware.build_tree_topo",
            return_value=mock_tree,
        ):
            result = topology_aware_schedule([mock_pg], "STRICT_PACK", "test_pg", None)
            assert len(result) == 1
            assert result == [mock_pg]

    def test_schedule_returns_original_pgs_on_tr_none(self, init_ray):
        """Test that scheduling returns original PGs when build_tree_topo returns None."""
        mock_pg = Mock()
        mock_pg.bundle_specs = [{"CPU": 1}]

        with patch(
            "verl.single_controller.ray.topology_aware.build_tree_topo",
            return_value=None,
        ):
            result = topology_aware_schedule([mock_pg], "STRICT_PACK", "test_pg", None)
            assert len(result) == 1
            assert result == [mock_pg]

    def test_schedule_success(self, init_ray):
        """Test successful topology-aware scheduling."""
        mock_pg = Mock()
        mock_pg.bundle_specs = [{"CPU": 1}]

        mock_tree = Mock()
        mock_tree.schedule.return_value = ["node1"]

        with patch(
            "verl.single_controller.ray.topology_aware.build_tree_topo",
            return_value=mock_tree,
        ):
            result = topology_aware_schedule([mock_pg], "STRICT_PACK", "test_pg", None)
            assert len(result) == 1
            assert "node1" in mock_pg.bundle_specs[0]
            assert mock_pg.bundle_specs[0]["node1"] == 0.01

    def test_schedule_multiple_pgs(self, init_ray):
        """Test scheduling with multiple placement groups."""
        mock_pg1 = Mock()
        mock_pg1.bundle_specs = [{"CPU": 1, "NPU": 1}]

        mock_pg2 = Mock()
        mock_pg2.bundle_specs = [{"CPU": 2}]

        mock_tree = Mock()
        mock_tree.schedule.return_value = ["node1", "node2"]

        with patch(
            "verl.single_controller.ray.topology_aware.build_tree_topo",
            return_value=mock_tree,
        ):
            result = topology_aware_schedule([mock_pg1, mock_pg2], "STRICT_PACK", "test_pg", None)
            assert len(result) == 2
            assert "node1" in mock_pg1.bundle_specs[0]
            assert "node2" in mock_pg2.bundle_specs[0]


class TestLogging:
    """Test logging functionality."""

    def test_exception_logging(self, init_ray, caplog):
        """Test that exceptions are properly logged with logger.exception."""
        mock_pg = Mock()
        mock_pg.bundle_specs = [{"CPU": 1}]

        mock_tree = Mock()
        mock_tree.schedule.side_effect = Exception("Test exception")

        with patch(
            "verl.single_controller.ray.topology_aware.build_tree_topo",
            return_value=mock_tree,
        ):
            with caplog.at_level("ERROR"):
                result = topology_aware_schedule([mock_pg], "STRICT_PACK", "test_pg", None)
                assert len(result) == 1
                assert "Topology-aware scheduling failed" in caplog.text
                assert "Test exception" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
