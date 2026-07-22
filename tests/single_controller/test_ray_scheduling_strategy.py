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

"""
Unit tests for verl.single_controller.ray.strategy
"""

import pytest

from verl.single_controller.ray.strategy import (
    BlockStrategy,
    CommonStrategy,
    StrategyFactory,
    StrategyInterface,
)


@pytest.fixture
def mock_node():
    """Fixture to create a basic mock node."""
    return MockNode()


class MockNode:
    """Mock node class for testing"""

    def __init__(self, label="mock", parent=None, layer=0, block_size=0):
        self.label = label
        self.layer = layer
        self.children = []
        self.parent = parent
        self.strategy = None
        self.block_size = block_size

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def set_strategy(self, st):
        self.strategy = st


class TestStrategyInterface:
    """Test cases for StrategyInterface abstract base class"""

    def test_cannot_instantiate_abstract_class(self):
        """Verify StrategyInterface cannot be instantiated directly due to abstract methods"""
        with pytest.raises(TypeError):
            StrategyInterface(MockNode())


class TestStrategyFactory:
    """Test cases for StrategyFactory"""

    @pytest.mark.parametrize(
        "block_size,expected_type,expected_name",
        [
            (0, CommonStrategy, "CommonStrategy"),
            (1, CommonStrategy, "CommonStrategy"),
            (2, BlockStrategy, "BlockStrategy"),
            (4, BlockStrategy, "BlockStrategy"),
            (8, BlockStrategy, "BlockStrategy"),
            (16, BlockStrategy, "BlockStrategy"),
            (32, BlockStrategy, "BlockStrategy"),
        ],
    )
    def test_create_strategy_various_block_sizes(self, block_size, expected_type, expected_name):
        """Test factory creates correct strategy type based on block_size"""
        node = MockNode(block_size=block_size)
        strategy = StrategyFactory.create(node)

        assert isinstance(strategy, expected_type)
        assert strategy.name == expected_name

        if hasattr(strategy, "block_size"):
            assert strategy.block_size == block_size

    def test_create_block_strategy_preserves_block_size(self):
        """Test that BlockStrategy correctly stores the block_size parameter"""
        block_size = 4
        node = MockNode(block_size=block_size)
        strategy = StrategyFactory.create(node)

        assert isinstance(strategy, BlockStrategy)
        assert strategy.block_size == block_size


class TestCommonStrategy:
    """Test cases for CommonStrategy"""

    def test_initialization(self, mock_node):
        """Test CommonStrategy initializes with correct default values"""
        strategy = CommonStrategy(mock_node)

        assert strategy.node is mock_node
        assert strategy.total_nodes == 0
        assert strategy.name == "CommonStrategy"

    def test_update_leaf_node_sets_total_nodes_to_one(self, mock_node):
        """Verify update() sets total_nodes to 1 for leaf nodes"""
        strategy = CommonStrategy(mock_node)
        mock_node.strategy = strategy

        strategy.update()

        assert strategy.total_nodes == 1

    def test_update_aggregates_children_total_nodes(self):
        """Verify update() sums up children's total_nodes"""
        parent = MockNode(label="parent")
        child1 = MockNode(label="child1")
        child2 = MockNode(label="child2")
        parent.children = [child1, child2]

        child1_strategy = CommonStrategy(child1)
        child2_strategy = CommonStrategy(child2)
        child1.strategy = child1_strategy
        child2.strategy = child2_strategy

        child1_strategy.total_nodes = 3
        child2_strategy.total_nodes = 5

        parent_strategy = CommonStrategy(parent)
        parent.strategy = parent_strategy
        parent_strategy.update()

        assert parent_strategy.total_nodes == 8

    def test_update_with_no_children(self, mock_node):
        """Test update with node that has empty children list"""
        strategy = CommonStrategy(mock_node)
        mock_node.strategy = strategy
        mock_node.children = []

        strategy.update()

        assert strategy.total_nodes == 1

    def test_update_with_many_children(self):
        """Test update aggregates many children correctly"""
        parent = MockNode(label="parent")
        num_children = 10

        for i in range(num_children):
            child = MockNode(label=f"child{i}")
            child_strategy = CommonStrategy(child)
            child_strategy.total_nodes = i + 1
            child.strategy = child_strategy
            parent.children.append(child)

        parent_strategy = CommonStrategy(parent)
        parent.strategy = parent_strategy
        parent_strategy.update()

        expected_total = sum(range(1, num_children + 1))
        assert parent_strategy.total_nodes == expected_total

    @pytest.mark.parametrize(
        "input_num,total_nodes,expected",
        [
            (5, 10, 10),
            (0, 10, 10),
            (100, 50, 50),
        ],
    )
    def test_preorder_returns_total_nodes(self, input_num, total_nodes, expected):
        """Verify preorder() always returns total_nodes regardless of input"""
        node = MockNode()
        strategy = CommonStrategy(node)
        strategy.total_nodes = total_nodes

        result = strategy.preorder(input_num)

        assert result == expected


class TestBlockStrategy:
    """Test cases for BlockStrategy"""

    def test_initialization(self, mock_node):
        """Test BlockStrategy initializes with correct values"""
        block_size = 4
        strategy = BlockStrategy(mock_node, block_size=block_size)

        assert strategy.node is mock_node
        assert strategy.block_size == block_size
        assert strategy.name == "BlockStrategy"
        assert strategy.total_nodes == 0

    def test_update_leaf_node_sets_total_nodes_to_one(self, mock_node):
        """Verify update() sets total_nodes to 1 for leaf nodes"""
        strategy = BlockStrategy(mock_node, block_size=4)
        mock_node.strategy = strategy

        strategy.update()

        assert strategy.total_nodes == 1

    @pytest.mark.parametrize(
        "child_totals,block_size,expected_total",
        [
            ([7, 10, 15], 4, 24),
            ([5, 5, 5], 2, 12),
            ([3, 7, 11], 5, 15),
            ([4, 8, 12], 4, 24),
        ],
    )
    def test_update_aligns_children_to_block_size(self, child_totals, block_size, expected_total):
        """Verify update() aligns each child's contribution to block_size boundary"""
        parent = MockNode(label="parent")

        for total in child_totals:
            child = MockNode(label=f"child_{total}")
            strat = CommonStrategy(child)
            strat.total_nodes = total
            child.strategy = strat
            parent.children.append(child)

        parent_strategy = BlockStrategy(parent, block_size=block_size)
        parent.strategy = parent_strategy
        parent_strategy.update()

        assert parent_strategy.total_nodes == expected_total

    def test_update_sorts_children_ascending(self):
        """Verify update() sorts children by total_nodes in ascending order"""
        parent = MockNode(label="parent")
        initial_totals = [15, 7, 10, 3]

        for total in initial_totals:
            child = MockNode(label=f"child_{total}")
            strat = CommonStrategy(child)
            strat.total_nodes = total
            child.strategy = strat
            parent.children.append(child)

        parent_strategy = BlockStrategy(parent, block_size=4)
        parent.strategy = parent_strategy
        parent_strategy.update()

        sorted_totals = [child.strategy.total_nodes for child in parent.children]
        assert sorted_totals == sorted(initial_totals)

    def test_preorder_leaf_node_returns_input(self, mock_node):
        """Verify preorder() on leaf node returns the input num"""
        strategy = BlockStrategy(mock_node, block_size=4)
        mock_node.strategy = strategy

        test_values = [0, 1, 5, 10, 100]
        for num in test_values:
            result = strategy.preorder(num)
            assert result == num

    def test_preorder_successful_allocation(self):
        """Test preorder successfully allocates requested nodes"""
        parent = MockNode(label="parent")
        child_totals = [4, 8]

        for total in child_totals:
            child = MockNode(label=f"child_{total}")
            strat = CommonStrategy(child)
            strat.total_nodes = total
            child.strategy = strat
            parent.children.append(child)

        parent_strategy = BlockStrategy(parent, block_size=4)
        parent.strategy = parent_strategy
        parent_strategy.update()

        result = parent_strategy.preorder(12)
        assert result == 12

    def test_preorder_insufficient_capacity_raises_error(self):
        """Verify preorder raises ValueError when requesting more than available"""
        parent = MockNode(label="parent")
        child = MockNode(label="child")
        child_strategy = CommonStrategy(child)
        child_strategy.total_nodes = 4
        child.strategy = child_strategy
        parent.children = [child]

        parent_strategy = BlockStrategy(parent, block_size=4)
        parent.strategy = parent_strategy
        parent_strategy.update()

        with pytest.raises(ValueError, match="Order failed"):
            parent_strategy.preorder(10)

    @pytest.mark.parametrize(
        "child_totals,block_size,request_num,should_fail",
        [
            ([4, 8], 4, 12, False),
            ([4, 8], 4, 8, False),
            ([4, 8], 4, 7, False),
            ([4, 8], 4, 13, True),
            ([4, 8], 4, 20, True),
            ([5, 10], 5, 15, False),
            ([5, 10], 5, 16, True),
        ],
    )
    def test_preorder_allocation_scenarios(self, child_totals, block_size, request_num, should_fail):
        """Test various preorder allocation scenarios"""
        parent = MockNode(label="parent")

        for total in child_totals:
            child = MockNode(label=f"child_{total}")
            strat = CommonStrategy(child)
            strat.total_nodes = total
            child.strategy = strat
            parent.children.append(child)

        parent_strategy = BlockStrategy(parent, block_size=block_size)
        parent.strategy = parent_strategy
        parent_strategy.update()

        if should_fail:
            with pytest.raises(ValueError, match="Order failed"):
                parent_strategy.preorder(request_num)
        else:
            result = parent_strategy.preorder(request_num)
            assert result == request_num


class TestStrategyIntegration:
    """Integration tests for strategy module"""

    def test_single_leaf_common_strategy(self):
        """Test complete workflow with single leaf using CommonStrategy"""
        root = MockNode(label="root")
        leaf = MockNode(label="leaf")
        root.children = [leaf]

        leaf_strategy = StrategyFactory.create(leaf)
        leaf.strategy = leaf_strategy
        leaf_strategy.update()

        root_strategy = StrategyFactory.create(root)
        root.strategy = root_strategy
        root_strategy.update()

        assert root_strategy.total_nodes == 1

    def test_multiple_leaves_common_strategy(self):
        """Test workflow with multiple leaves using CommonStrategy"""
        root = MockNode(label="root")
        num_leaves = 5

        for i in range(num_leaves):
            leaf = MockNode(label=f"leaf{i}")
            leaf_strategy = StrategyFactory.create(leaf)
            leaf.strategy = leaf_strategy
            leaf_strategy.update()
            root.children.append(leaf)

        root_strategy = StrategyFactory.create(root)
        root.strategy = root_strategy
        root_strategy.update()

        assert root_strategy.total_nodes == num_leaves

    def test_block_strategy_with_two_leaves(self):
        """Test workflow with BlockStrategy managing two leaf nodes"""
        root = MockNode(label="root", block_size=2)
        leaf1 = MockNode(label="leaf1")
        leaf2 = MockNode(label="leaf2")
        root.children = [leaf1, leaf2]

        for leaf in [leaf1, leaf2]:
            leaf_strategy = StrategyFactory.create(leaf)
            leaf.strategy = leaf_strategy
            leaf_strategy.update()

        root_strategy = StrategyFactory.create(root)
        root.strategy = root_strategy
        root_strategy.update()

        assert root_strategy.total_nodes == 0

    def test_block_strategy_alignment_effect(self):
        """Test that BlockStrategy properly aligns to block boundaries"""
        root = MockNode(label="root", block_size=4)

        leaf_totals = [3, 5, 7]
        for i, total in enumerate(leaf_totals):
            intermediate = MockNode(label=f"intermediate_{i}")

            for j in range(total):
                leaf = MockNode(label=f"leaf_{i}_{j}")
                leaf_strategy = StrategyFactory.create(leaf)
                leaf.strategy = leaf_strategy
                leaf_strategy.update()
                intermediate.children.append(leaf)

            int_strategy = StrategyFactory.create(intermediate)
            intermediate.strategy = int_strategy
            int_strategy.update()
            root.children.append(intermediate)

        root_strategy = StrategyFactory.create(root)
        root.strategy = root_strategy
        root_strategy.update()

        expected = sum(t // 4 * 4 for t in leaf_totals)
        assert root_strategy.total_nodes == expected

    def test_deep_tree_hierarchy_common_strategy(self):
        """Test strategy propagation in a deep tree hierarchy"""
        depth = 4
        nodes_per_level = 2

        def create_subtree(depth, current_depth=0):
            node = MockNode(label=f"node_{current_depth}")
            if current_depth < depth - 1:
                for _ in range(nodes_per_level):
                    child = create_subtree(depth, current_depth + 1)
                    node.children.append(child)
            return node

        root = create_subtree(depth)

        def apply_strategies(node):
            strategy = StrategyFactory.create(node)
            node.strategy = strategy
            for child in node.children:
                apply_strategies(child)
            if node.children:
                for child in node.children:
                    child.strategy.update()
                strategy.update()

        apply_strategies(root)

        expected_leaves = nodes_per_level ** (depth - 1)
        assert root.strategy.total_nodes == expected_leaves


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
