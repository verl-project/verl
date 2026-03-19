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

    def test_abstract_methods(self):
        """Test that StrategyInterface cannot be instantiated directly"""
        with pytest.raises(TypeError):
            StrategyInterface(MockNode())


class TestStrategyFactory:
    """Test cases for StrategyFactory"""

    def test_create_common_strategy(self):
        """Test factory creates CommonStrategy when block_size <= 1"""
        node = MockNode(block_size=0)
        strategy = StrategyFactory.create(node)
        assert isinstance(strategy, CommonStrategy)
        assert strategy.name == "CommonStrategy"

    def test_create_block_strategy(self):
        """Test factory creates BlockStrategy when block_size > 1"""
        node = MockNode(block_size=4)
        strategy = StrategyFactory.create(node)
        assert isinstance(strategy, BlockStrategy)
        assert strategy.name == "BlockStrategy"
        assert strategy.block_size == 4

    def test_create_block_strategy_with_different_sizes(self):
        """Test BlockStrategy creation with various block sizes"""
        for block_size in [2, 8, 16, 32]:
            node = MockNode(block_size=block_size)
            strategy = StrategyFactory.create(node)
            assert isinstance(strategy, BlockStrategy)
            assert strategy.block_size == block_size


class TestCommonStrategy:
    """Test cases for CommonStrategy"""

    def test_init(self):
        """Test CommonStrategy initialization"""
        node = MockNode()
        strategy = CommonStrategy(node)
        assert strategy.node == node
        assert strategy.total_nodes == 0
        assert strategy.name == "CommonStrategy"

    def test_update_leaf_node(self):
        """Test update method with leaf node"""
        node = MockNode()
        strategy = CommonStrategy(node)
        node.strategy = strategy

        strategy.update()
        assert strategy.total_nodes == 1

    def test_update_with_children(self):
        """Test update method with child nodes"""
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

    def test_preorder(self):
        """Test preorder method returns total_nodes"""
        node = MockNode()
        strategy = CommonStrategy(node)
        strategy.total_nodes = 10

        result = strategy.preorder(5)
        assert result == 10


class TestBlockStrategy:
    """Test cases for BlockStrategy"""

    def test_init(self):
        """Test BlockStrategy initialization"""
        node = MockNode()
        strategy = BlockStrategy(node, block_size=4)
        assert strategy.node == node
        assert strategy.block_size == 4
        assert strategy.name == "BlockStrategy"
        assert strategy.total_nodes == 0

    def test_update_leaf_node(self):
        """Test update method with leaf node"""
        node = MockNode()
        strategy = BlockStrategy(node, block_size=4)
        node.strategy = strategy

        strategy.update()
        assert strategy.total_nodes == 1

    def test_update_with_children(self):
        """Test update method aligns children to block size"""
        parent = MockNode(label="parent")
        child1 = MockNode(label="child1")
        child2 = MockNode(label="child2")
        child3 = MockNode(label="child3")

        parent.children = [child1, child2, child3]

        child_strategies = []
        for i, child in enumerate([child1, child2, child3]):
            strat = CommonStrategy(child)
            strat.total_nodes = [7, 10, 15][i]
            child.strategy = strat
            child_strategies.append(strat)

        parent_strategy = BlockStrategy(parent, block_size=4)
        parent.strategy = parent_strategy

        parent_strategy.update()
        # Expected: 7//4*4 + 10//4*4 + 15//4*4 = 4 + 8 + 12 = 24
        assert parent_strategy.total_nodes == 24

    def test_update_sorts_children(self):
        """Test update method sorts children by total_nodes"""
        parent = MockNode(label="parent")
        child1 = MockNode(label="child1")
        child2 = MockNode(label="child2")
        child3 = MockNode(label="child3")

        parent.children = [child1, child2, child3]

        for i, child in enumerate([child1, child2, child3]):
            strat = CommonStrategy(child)
            strat.total_nodes = [15, 7, 10][i]
            child.strategy = strat

        parent_strategy = BlockStrategy(parent, block_size=4)
        parent.strategy = parent_strategy

        parent_strategy.update()
        # Children should be sorted in ascending order after update
        assert parent.children[0].strategy.total_nodes == 7
        assert parent.children[1].strategy.total_nodes == 10
        assert parent.children[2].strategy.total_nodes == 15

    def test_preorder_leaf_node(self):
        """Test preorder with leaf node returns num"""
        node = MockNode()
        strategy = BlockStrategy(node, block_size=4)
        node.strategy = strategy

        result = strategy.preorder(5)
        assert result == 5

    def test_preorder_success(self):
        """Test successful preorder traversal"""
        parent = MockNode(label="parent")
        child1 = MockNode(label="child1")
        child2 = MockNode(label="child2")

        parent.children = [child1, child2]

        for i, child in enumerate([child1, child2]):
            strat = CommonStrategy(child)
            strat.total_nodes = [4, 8][i]
            child.strategy = strat

        parent_strategy = BlockStrategy(parent, block_size=4)
        parent.strategy = parent_strategy
        parent_strategy.update()

        result = parent_strategy.preorder(12)
        assert result == 12

    def test_preorder_failure(self):
        """Test preorder raises error when allocation fails"""
        parent = MockNode(label="parent")
        child1 = MockNode(label="child1")

        parent.children = [child1]

        child_strategy = CommonStrategy(child1)
        child_strategy.total_nodes = 4
        child1.strategy = child_strategy

        parent_strategy = BlockStrategy(parent, block_size=4)
        parent.strategy = parent_strategy
        parent_strategy.update()

        # Request more than available should raise error
        with pytest.raises(ValueError, match="Order failed"):
            parent_strategy.preorder(10)


class TestStrategyIntegration:
    """Integration tests for strategy module"""

    def test_factory_and_execution_common(self):
        """Test factory creation and execution of CommonStrategy"""
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

    def test_factory_and_execution_block(self):
        """Test factory creation and execution of BlockStrategy"""
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

        assert root_strategy.total_nodes == 2
