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
from abc import ABC, abstractmethod


class StrategyInterface(ABC):
    def __init__(self, node):
        self.node = node
        self.total_nodes = 0

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def preorder(self, num):
        pass


class StrategyFactory:
    @staticmethod
    def create(node):
        if node.block_size > 1:
            return BlockStrategy(node, node.block_size)
        else:
            return CommonStrategy(node)


class CommonStrategy(StrategyInterface):
    def __init__(self, node):
        super().__init__(node)
        self.name = "CommonStrategy"

    def update(self):
        if self.node.is_leaf():
            self.total_nodes = 1
            return
        self.total_nodes = 0
        for child in self.node.children:
            self.total_nodes += child.strategy.total_nodes

    def preorder(self, num):
        return self.total_nodes


class BlockStrategy(StrategyInterface):
    def __init__(self, node, block_size):
        super().__init__(node)
        self.name = "BlockStrategy"
        self.block_size = block_size

    def update(self):
        if self.node.is_leaf():
            self.total_nodes = 1
            return
        self.total_nodes = 0
        for child in self.node.children:
            self.total_nodes += child.strategy.total_nodes // self.block_size * self.block_size
        self.node.children.sort(key=lambda x: x.strategy.total_nodes, reverse=False)

    def preorder(self, num):
        if self.node.is_leaf():
            return num
        child_array = sorted(self.node.children, key=lambda x: x.strategy.total_nodes, reverse=False)
        total = num
        for child in child_array:
            size = child.strategy.total_nodes // self.block_size * self.block_size
            total -= size
            if total < 0:
                break
        if total > 0:
            raise ValueError("Order failed!")

        return num
