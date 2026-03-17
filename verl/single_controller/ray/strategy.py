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
            raise Exception("Order failed!")

        return num

