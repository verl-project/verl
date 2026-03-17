from strategy import CommonStrategy

class Node:
    def __init__(self, label, parent, layer):
        self.label = label
        self.layer = layer
        self.children = []
        self.parent = parent
        self.strategy = None
        self.block_size = 0

    def set_block_size(self, block_size):
        self.block_size = block_size

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def set_strategy(self, st):
        self.strategy = st

    def update(self):
        self.strategy.update()

    def set_strategy_by_layer(self):
        self.set_strategy(StrategyFactory.create(self))

    def add_child(self, node):
        self.children.append(node)

class TreeTopology:
    def __init__(self, height):
        self.root = Node("root", None, height)
        self.node_map = {"root": self.root}

    def get_node_by_label(self, label):
        return self.node_map.get(label)

    def insert_node(self, labels, block_sizes):
        self.add_child(self.root, 0, labels, block_sizes)

    def add_child(self, cur, layer, labels, block_sizes):
        if layer == len(labels):
            return

        for child in cur.children:
            if child.label == labels[layer]:
                self.add_child(child, layer + 1, labels, block_sizes)
                return

        now = Node(labels[layer], cur, layer)
        if block_sizes[layer] > 0:
            now.set_block_size(block_sizes[layer])
        cur.add_child(now)

        self.add_child(now, layer + 1, labels, block_sizes)

    def update_topology(self, cur):
        if cur.is_leaf():
            cur.update()
            return
        for child in cur.children:
            self.update_topology(child)

    def set_strategy(self, cur):
        cur.set_strategy_by_layer()
        if cur.is_leaf():
            return
        for child in cur.children:
            self.set_strategy(child)


    def get_node(self, cur, depth, labels):
        if depth >= len(labels):
            return cur
        for child in cur.children:
            if child.label == labels[depth]:
                return self.get_node(child, depth + 1, labels)
        return None

    def preorder(self, cur, num, order_array):
        if num > cur.strategy.preorder(num):
            raise ValueError("number overflow")
        if cur.is_leaf():
            order_array.append(cur.label)
            return
        for child in cur.children:
            now = child.strategy.preorder(num)
            if isinstance(cur.strategy, SpBlockStrategy):
                now = now // cur.strategy.block_size * cur.strategy.block_size
            now = min(now, num)
            num -= now
            if now > 0:
                self.preorder(child, now, order_array)
            if num <= 0:
                break

    def schedule(self, num):
        result = []
        self.preorder(self.root, num, result)
        return result

