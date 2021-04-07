import math
from copy import deepcopy


class DSU:
    def __init__(self, size: int):
        self.size = size
        self.parent = [i for i in range(size)]
        self.rank = [1 for _ in range(size)]

    def find(self, v: int) -> int:
        if self.parent[v] != v:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, v1: int, v2: int) -> int:
        """
        Unions set of v1 and set of v2.
        Returns:
            (int): New leader of union.
        """
        v1 = self.find(v1)
        v2 = self.find(v2)
        if v1 != v2:
            if self.rank[v1] < self.rank[v2]:
                v1, v2 = v2, v1
            self.parent[v2] = v1
            if self.rank[v1] == self.rank[v2]:
                self.rank[v1] += 1
        return v1

    def connected(self, v1: int, v2: int) -> bool:
        """
        Checks that v1 and v2 are from the same set.
        """
        return self.find(v1) == self.find(v2)


class ScanLineElement:
    def __init__(self):
        self.starts = []
        self.ends = []
        self.weight = 0

    def add(self, start: int, end: int, weight: int):
        self.starts.append(start)
        self.ends.append(end)
        self.weight += weight


class Element:
    def __init__(self):
        self.left = 0
        self.right = 0
        self.value = 0
        self.delta = 0


class SegmentTree:
    """
    A class with segment tree implementation to work with coverages properly.
    """

    def __init__(self, size: int):
        self.size = size
        self.find_tree_size()
        self.tree = [Element() for _ in range(self.size)]
        self.build_tree()

    def build_tree(self):
        for i in range(self.size // 2, self.size):
            self.tree[i].left = i - self.size // 2
            self.tree[i].right = i - self.size // 2

        for i in range(self.size // 2 - 1, -1, -1):
            self.tree[i].left = self.tree[2 * i].left
            self.tree[i].right = self.tree[2 * i + 1].right

    def find_tree_size(self):
        self.size = 1 << (math.ceil(math.log2(self.size)) + 1)

    def push(self, node: int):
        self.tree[2 * node].delta += self.tree[node].delta
        self.tree[2 * node + 1].delta += self.tree[node].delta
        self.tree[node].delta = 0

    def add(self, start: int, end: int, weight: int):
        self.update(0, start, end, weight)

    def delete(self, start: int, end: int, weight: int):
        self.update(0, start, end, -weight)

    def update(self, node, left, right, weight):
        l = self.tree[node].left
        r = self.tree[node].right
        if left > r or right < l:
            return
        elif left < l and r < right:
            self.tree[node].delta = weight
            return
        self.push(node)
        self.update(2 * node + 1, left, right, weight)
        self.update(2 * node + 2, left, right, weight)
        self.tree[node].value = self.tree[2 * node].value + self.tree[2 * node].delta \
                                + self.tree[2 * node + 1].value + self.tree[2 * node + 1].delta

    def push_everything(self):
        for i in range(self.size // 2):
            if self.tree[i].delta != 0:
                self.tree[2 * i].delta += self.tree[i].delta
                self.tree[2 * i + 1].delta += self.tree[i].delta
                self.tree[i].value = self.tree[2 * i].value + self.tree[2 * i].delta \
                                     + self.tree[2 * i + 1].value + self.tree[2 * i + 1].delta
                self.tree[i].delta = 0

    def trees_union(self, tree2):
        self.push_everything()
        tree2.push_everything()
        new_tree = deepcopy(self)
        for i in range(self.size):
            new_tree.tree[i] += tree2.tree[i]
        return new_tree
