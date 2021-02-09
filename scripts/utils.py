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
