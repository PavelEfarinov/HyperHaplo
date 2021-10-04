from typing import List

from scripts.utils import ScanLineElement


class Coverage:
    def __init__(self, coverage: ScanLineElement = None, cov_weight=0):
        self.coverage_weight = cov_weight
        if coverage is not None:
            self.coverage_map = coverage
        else:
            self.coverage_map = ScanLineElement()

    def add(self, start_position: List[int], end_position: List[int], value: int = 1):
        """ Add coverage on the interval.
        Args:
            start_position: Start position of the new coverage interval (inclusive).
            end_position: End position of the new coverage interval (inclusive).
            value: Delta value of the new coverage interval.
        """
        self.coverage_weight += value
        for start, end in zip(start_position, end_position):
            self.coverage_map.add(start, end, 1)