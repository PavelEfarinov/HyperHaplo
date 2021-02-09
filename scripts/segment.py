from __future__ import annotations
from typing import Tuple, List


class Segment:
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end

    def __lt__(self, other: Segment) -> bool:
        return self.end < other.begin

    def __gt__(self, other: Segment) -> bool:
        return self.begin > other.end

    def __eq__(self, other: Segment) -> bool:
        return self.begin == other.begin and self.end == other.end

    def __contains__(self, other: Segment) -> bool:
        return self.begin <= other.begin and other.end <= self.end

    def has_intersection(self, other: Segment) -> bool:
        l = max(self.begin, other.begin)
        r = min(self.end, other.end)
        return l <= r

    def __repr__(self):
        return f'Segment(L={self.begin},R={self.end})'


class SegmentList:
    def __init__(self, *segments):
        assert all(map(len, segments)), 'All segments must be not empty'
        self.segments = segments
        self.active_idx = 0

    def active_begin(self) -> int:
        return self.segments[self.active_idx][0]

    def active_end(self) -> int:
        return self.segments[self.active_idx][-1]

    def active_len(self) -> int:
        return len(self.segments[self.active_idx])

    def next(self) -> bool:
        self.active_idx += 1
        return self.active_idx < len(self.segments)

    def has_more(self) -> bool:
        return self.active_idx < len(self.segments)

    @staticmethod
    def intersection(
            sl1: SegmentList,
            sl2: SegmentList
    ) -> List[LocalSegmentIntersection]:
        """ Fast intersection of 2 segment lists.
        Speed O(SEG_NUM(sl1) + SEG_NUM(sl2)), where SEG_NUM(sl)
        is number of segments in the segment list `sl`.

        Args:
            sl1: First list of ordered segments.
            sl2: Second list of ordered segments.

        Returns:
            List[LocalSegmentIntersection]: List of intersection items.
        """
        result_intersections: List[LocalSegmentIntersection] = []

        while sl1.has_more() and sl2.has_more():
            l1, r1 = sl1.active_begin(), sl1.active_end()
            l2, r2 = sl2.active_begin(), sl2.active_end()
            l, r = max(l1, l2), min(r1, r2)

            if l <= r:
                l1_idx = l - l1 if l1 != l else 0
                l2_idx = l - l2 if l2 != l else 0
                r1_idx = sl1.active_len() - (r1 - r) - 1
                r2_idx = sl2.active_len() - (r2 - r) - 1

                result_intersections.append(LocalSegmentIntersection(
                    first_segment_idx=sl1.active_idx,
                    first_positions_boundaries=(l1_idx, r1_idx),
                    second_segment_idx=sl2.active_idx,
                    second_positions_boundaries=(l2_idx, r2_idx)
                ))

            if r1 <= r2:
                sl1.next()
            else:
                sl2.next()
        return result_intersections


class LocalSegmentIntersection:
    def __init__(
            self,
            first_segment_idx: int,
            first_positions_boundaries: Tuple[int, int],
            second_segment_idx: int,
            second_positions_boundaries: Tuple[int, int]
    ):
        """

        Args:
            first_segment_idx:
            first_positions_boundaries: (inclusive)
            second_segment_idx:
            second_positions_boundaries: (inclusive)
        """
        self.first_segment_idx = first_segment_idx
        self.first_positions_boundaries = first_positions_boundaries
        self.second_segment_idx = second_segment_idx
        self.second_positions_boundaries = second_positions_boundaries

    def __len__(self) -> int:
        l, r = self.first_positions_boundaries
        return r - l + 1

    def __repr__(self) -> str:
        l1, r1 = self.first_positions_boundaries
        l2, r2 = self.second_positions_boundaries
        return f'LocalSegmentIntersection(' \
               f'First[{self.first_segment_idx}](Li={l1},Ri={r1}), ' \
               f'Second[{self.second_segment_idx}](Li={l2},Ri={r2}))'


if __name__ == '__main__':
    # TODO Tests coverage

    # ==== Segment ====
    s1 = Segment(1, 4)
    s2 = Segment(2, 3)
    s3 = Segment(5, 6)

    print(s1 < s2, s1 < s3)
    print(s1 in s2, s2 in s1)
    print(s1.has_intersection(s2), s1.has_intersection(s3))

    # ==== SegmentList ====
    list1 = [[1, 2], [5, 6, 7], [10]]
    # list2 = [[2], [3, 4, 5, 6]]
    list2 = [[2], [3, 4, 5, 6, 7, 8, 9, 10, 11]]

    sl1 = SegmentList(*list1)
    sl2 = SegmentList(*list2)
    inters = SegmentList.intersection(sl1, sl2)

    for local_inter in inters:
        i = local_inter.first_segment_idx
        l, r = local_inter.first_positions_boundaries
        print(f'{i}[{l},{r}]: {list1[i][l:r + 1]}')
