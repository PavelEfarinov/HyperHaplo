from __future__ import annotations

import math
from bisect import bisect_left
from typing import List, Optional, Dict, Tuple
from collections import Counter, defaultdict
from more_itertools import pairwise

from scripts.segment import SegmentList, LocalSegmentIntersection
from scripts.utils import SegmentTree, ScanLineElement


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

    @staticmethod
    def union(cov1: Coverage, cov2: Coverage) -> Coverage:
        cov = Coverage()
        min_index = min([cov1.coverage_map.starts[0], cov2.coverage_map.starts[0]])
        max_index = max([cov1.coverage_map.starts[0], cov2.coverage_map.starts[0]])
        cov.add([min_index], [max_index], 1)
        return cov


class HEdge:
    def __init__(
            self,
            snp2genome: Dict[int, int],
            genome2snp: Dict[int, int],
    ):
        self.snp2genome = snp2genome
        self.genome2snp = genome2snp
        self.weight = 0
        self.count_snp = 0
        self.coverage = Coverage()
        self.frequency = 1
        self.start_pos = -1
        self.used = False
        self.snp_to_nucl = {}

    class Overlap:
        def __init__(
                self,
                fst_positions_count: int,
                snd_positions_count: int,
                overlap_size: int,
                match_size: int
        ):
            self.fst_positions_count = fst_positions_count
            self.snd_positions_count = snd_positions_count
            self.overlap_size = overlap_size
            self.match_size = match_size

            union = (self.fst_positions_count + self.snd_positions_count
                     - self.overlap_size)
            self.positional_jaccard = self.overlap_size / union
            if overlap_size > 0:
                self.overlap_accuracy = self.match_size / self.overlap_size
            else:
                self.overlap_accuracy = None
            self.is_consistent = self.overlap_size == self.match_size

        @staticmethod
        def metrics(
                he1: HEdge,
                he2: HEdge,
                intersection: List[LocalSegmentIntersection]
        ) -> HEdge.Overlap:
            """Evaluate all objective metrics on linkage of 2 hedges."""
            size = 0
            match_size = 0

            nucls1 = he1.get_nucl_segments()
            nucls2 = he2.get_nucl_segments()

            for inter_item in intersection:
                l1, r1 = inter_item.first_positions_boundaries
                fst_idx = inter_item.first_segment_idx
                l2, r2 = inter_item.second_positions_boundaries
                snd_idx = inter_item.second_segment_idx

                size += r1 - l1 + 1
                for i, j in zip(range(l1, r1 + 1), range(l2, r2 + 1)):
                    if nucls1[fst_idx][i] == nucls2[snd_idx][j]:
                        match_size += 1

            return HEdge.Overlap(
                fst_positions_count=sum(map(len, nucls1)),
                snd_positions_count=sum(map(len, nucls2)),
                overlap_size=size,
                match_size=match_size
            )

        def __repr__(self):
            return f'Overlap(\n' \
                   f'fst_positions_count={self.fst_positions_count},\n' \
                   f'snd_positions_count={self.snd_positions_count},\n' \
                   f'overlap_size={self.overlap_size},\n' \
                   f'match_size={self.match_size},\n' \
                   f'positional_jaccard={round(self.positional_jaccard, 3)},\n' \
                   f'overlap_accuracy={round(self.overlap_accuracy, 3)},\n' \
                   f'is_consistent={self.is_consistent})'

    @staticmethod
    def overlap_size(intersection: List[LocalSegmentIntersection]) -> int:
        """How many common positions h1 and h2 have.
        Time O(len(intersection)).
        """
        size = 0
        for inter_item in intersection:
            l, r = inter_item.first_positions_boundaries
            size += r - l + 1
        return size

    @staticmethod
    def is_allowed_merge_linkage(
            he1: HEdge,
            he2: HEdge,
            intersection: List[LocalSegmentIntersection]
    ) -> bool:
        # TODO перебрать варианты
        first_segments_indexes = set(inter.first_segment_idx for inter in intersection)
        second_segments_indexes = set(inter.second_segment_idx for inter in intersection)

        if isinstance(he1, PairedHEdge) and isinstance(he2, PairedHEdge):
            return len(first_segments_indexes) == len(second_segments_indexes) == 2
        elif isinstance(he1, SingleHEdge) and isinstance(he2, SingleHEdge):
            return len(intersection) > 0
        else:
            # case 1: single and paired => 1 + 2
            # case 2: single and paired => 2 + 1
            return (len(first_segments_indexes) + len(second_segments_indexes)) == 3

    def is_ambiguous_with(self, other: HEdge) -> bool:
        """
        Two hedges are ambiguous if they have distinct nucleotides in common positions
        and their leftmost and rightmost nucleotides are not the same.
        """
        raise NotImplementedError('Outdated functionality')

        left_pos = max(self.positions[0], other.positions[0])
        right_pos = min(self.positions[-1], other.positions[-1])
        if left_pos > right_pos:
            # Does not have intersection
            return False

        self_leftmost_nucl = self.nucls[bisect_left(self.positions, left_pos)]
        other_leftmost_nucl = other.nucls[bisect_left(other.positions, left_pos)]

        self_rightmost_nucl = self.nucls[bisect_left(self.positions, right_pos)]
        other_rightmost_nucl = other.nucls[bisect_left(other.positions, right_pos)]

        L = self_leftmost_nucl == other_leftmost_nucl
        R = self_rightmost_nucl == other_rightmost_nucl

        if not L and not R:
            return False

        return not self.is_consistent_with(other)

    def merge_with_(self, other: HEdge):
        """
        Inplace merging self with other hedge.
        """
        positions = []
        nucls = []
        i, j = 0, 0

        while i < len(self.positions) and j < len(other.positions):
            if self.positions[i] < other.positions[j]:
                positions.append(self.positions[i])
                nucls.append(self.nucls[i])
                i += 1
            elif self.positions[i] > other.positions[j]:
                positions.append(other.positions[j])
                nucls.append(other.nucls[j])
                j += 1
            else:
                positions.append(self.positions[i])
                nucls.append(self.nucls[i])
                i += 1
                j += 1

        positions += self.positions[i:]
        nucls += self.nucls[i:]

        positions += other.positions[j:]
        nucls += other.nucls[j:]

        # Reset attributes
        self.positions = positions
        self.nucls = nucls
        self.weight += other.weight
        # self.weight = 0  # TODO Now self.weight is undefined.
        for i, cover in enumerate(other.coverage):
            self.coverage[i] += cover

        # if 'avg_coverage' in self.__dict__:
        # Invalidate cache
        # del self.__dict__['avg_coverage']

    def merge_with(self, other: HEdge) -> HEdge:
        """
        Merge self and other hedges.
        """
        positions = []
        nucls = []
        i, j = 0, 0

        while i < len(self.positions) and j < len(other.positions):
            if self.positions[i] < other.positions[j]:
                positions.append(self.positions[i])
                nucls.append(self.nucls[i])
                i += 1
            elif self.positions[i] > other.positions[j]:
                positions.append(other.positions[j])
                nucls.append(other.nucls[j])
                j += 1
            else:
                positions.append(self.positions[i])
                nucls.append(self.nucls[i])
                i += 1
                j += 1

        positions += self.positions[i:]
        nucls += self.nucls[i:]

        positions += other.positions[j:]
        nucls += other.nucls[j:]

        new_coverage = [c1 + c2 for c1, c2 in zip(self.coverage, other.coverage)]
        return HEdge(
            positions=positions,
            nucls=nucls,
            weight=0,  # TODO Redefine weight
            size=self.weight + other.weight,
            coverage=new_coverage
        )

    def __hash__(self):
        if not hasattr(self, '_hash'):
            raise NotImplementedError('Must be defined in subclass')
        return self._hash

    @staticmethod
    def check_continuity_condition(
            list_of_positions: List[List[int]]
    ) -> Optional[str]:
        """ Necessary condition for successful hedge creation.
        Args:
            list_of_positions:
        Returns:
            (str): Reason of failure.
        """
        for i, positions in enumerate(list_of_positions):
            for cur, next in pairwise(positions):
                if not cur + 1 == next:
                    return f'Warning: must be subsequent [part {i}]: {cur} -> {next}'

        for cur_part, next_part in pairwise(list_of_positions):
            if not cur_part[-1] + 1 != next_part[0]:
                return f'Warning: parts must be merged'

        return None

    @staticmethod
    def build(
            positions: List[List[int]],
            nucls: List[List[str]],
            snp2genome: Dict[int, int],
            genome2snp: Dict[int, int],
            start_pos: List[int],
            reindex_snps: bool = True
    ) -> Optional[HEdge]:
        """ Builder method for HEdge subclasses.
        Args:
            positions: List of list of (SNP order) positions.
            nucls: List of list of nucleotides corresponding to positions argument.
            reindex_snps: If flat is set then `positions` are genome positions, else SNP indexes.
        Returns:
            HEdge: New HEdge instance.
        """
        assert len(positions) == len(nucls), f'{len(positions)} != {len(nucls)}'

        if reindex_snps:
            if len(positions) == 1:
                positions = [HEdge.reindex_to_snp_order(positions[0], genome2snp)]

            if len(positions) == 2:
                positions = [
                    HEdge.reindex_to_snp_order(positions[0], genome2snp),
                    HEdge.reindex_to_snp_order(positions[1], genome2snp)
                ]

        if len(positions) == 2:
            # merge parts if they are subsequent
            if positions[0][-1] + 1 == positions[1][0]:
                positions = [positions[0] + positions[1]]
                nucls = [nucls[0] + nucls[1]]

        fail_reason = HEdge.check_continuity_condition(positions)
        if fail_reason is not None:
            # TODO add logging level for warnings
            # print(fail_reason)
            return None

        if len(positions) == 1:
            return SingleHEdge(
                positions=positions[0],
                nucls=nucls[0],
                snp2genome=snp2genome,
                genome2snp=genome2snp,
                start_pos=start_pos[0],
            )
        elif len(positions) == 2:
            return PairedHEdge(
                left_positions=positions[0],
                left_nucls=nucls[0],
                right_positions=positions[1],
                right_nucls=nucls[1],
                snp2genome=snp2genome,
                genome2snp=genome2snp,
            )
        else:
            # TODO try MultiHEdge
            raise ValueError(f'Unsupported intervals count for HEdge: {len(positions)}')

    @staticmethod
    def intersect(he1: HEdge, he2: HEdge) -> List[LocalSegmentIntersection]:
        segments1 = he1.get_position_segments()
        segments2 = he2.get_position_segments()
        seg_list1 = SegmentList(*segments1)
        seg_list2 = SegmentList(*segments2)

        return SegmentList.intersection(seg_list1, seg_list2)

    @staticmethod
    def union(he1: HEdge, he2: HEdge):
        new_nucls = []
        new_positions = []

        pos_segments1 = he1.get_position_segments()
        pos_segments2 = he2.get_position_segments()
        nucl_segments1 = he1.get_nucl_segments()
        nucl_segments2 = he2.get_nucl_segments()

        nucls = []
        positions = []

        idx1, idx2 = 0, 0
        i1, i2 = 0, 0
        nucl, pos = None, None
        while idx1 < len(pos_segments1) and idx2 < len(pos_segments2):
            while i1 < len(pos_segments1[idx1]) and i2 < len(pos_segments2[idx2]):
                if pos_segments1[idx1][i1] < pos_segments2[idx2][i2]:
                    pos = pos_segments1[idx1][i1]
                    nucl = nucl_segments1[idx1][i1]
                    i1 += 1
                elif pos_segments1[idx1][i1] > pos_segments2[idx2][i2]:
                    pos = pos_segments2[idx2][i2]
                    nucl = nucl_segments2[idx2][i2]
                    i2 += 1
                else:
                    pos = pos_segments1[idx1][i1]
                    nucl = nucl_segments1[idx1][i1]
                    i1 += 1
                    i2 += 1

                if len(positions) and positions[-1] + 1 != pos:
                    new_positions.append(positions)
                    new_nucls.append(nucls)
                    positions = []
                    nucls = []

                positions.append(pos)
                nucls.append(nucl)

            if i1 == len(pos_segments1[idx1]):
                idx1 += 1
                i1 = 0
            if i2 == len(pos_segments2[idx2]):
                idx2 += 1
                i2 = 0

        # add rest of he1
        while idx1 < len(pos_segments1):
            while i1 < len(pos_segments1[idx1]):
                pos = pos_segments1[idx1][i1]
                nucl = nucl_segments1[idx1][i1]
                if len(positions) and positions[-1] + 1 != pos:
                    new_positions.append(positions)
                    new_nucls.append(nucls)
                    positions = []
                    nucls = []

                positions.append(pos)
                nucls.append(nucl)
                i1 += 1
            idx1 += 1
            i1 = 0

        # add rest of he2
        while idx2 < len(pos_segments2):
            while i2 < len(pos_segments2[idx2]):
                pos = pos_segments2[idx2][i2]
                nucl = nucl_segments2[idx2][i2]
                if len(positions) and positions[-1] + 1 != pos:
                    new_positions.append(positions)
                    new_nucls.append(nucls)
                    positions = []
                    nucls = []

                positions.append(pos)
                nucls.append(nucl)
                i2 += 1
            idx2 += 1
            i2 = 0

        if len(positions):
            new_positions.append(positions)
            new_nucls.append(nucls)

        # TODO add the start positions here
        new_hedge = HEdge.build(
            start_pos=[min(he1.start_pos, he2.start_pos)],
            positions=new_positions,
            nucls=new_nucls,
            genome2snp=he1.genome2snp,
            snp2genome=he1.snp2genome,
            reindex_snps=False)
        new_hedge.init_weight(
            value=min(he1.weight, he2.weight),
            coverage=Coverage.union(he1.coverage, he2.coverage)
        )
        new_hedge.frequency = min(he1.frequency, he2.frequency)
        return new_hedge

    def init_weight(self, value: int, coverage: Coverage = None):
        raise NotImplementedError('Must be overrided in subclass')

    @property
    def snp_count(self):
        raise NotImplementedError('Must be overrided in subclass')

    def begin(self) -> int:
        raise NotImplementedError('Must be overrided in subclass')

    def end(self) -> int:
        raise NotImplementedError('Must be overrided in subclass')

    def begin_in_genome(self) -> int:
        raise NotImplementedError('Must be overrided in subclass')

    def end_in_genome(self) -> int:
        raise NotImplementedError('Must be overrided in subclass')

    def get_position_segments(self) -> List[List[int]]:
        raise NotImplementedError('Must be overrided in subclass')

    def get_nucl_segments(self) -> List[List[str]]:
        raise NotImplementedError('Must be overrided in subclass')

    def get_flatten_snp(self) -> List[Tuple[int, str]]:
        raise NotImplementedError('Must be overrided in subclass')

    def __len__(self):
        return self.end_in_genome() - self.begin_in_genome() + 1

    @staticmethod
    def reindex_to_snp_order(
            positions: List[int],
            genome2snp: Dict[int, int]
    ) -> List[int]:
        return [genome2snp[pos] for pos in positions]


class SingleHEdge(HEdge):
    def __init__(
            self,
            positions: List[int],
            nucls: List[str],
            snp2genome: Dict[int, int],
            genome2snp: Dict[int, int],
            start_pos: int,
    ):
        assert len(positions) == len(nucls), f'{len(positions)} != {len(nucls)}'
        assert len(positions), 'Hyper Edge is empty'

        super().__init__(snp2genome, genome2snp)
        self.count_snp = len(positions)
        self.positions = positions
        self.nucls = nucls
        self._hash = hash(frozenset(zip(self.positions, self.nucls)))
        self.start_pos = start_pos
        self.snp_to_nucl = {snp: nucl for snp, nucl in zip(positions, nucls)}

    def init_weight(self, value: int, coverage: Coverage = None):
        self.weight = value
        if coverage is None:
            self.coverage.add(
                start_position=[self.positions[0]],
                end_position=[self.positions[-1]],
                value=self.weight
            )
        else:
            self.coverage = coverage

    @property
    def snp_count(self):
        return len(self.positions)

    def begin(self) -> int:
        return self.positions[0]

    def end(self) -> int:
        return self.positions[-1]

    def begin_in_genome(self) -> int:
        return self.snp2genome[self.positions[0]]

    def end_in_genome(self) -> int:
        return self.snp2genome[self.positions[-1]]

    def get_position_segments(self) -> List[List[int]]:
        return [self.positions]

    def get_nucl_segments(self) -> List[List[str]]:
        return [self.nucls]

    def get_flatten_snp(self) -> List[Tuple[int, str]]:
        genome_positions = [self.snp2genome[snp_idx] for snp_idx in self.positions]
        return list(zip(genome_positions, self.nucls))

    def __repr__(self):
        return f'SingleHEdge(L={self.positions[0]}, R={self.positions[-1]}, frequency={self.frequency}, snps={self.nucls})'


class PairedHEdge(HEdge):
    def __init__(
            self,
            left_positions: List[int],
            left_nucls: List[str],
            right_positions: List[int],
            right_nucls: List[str],
            snp2genome: Dict[int, int],
            genome2snp: Dict[int, int],
    ):
        assert len(left_positions) == len(left_nucls), f'{len(left_positions)} != {len(left_nucls)}'
        assert len(right_positions) == len(right_nucls), f'{len(right_positions)} != {len(right_nucls)}'
        assert len(left_positions), 'Left part of PairedHEdge is empty'
        assert len(right_positions), 'Right part of PairedHEdge is empty'

        super().__init__(snp2genome, genome2snp)
        self.count_snp = len(left_positions) + len(right_positions)
        self.left_positions = left_positions
        self.left_nucls = left_nucls
        self.right_positions = right_positions
        self.right_nucls = right_nucls
        self._hash = hash(frozenset(zip(
            self.left_positions + self.right_positions,
            self.left_nucls + self.right_nucls
        )))

    def init_weight(self, value: int, coverage: Coverage = None):
        self.weight = value
        if coverage is None:
            self.coverage.add(start_position=[self.left_positions[0], self.right_positions[0]],
                              end_position=[self.left_positions[-1], self.right_positions[-1]],
                              value=self.weight
                              )
        else:
            self.coverage = coverage

    @property
    def snp_count(self):
        return len(self.left_positions) + len(self.right_positions)

    def begin(self) -> int:
        return self.left_positions[0]

    def end(self) -> int:
        return self.right_positions[-1]

    def begin_in_genome(self) -> int:
        return self.snp2genome[self.left_positions[0]]

    def end_in_genome(self) -> int:
        return self.snp2genome[self.right_positions[-1]]

    def first_segment_in_genome(self) -> Tuple[int, int]:
        return self.snp2genome[self.left_positions[0]], self.snp2genome[self.left_positions[-1]]

    def second_segment_in_genome(self) -> Tuple[int, int]:
        return self.snp2genome[self.right_positions[0]], self.snp2genome[self.right_positions[-1]]

    def get_position_segments(self) -> List[List[int]]:
        return [self.left_positions, self.right_positions]

    def get_nucl_segments(self) -> List[List[str]]:
        return [self.left_nucls, self.right_nucls]

    def get_flatten_snp(self) -> List[Tuple[int, str]]:
        genome_positions = [
            self.snp2genome[snp_idx]
            for snp_idx in self.left_positions + self.right_positions
        ]
        return list(zip(genome_positions, self.left_nucls + self.right_nucls))

    def __repr__(self):
        return f'PairedHEdge(' \
               f'L1={self.left_positions[0]}, R1={self.left_positions[-1]}, ' \
               f'L2={self.right_positions[0]}, R2={self.right_positions[-1]}, ' \
               f'weight={self.weight})'


if __name__ == '__main__':
    # TODO Tests coverage
    # Test linkage
    # case 1
    snp2genome = dict(zip(range(1, 4), range(1, 4)))
    genome2snp = dict(zip(range(1, 4), range(1, 4)))
    he1 = HEdge.build([[1, 2, 3]], [['A', 'T', 'C']], snp2genome, genome2snp)
    he1.init_weight(1)
    he2 = HEdge.build([[1, 2], [3]], [['A', 'T'], ['C']], snp2genome, genome2snp)
    he2.init_weight(2)

    intersection = HEdge.intersect(he1, he2)
    is_allowed_merge_linkage = HEdge.is_allowed_merge_linkage(he1, he2, intersection)
    assert is_allowed_merge_linkage, 's - p'
    print(is_allowed_merge_linkage)

    h = HEdge.union(he1, he2)
    assert len(h.get_nucl_segments()) == len(h.get_position_segments()) == 1
    assert h.get_nucl_segments() == [['A', 'T', 'C']]
    assert h.weight == 3

    # case 2
    he3 = HEdge.build([[1], [2, 3]], [['A'], ['T', 'C']], snp2genome, genome2snp)
    intersection = HEdge.intersect(he3, he2)
    is_allowed_merge_linkage = HEdge.is_allowed_merge_linkage(he3, he2, intersection)
    assert is_allowed_merge_linkage, 'p - p'
    print(is_allowed_merge_linkage)

    h = HEdge.union(he3, he2)
    assert len(h.get_nucl_segments()) == len(h.get_position_segments()) == 1
    assert h.get_nucl_segments() == [['A', 'T', 'C']]
    assert h.weight == 2

    # case 3
    he4 = HEdge.build([[2, 3]], [['T', 'C']], snp2genome, genome2snp)
    intersection = HEdge.intersect(he1, he3)
    is_allowed_merge_linkage = HEdge.is_allowed_merge_linkage(he1, he3, intersection)
    assert is_allowed_merge_linkage, 's - s'
    print(is_allowed_merge_linkage)
