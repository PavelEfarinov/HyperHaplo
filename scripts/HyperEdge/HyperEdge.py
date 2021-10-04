from typing import Dict, List, Tuple

from scripts.HyperEdge.Coverage import Coverage


class HyperEdge:
    def __init__(
            self,
            snp2genome: Dict[int, int],
            genome2snp: Dict[int, int],
    ):
        self.snp2genome = snp2genome
        self.genome2snp = genome2snp
        self.weight = 0
        self.positions = 0
        self.nucls = 0
        self.count_snp = 0
        self.coverage = Coverage()
        self.frequency = 1
        self.start_pos = -1
        self.used = False
        self.snp_to_nucl = {}
        self.edge_ids = set()

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

        def __repr__(self):
            return f'Overlap(\n' \
                   f'fst_positions_count={self.fst_positions_count},\n' \
                   f'snd_positions_count={self.snd_positions_count},\n' \
                   f'overlap_size={self.overlap_size},\n' \
                   f'match_size={self.match_size},\n' \
                   f'positional_jaccard={round(self.positional_jaccard, 3)},\n' \
                   f'overlap_accuracy={round(self.overlap_accuracy, 3)},\n' \
                   f'is_consistent={self.is_consistent})'

    def is_ambiguous_with(self, other) -> bool:
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

    def merge_with_(self, other):
        """
        Inplace merging self with other hedge.
        """
        result = self.merge_with(other)

        # Reset attributes
        self.positions = result.positions
        self.nucls = result.nucls
        self.weight += other.weight
        for i, cover in enumerate(other.coverage):
            self.coverage[i] += result.cover
        # if 'avg_coverage' in self.__dict__:
        # Invalidate cache
        # del self.__dict__['avg_coverage']

    def merge_with(self, other):
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
        return HyperEdge(
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
