from typing import Tuple, List, Dict

from scripts.HyperEdge.Coverage import Coverage
from scripts.HyperEdge.HyperEdge import HyperEdge


class PairedHyperEdge(HyperEdge):
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
        self.left_nucls = ''.join(left_nucls)
        self.right_positions = right_positions
        self.right_nucls = ''.join(right_nucls)
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

    def get_nucl_segments(self) -> List[str]:
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
