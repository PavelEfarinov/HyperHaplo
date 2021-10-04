from typing import Dict, List, Tuple

from scripts.HyperEdge.Coverage import Coverage
from scripts.HyperEdge.HyperEdge import HyperEdge


class SingleHyperEdge(HyperEdge):
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
        self.nucls = ''.join(nucls)
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

    def get_nucl_segments(self) -> List[str]:
        return [self.nucls]

    def get_flatten_snp(self) -> List[Tuple[int, str]]:
        genome_positions = [self.snp2genome[snp_idx] for snp_idx in self.positions]
        return list(zip(genome_positions, self.nucls))

    def __repr__(self):
        return f'SingleHEdge(L={self.positions[0]}, R={self.positions[-1]}, frequency={self.frequency}, weight={self.weight}, snps={self.nucls})'
