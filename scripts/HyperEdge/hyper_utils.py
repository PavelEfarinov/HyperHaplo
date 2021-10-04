from more_itertools import pairwise
from typing import Optional, Dict, List

from scripts.HyperEdge.Coverage import Coverage
from scripts.HyperEdge.HyperEdge import HyperEdge
from scripts.HyperEdge.PairedHyperEdge import PairedHyperEdge
from scripts.HyperEdge.SingleHyperEdge import SingleHyperEdge
from scripts.segment import SegmentList, LocalSegmentIntersection


def reindex_to_snp_order(
        positions: List[int],
        genome2snp: Dict[int, int]
) -> List[int]:
    return [genome2snp[pos] for pos in positions]


def metrics(
        he1: HyperEdge,
        he2: HyperEdge,
        intersection: List[LocalSegmentIntersection]
) -> HyperEdge.Overlap:
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

    return HyperEdge.Overlap(
        fst_positions_count=sum(map(len, nucls1)),
        snd_positions_count=sum(map(len, nucls2)),
        overlap_size=size,
        match_size=match_size
    )


def overlap_size(intersection: List[LocalSegmentIntersection]) -> int:
    """How many common positions h1 and h2 have.
    Time O(len(intersection)).
    """
    size = 0
    for inter_item in intersection:
        l, r = inter_item.first_positions_boundaries
        size += r - l + 1
    return size


def is_allowed_merge_linkage(
        he1: HyperEdge,
        he2: HyperEdge,
        intersection: List[LocalSegmentIntersection]
) -> bool:
    # TODO перебрать варианты
    first_segments_indexes = set(inter.first_segment_idx for inter in intersection)
    second_segments_indexes = set(inter.second_segment_idx for inter in intersection)

    if isinstance(he1, PairedHyperEdge) and isinstance(he2, PairedHyperEdge):
        return len(first_segments_indexes) == len(second_segments_indexes) == 2
    elif isinstance(he1, SingleHyperEdge) and isinstance(he2, SingleHyperEdge):
        return len(intersection) > 0
    else:
        # case 1: single and paired => 1 + 2
        # case 2: single and paired => 2 + 1
        return (len(first_segments_indexes) + len(second_segments_indexes)) == 3


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


def build_hyper_edge(
        positions: List[List[int]],
        nucls: List[List[str]],
        snp2genome: Dict[int, int],
        genome2snp: Dict[int, int],
        start_pos: List[int],
        reindex_snps: bool = True
) -> Optional[HyperEdge]:
    """ Builder method for HEdge subclasses.
    Args:
        positions: List of list of (SNP order) positions.
        nucls: List of list of nucleotides corresponding to positions argument.
        reindex_snps: If flat is set then `positions` are genome positions, else SNP indexes.
    Returns:
        HyperEdge: New HEdge instance.
    """
    assert len(positions) == len(nucls), f'{len(positions)} != {len(nucls)}'

    if reindex_snps:
        if len(positions) == 1:
            positions = [reindex_to_snp_order(positions[0], genome2snp)]

        if len(positions) == 2:
            positions = [
                reindex_to_snp_order(positions[0], genome2snp),
                reindex_to_snp_order(positions[1], genome2snp)
            ]

    if len(positions) == 2:
        # merge parts if they are subsequent
        if positions[0][-1] + 1 == positions[1][0]:
            positions = [positions[0] + positions[1]]
            nucls = [nucls[0] + nucls[1]]

    fail_reason = check_continuity_condition(positions)
    if fail_reason is not None:
        # TODO add logging level for warnings
        # print(fail_reason)
        return None

    if len(positions) == 1:
        return SingleHyperEdge(
            positions=positions[0],
            nucls=nucls[0],
            snp2genome=snp2genome,
            genome2snp=genome2snp,
            start_pos=start_pos[0],
        )
    elif len(positions) == 2:
        return PairedHyperEdge(
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


def intersect_hyper_edges(he1: HyperEdge, he2: HyperEdge) -> List[LocalSegmentIntersection]:
    segments1 = he1.get_position_segments()
    segments2 = he2.get_position_segments()
    seg_list1 = SegmentList(*segments1)
    seg_list2 = SegmentList(*segments2)

    return SegmentList.intersection(seg_list1, seg_list2)


def hyper_edges_union(he1: HyperEdge, he2: HyperEdge):
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

            append_new_positions_if_ended(pos, positions, nucls, new_positions, new_nucls)

            positions.append(pos)
            nucls.append(nucl)

        if i1 == len(pos_segments1[idx1]):
            idx1 += 1
            i1 = 0
        if i2 == len(pos_segments2[idx2]):
            idx2 += 1
            i2 = 0

    # add rest of he1
    process_rest_of_edge(idx1, i1, pos_segments1, nucl_segments1, positions, nucls, new_positions, new_nucls)

    # add rest of he2
    process_rest_of_edge(idx2, i2, pos_segments2, nucl_segments2, positions, nucls, new_positions, new_nucls)

    if len(positions):
        new_positions.append(positions)
        new_nucls.append(nucls)

    # TODO add the start positions here
    new_hedge = build_hyper_edge(
        start_pos=[min(he1.start_pos, he2.start_pos)],
        positions=new_positions,
        nucls=new_nucls,
        genome2snp=he1.genome2snp,
        snp2genome=he1.snp2genome,
        reindex_snps=False)
    new_hedge.init_weight(
        value=round(min(he1.frequency, he2.frequency) * max(he1.weight, he2.weight) / 100),
        coverage=coverages_union(he1.coverage, he2.coverage)
    )
    new_hedge.edge_ids = he1.edge_ids.hyper_edges_union(he2.edge_ids)
    new_hedge.frequency = min(he1.frequency, he2.frequency)
    return new_hedge


def append_new_positions_if_ended(current_position, current_positions, current_nucl, new_positions, new_nucls):
    if len(current_positions) and current_positions[-1] + 1 != current_position:
        new_positions.append(current_positions)
        new_nucls.append(current_nucl)
        current_positions.clear()
        current_nucl.clear()


def process_rest_of_edge(idx, i, pos_segments, nucl_segments, positions, nucls, new_positions, new_nucls):
    while idx < len(pos_segments):
        while i < len(pos_segments[idx]):
            pos = pos_segments[idx][i]
            nucl = nucl_segments[idx][i]
            append_new_positions_if_ended(pos, positions, nucls, new_positions, new_nucls)
            positions.append(pos)
            nucls.append(nucl)
            i += 1
        idx += 1
        i = 0


def coverages_union(cov1: Coverage, cov2: Coverage) -> Coverage:
    cov = Coverage()
    min_index = min([cov1.coverage_map.starts[0], cov2.coverage_map.starts[0]])
    max_index = max([cov1.coverage_map.starts[0], cov2.coverage_map.starts[0]])
    cov.add([min_index], [max_index], 1)
    return cov


def get_distance_between_hedges(hedge1: HyperEdge, hedge2: HyperEdge):
    nucls1 = hedge1.nucls
    nucls2 = hedge2.nucls
    distance = 0
    for i, j in zip(nucls1, nucls2):
        if i != j:
            distance += 1
    return distance
