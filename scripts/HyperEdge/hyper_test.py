from __future__ import annotations


from scripts.HyperEdge.hyper_utils import build_hyper_edge, intersect_hyper_edges, hyper_edges_union, \
    is_allowed_merge_linkage

if __name__ == '__main__':
    # TODO Tests coverage
    # Test linkage
    # case 1
    snp2genome = dict(zip(range(1, 4), range(1, 4)))
    genome2snp = dict(zip(range(1, 4), range(1, 4)))
    he1 = build_hyper_edge([[1, 2, 3]], [['A', 'T', 'C']], snp2genome, genome2snp)
    he1.init_weight(1)
    he2 = build_hyper_edge([[1, 2], [3]], [['A', 'T'], ['C']], snp2genome, genome2snp)
    he2.init_weight(2)

    intersection = intersect_hyper_edges(he1, he2)
    is_allowed = is_allowed_merge_linkage(he1, he2, intersection)
    assert is_allowed, 's - p'
    print(is_allowed)

    h = hyper_edges_union(he1, he2)
    assert len(h.get_nucl_segments()) == len(h.get_position_segments()) == 1
    assert h.get_nucl_segments() == [['A', 'T', 'C']]
    assert h.weight == 3

    # case 2
    he3 = build_hyper_edge([[1], [2, 3]], [['A'], ['T', 'C']], snp2genome, genome2snp)
    intersection = intersect_hyper_edges(he3, he2)
    is_allowed = is_allowed_merge_linkage(he3, he2, intersection)
    assert is_allowed, 'p - p'
    print(is_allowed)

    h = hyper_edges_union(he3, he2)
    assert len(h.get_nucl_segments()) == len(h.get_position_segments()) == 1
    assert h.get_nucl_segments() == [['A', 'T', 'C']]
    assert h.weight == 2

    # case 3
    he4 = build_hyper_edge([[2, 3]], [['T', 'C']], snp2genome, genome2snp)
    intersection = intersect_hyper_edges(he1, he3)
    is_allowed = is_allowed_merge_linkage(he1, he3, intersection)
    assert is_allowed, 's - s'
    print(is_allowed)
