from scripts.HyperEdge.hyper_test import HyperEdge


def test_not_conflicting_hedges__with_common_positions():
    """
    ATG
     ||
     TGA
    """
    he1 = HyperEdge(positions=[1, 2, 3], nucls=list('ATG'))
    he2 = HyperEdge(positions=[2, 3, 4], nucls=list('TGA'))
    assert not he1.is_ambiguous_with(he2)


def test_not_conflicting_hedges__without_common_positions():
    """
    ATG
       TGA
    """
    he1 = HyperEdge(positions=[1, 2, 3], nucls=list('ATG'))
    he2 = HyperEdge(positions=[4, 5, 6], nucls=list('TGA'))
    assert not he1.is_ambiguous_with(he2)


def test_not_conflicting_hedges__all_common_positions_are_distinct():
    """
    ATG
     **
     GTA
    """
    he1 = HyperEdge(positions=[1, 2, 3], nucls=list('ATG'))
    he2 = HyperEdge(positions=[2, 3, 4], nucls=list('GTA'))
    assert not he1.is_ambiguous_with(he2)


def test_not_conflicting_hedges__distinct_endings_of_common_positions():
    """
    ATAG
     *|*
     GATA
    """
    he1 = HyperEdge(positions=[1, 2, 3, 4], nucls=list('ATAG'))
    he2 = HyperEdge(positions=[2, 3, 4, 5], nucls=list('GATA'))
    assert not he1.is_ambiguous_with(he2)


def test_conflicting_hedges__distinct_leftmost_common_nucl():
    """
    AAG
     *|
     TGA
    """
    he1 = HyperEdge(positions=[1, 2, 3], nucls=list('AAG'))
    he2 = HyperEdge(positions=[2, 3, 4], nucls=list('TGA'))
    assert he1.is_ambiguous_with(he2), 'Leftmost common nucl must raise conflict'


def test_conflicting_hedges__distinct_rightmost_common_nucl():
    """
    AAG
     |*
     ATA
    """
    he1 = HyperEdge(positions=[1, 2, 3], nucls=list('AAG'))
    he2 = HyperEdge(positions=[2, 3, 4], nucls=list('ATA'))
    assert he1.is_ambiguous_with(he2), 'Rightmost common nucl must raise conflict'


def test_merge_hedges__intersection():
    """
    ATG
     ||
     TGA
    """
    he1 = HyperEdge(positions=[1, 2, 3], nucls=list('ATG'))
    he2 = HyperEdge(positions=[2, 3, 4], nucls=list('TGA'))
    he = he1.merge_with(he2)
    assert he.positions == [1, 2, 3, 4]
    assert he.nucls == list('ATGA')
    assert he1.merge_with(he2) == he2.merge_with(he1)


def test_merge_hedges__inclusion():
    """
    ATGA
     ||
     TG
    """
    he1 = HyperEdge(positions=[1, 2, 3, 4], nucls=list('ATGA'))
    he2 = HyperEdge(positions=[2, 3], nucls=list('TG'))
    he = he1.merge_with(he2)
    assert he.positions == [1, 2, 3, 4]
    assert he.nucls == list('ATGA')
    assert he1.merge_with(he2) == he2.merge_with(he1)
