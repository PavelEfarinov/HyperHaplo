import math
from collections import defaultdict
# from functools import reduce
from copy import deepcopy
from typing import List, Mapping, Tuple, Set, Dict
from graphviz import Digraph
from scipy.stats import norm, binom
from tqdm import tqdm
import numpy as np

from scripts.HyperEdge.HyperEdge import HyperEdge
from scripts.HyperEdge.hyper_utils import hyper_edges_union, coverages_union
from scripts.HyperGraph import HyperGraph


def get_intervals(xi_i, p_xi, frequencies, alpha=0.95):
    N = xi_i / p_xi
    dist_xi = norm.interval(alpha, loc=xi_i, scale=math.sqrt(xi_i * (1 - p_xi)))
    dist_xi = (dist_xi[0] / N, dist_xi[1] / N)
    return dist_xi


def get_frequencies(xi_i, p_xi, eta_i, p_eta):
    N = xi_i / p_xi
    K = eta_i / p_eta
    frequencies = {'average': (p_eta + p_xi) / 2, 'weighted': p_eta * K / (K + N) + p_xi * N / (K + N),
                   'minimum': min(p_xi, p_eta)}
    return frequencies


def check_leftovers_distribution(count1, freq1, count2, freq2):
    """
    :param count1:
    :param freq1:
    :param count2:
    :param freq2:
    :return: frequencies of left, right and new, normalized to 1
    """
    p1 = freq1 / 100
    p2 = freq2 / 100
    frequencies = get_frequencies(count1, p1, count2, p2)
    sum_sample_interval = get_intervals(count1 + count2, (p1 * count1 + p2 * count2) / (count1 + count2), frequencies)
    # interval1 = get_intervals(count1, p1, frequencies)
    # interval2 = get_intervals(count2, p2, frequencies)
    # print(sum_sample_interval, frequencies, freq1, freq2)
    if sum_sample_interval[0] < frequencies['average'] < sum_sample_interval[1]:
        return 0, 0, frequencies['weighted']
    else:
        return p1 - frequencies['minimum'], p2 - frequencies['minimum'], frequencies['minimum']


def algo_merge_hedge_contigs(
        hedges: HyperGraph,
        target_snp_count: int,
        error_probability: float = 0,
        verbose: bool = False,
        debug: bool = False
) -> Tuple[List[HyperEdge], Mapping[str, List]]:
    metrics = defaultdict(list)
    if verbose:
        print('----Algo started----')

    hedges.remove_leftovers(error_probability)

    hedges.remove_referencing_hyperedges()

    haplo_hedges = hedges.merge_all_pairs(target_snp_count, error_probability, verbose)

    print('----Finished algo----')
    print(haplo_hedges)
    print('----Recounting frequencies----')
    freq_sum = 0
    for hedge in haplo_hedges:
        freq_sum += hedge.frequency
    print(freq_sum)
    for i in range(len(haplo_hedges)):
        haplo_hedges[i].frequency = haplo_hedges[i].frequency / freq_sum * 100
    for haplo in haplo_hedges:
        print(haplo)
    return haplo_hedges, metrics

def get_max_pair(pairs):
    return max(pairs, key=lambda x: (get_intersection_snp_length(x),
                                     get_union_snp_length(x),
                                     get_union_pairs_percent(x),
                                     min(x[0].frequency, x[1].frequency),
                                     -abs(x[0].frequency - x[1].frequency),
                                     min(x[0].positions[0], x[1].positions[0]),
                                     ))


def get_intersection_snp_length(pair: Tuple[HyperEdge]):
    intersection = set(pair[0].positions).intersection(set(pair[1].positions))
    return len(intersection)


def get_union_snp_length(pair: Tuple[HyperEdge]):
    union = set(pair[0].positions).union(set(pair[1].positions))
    return len(union)


def get_union_pairs_percent(pair: Tuple[HyperEdge]):
    intersection = pair[0].edge_ids.intersection(pair[1].edge_ids)
    union = pair[0].edge_ids.hyper_edges_union(pair[1].edge_ids)
    return len(intersection) / len(union)
