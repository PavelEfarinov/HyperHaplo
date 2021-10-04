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
    ever_created_hedges = deepcopy(hedges)
    metrics = defaultdict(list)
    if verbose:
        print('----Algo started----')

    hedges.remove_leftovers(error_probability)

    hedges.remove_referencing_hyperedges()

    pairs = hedges.get_pairs(ever_created_hedges)
    old_hedges = None
    haplo_hedges = []
    while len(pairs) > 0:
        print('Iteration started, pairs count: ', len(pairs))
        if verbose:
            print('Current hedges')
            for s, h in hedges.items():
                print(s)
                print(h)

            print('-------------printing pairs---------')
            for pair in pairs:
                print(pair)

        pair = get_max_pair(pairs)

        index = pairs.index(pair)
        he1 = pair[0]
        he2 = pair[1]
        if verbose:
            print(f'Merging {pair[0]} and {pair[1]}')
        new_hedge = hyper_edges_union(he1, he2)
        freq1, freq2, freq_new = check_leftovers_distribution(he1.weight, he1.frequency, he2.weight, he2.frequency)
        new_hedge.frequency = freq_new * 100
        new_hedge.weight = (1 - freq1) * he1.weight + (1 - freq2) * he2.weight
        if len(new_hedge.positions) == target_snp_count:
            if new_hedge.frequency > error_probability * 100:
                haplo_hedges.append(new_hedge)
        else:
            new_h_nucls = new_hedge.nucls
            frozen_positions = frozenset(new_hedge.positions)
            if frozen_positions not in ever_created_hedges or new_h_nucls not in ever_created_hedges[frozen_positions]:
                hedges[frozen_positions][new_h_nucls] = new_hedge
                ever_created_hedges[frozen_positions][new_h_nucls] = new_hedge
            else:
                hedges.remove_leftovers(error_probability)
                pairs = hedges.get_pairs(ever_created_hedges)
                continue
        pairs[index][0].weight *= freq1 * 100 / pairs[index][0].frequency
        pairs[index][1].weight *= freq2 * 100 / pairs[index][1].frequency
        pairs[index][0].frequency = freq1 * 100
        pairs[index][1].frequency = freq2 * 100
        # print(pairs[i])
        hedges[frozenset(pairs[index][0].positions)][he1.nucls] = pairs[index][0]
        hedges[frozenset(pairs[index][1].positions)][he2.nucls] = pairs[index][1]
        hedges.remove_leftovers(error_probability)
        pairs = hedges.get_pairs(ever_created_hedges)
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
