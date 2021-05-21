import math
from collections import defaultdict
# from functools import reduce
from copy import deepcopy
from typing import List, Mapping, Tuple, Set, Dict
from graphviz import Digraph
from scipy.stats import norm, binom
from tqdm import tqdm
import numpy as np

from scripts.hedge import HEdge


def plot_graph(g):
    d = Digraph()
    for u, vertexes in g.items():
        for v in set(vertexes):
            d.edge(str(u), str(v))

    name = 'snv-w=80'
    d.format = 'png'
    d.render(f'output/graphviz/{name}.gv', view=True)


def plot_graph_bfs(g, vertex_count, depth=5):
    s, t = 0, vertex_count - 1
    used = [False for _ in range(vertex_count)]
    used[s] = True

    d = Digraph()

    q_cur = [s]
    q_next = []
    for _ in range(depth):
        print('q_cur', len(q_cur))
        for u in q_cur:
            for v in g[u]:
                if not used[v]:
                    used[v] = True
                    d.edge(str(u), str(v))
                    q_next.append(v)
        q_cur = q_next
        q_next = []

    name = f'bfs_prefix_{depth}'
    d.format = 'png'
    d.render(f'output/graphviz/{name}.gv', view=True)


def get_distance_between_hedges(hedge1: HEdge, hedge2: HEdge):
    nucls1 = hedge1.nucls
    nucls2 = hedge2.nucls
    distance = 0
    for i, j in zip(nucls1, nucls2):
        if i != j:
            distance += 1
    return distance


def remove_leftovers(hedges: Dict[frozenset, Dict[str, HEdge]], error_prob):
    for key in hedges:
        hedges_dict = hedges[key]
        popping = []
        heavy_hedges = [h for n, h in hedges_dict.items() if (h.frequency > error_prob and h.weight > 0)]
        light_hedges = [h for n, h in hedges_dict.items() if (h.frequency <= error_prob or h.weight == 0)]
        for hedge in light_hedges:
            max_proba, max_heavy_hedge = 0, None
            if hedge.weight != 0:
                for heavy_hedge in heavy_hedges:
                    distance = get_distance_between_hedges(hedge, heavy_hedge)
                    proba = binom.pmf(x=hedge.weight, n=hedge.weight + heavy_hedge.weight, p=error_prob ** distance)
                    if proba > max_proba:
                        max_proba, max_heavy_hedge = proba, heavy_hedge
                if max_heavy_hedge is not None:
                    hedges_dict[max_heavy_hedge.nucls].weight += hedge.weight
                    hedges_dict[max_heavy_hedge.nucls].frequency = (hedge.weight * hedge.frequency +
                                                                    max_heavy_hedge.weight * max_heavy_hedge.frequency) / (
                                                                           hedge.weight + max_heavy_hedge.weight)
            popping.append(hedge.nucls)
        for p in popping:
            # print(hedges_dict, p)
            hedges_dict.pop(p)
        hedges[key] = hedges_dict
    return hedges


def get_intervals(xi_i, p_xi, frequencies, alpha=0.85):
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
        hedges: Dict[frozenset, Dict[str, HEdge]],
        target_snp_count: int,
        error_probability: float = 0,
        verbose: bool = False,
        debug: bool = False
) -> Tuple[List[HEdge], Mapping[str, List]]:
    ever_created_hedges = deepcopy(hedges)
    metrics = defaultdict(list)
    if verbose:
        print('----Algo started----')

    remove_leftovers(hedges, error_probability)

    keys_to_delete = set()
    hedges_keys = list(hedges.keys())
    for key1_i in range(len(hedges_keys)):
        key1 = hedges_keys[key1_i]
        for key2_i in range(key1_i + 1, len(hedges_keys)):
            key2 = hedges_keys[key2_i]
            if key1.issubset(key2):
                keys_to_delete.add(key1)
            elif key2.issubset(key1):
                keys_to_delete.add(key2)
    print(keys_to_delete)
    for key in keys_to_delete:
        del hedges[key]

    pairs = get_pairs(hedges, ever_created_hedges)
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

        pair = max(pairs, key=lambda x: (get_intersection_snp_length(x),
                                         get_union_snp_length(x),
                                         min(x[0].frequency, x[1].frequency),
                                         -abs(x[0].frequency - x[1].frequency),
                                         min(x[0].positions[0], x[1].positions[0]),
                                         ))
        index = pairs.index(pair)
        he1 = pair[0]
        he2 = pair[1]
        if verbose:
            print(f'Merging {pair[0]} and {pair[1]}')
        new_hedge = HEdge.union(he1, he2)
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
                hedges = remove_leftovers(hedges, error_probability)
                pairs = get_pairs(hedges, ever_created_hedges)
                continue
        pairs[index][0].weight *= freq1 * 100 / pairs[index][0].frequency
        pairs[index][1].weight *= freq2 * 100 / pairs[index][1].frequency
        pairs[index][0].frequency = freq1 * 100
        pairs[index][1].frequency = freq2 * 100
        # print(pairs[i])
        hedges[frozenset(pairs[index][0].positions)][he1.nucls] = pairs[index][0]
        hedges[frozenset(pairs[index][1].positions)][he2.nucls] = pairs[index][1]
        hedges = remove_leftovers(hedges, error_probability)
        pairs = get_pairs(hedges, ever_created_hedges)
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


def get_intersection_snp_length(pair: Tuple[HEdge]):
    intersection = set(pair[0].positions).intersection(set(pair[1].positions))
    return len(intersection)


def get_union_snp_length(pair: Tuple[HEdge]):
    union = set(pair[0].positions).union(set(pair[1].positions))
    return len(union)


def get_pairs(hedges: Dict[frozenset, Dict[str, HEdge]], ever_created_hedges: Dict[frozenset, Dict[str, HEdge]]):
    int_sets = list(k for k in hedges.keys() if len(hedges[k]) > 0)
    interesting_snp_sets = set(k for k in hedges.keys() if len(hedges[k]) > 0)
    found_any_new_conditions = False
    pairs = []
    for set1_i in range(len(int_sets)):
        set1 = int_sets[set1_i]
        for set2_i in range(set1_i + 1, len(interesting_snp_sets)):
            set2 = int_sets[set2_i]
            if min(max(set1), max(set2)) + 1 < max(min(set1), min(set2)):
                continue
            if set1 != set2:
                if not set1.issubset(set2) and not set2.issubset(set1):
                    found_any_new_conditions = True
                # check_for_snps = not set1.isdisjoint(set2)
                for nucls1, hedge1 in hedges[set1].items():
                    for nucls2, hedge2 in hedges[set2].items():
                        can_merge = True
                        for pos in hedge1.positions:
                            if pos in hedge2.positions:
                                if hedge1.snp_to_nucl[pos] != hedge2.snp_to_nucl[pos]:
                                    can_merge = False
                                    break

                        if set(hedge1.positions).issubset(set(hedge2.positions)) or \
                                set(hedge2.positions).issubset(set(hedge1.positions)):
                            continue
                        if can_merge:
                            new_hedge = HEdge.union(hedge1, hedge2)
                            new_h_nucls = new_hedge.nucls
                            frozen_positions = frozenset(new_hedge.positions)
                            if frozen_positions not in ever_created_hedges or new_h_nucls not in ever_created_hedges[
                                frozen_positions]:
                                hedge1.used = False
                                hedge2.used = False
                                pairs.append((hedge1, hedge2))
    return pairs if found_any_new_conditions else []
