import math
from collections import defaultdict
# from functools import reduce
from copy import deepcopy
from typing import List, Mapping, Tuple, Set, Dict
from graphviz import Digraph
from scipy.stats import norm, binom
from tqdm import tqdm
import numpy as np
# from sklearn.cluster import AgglomerativeClustering

from scripts.hedge import HEdge
from scripts.utils import DSU


def create_hedge_connectivity_graph(hedges: List[HEdge]):
    g = defaultdict(list)
    edges_count = 0

    def add_edge(u, v):
        nonlocal edges_count
        edges_count += 1
        g[u].append(v)

    # vertex 0 for fake source
    for i, he_i in enumerate(tqdm(hedges, desc='Create vertexes'), start=1):
        for j, he_j in enumerate(hedges, start=1):
            if i < j:
                if he_i.is_consistent_with(he_j):
                    g[i].append(j)
                    if he_i < he_j:
                        add_edge(i, j)
                    elif he_j < he_i:
                        add_edge(j, i)
                    else:
                        # TODO not comparable
                        pass

    return g, edges_count


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


def calc_in_out_deg(g) -> Tuple[Mapping[int, int], Mapping[int, int]]:
    in_deg = defaultdict(int)
    out_deg = defaultdict(int)

    for u, vertexes in g.items():
        out_deg[u] = len(vertexes)
        for v in set(vertexes):
            in_deg[v] += 1
    return in_deg, out_deg


def get_inv_graph(g) -> Mapping[int, List[int]]:
    inv_g = defaultdict(list)
    for u, vertexes in g.items():
        for v in set(vertexes):
            inv_g[v].append(u)
    return inv_g


def calc_paths_count(inv_g, vertex_count) -> List[int]:
    used = [False for _ in range(vertex_count)]
    paths_count = [0 for _ in range(vertex_count)]
    s, t = 0, vertex_count - 1

    paths_count[s] = 1
    used[s] = True

    def count(v):
        if used[v]:
            return paths_count[v]

        acc = 0
        for u in inv_g[v]:
            if not used[u]:
                count(u)
            acc += paths_count[u]

        paths_count[v] = acc
        used[v] = True

    count(t)
    return paths_count


def calc_graph_depth(g, vertex_count) -> List[int]:
    INF = int(1e5)
    used = [False for _ in range(vertex_count)]
    dist = [INF for _ in range(vertex_count)]
    s, t = 0, vertex_count - 1

    dist[s] = 0
    used[s] = True

    q = [s]
    while len(q):
        q_next = []
        for u in q:
            for v in g[u]:
                dist[v] = min(dist[u] + 1, dist[v])
                if not used[v]:
                    used[v] = True
                    q_next.append(v)
        q = q_next
    return dist


def find_conflicting_hedges(hedges: List[HEdge]) -> Set[int]:
    conflicting_hedge_indexes = set()
    for i, he in enumerate(tqdm(hedges, desc='Searching conflicting hedges')):
        for he_other in hedges:
            if he.is_ambiguous_with(he_other):
                conflicting_hedge_indexes.add(i)
                break
    return conflicting_hedge_indexes


def algo_hedge_connectivity_graph(hedges: List[HEdge], save_graph_bfs_path=False):
    # Build graph
    graph, edges_count = create_hedge_connectivity_graph(hedges)
    vertex_count = len(hedges)
    print('Vertexes count', vertex_count)
    print('Edges count', edges_count)
    del edges_count

    in_deg, out_deg = calc_in_out_deg(graph)

    # Add source and target vertexes
    s, t = 0, vertex_count + 1
    for v in range(1, vertex_count + 1):
        if not in_deg[v] and out_deg[v]:
            graph[s].append(v)
        if in_deg[v] and not out_deg[v]:
            graph[v].append(t)
    vertex_count += 2
    print('Source vertex count:', len(graph[s]))
    print('Target vertex count:', len([u for u, vertexes in graph.items() if t in vertexes]))

    if save_graph_bfs_path:
        for depth in range(1, 2):
            plot_graph_bfs(graph, vertex_count, depth)
            print(f'Depth[{depth}]: done')

    # inv_g = get_inv_graph(g)
    # paths_count = calc_paths_count(inv_g, vertex_count)
    # print('Total paths count:', paths_count)

    dist = calc_graph_depth(graph, vertex_count)
    print('Shortest path len:', dist[t])


def eval_coverage_dist_matrix(hedges: List[HEdge], INF) -> np.ndarray:
    coverages = [hedge.avg_coverage for hedge in hedges]

    n = len(hedges)
    coverage_dist_matrix = np.full(shape=(n, n), fill_value=INF)

    for he1_idx, he1, in enumerate(hedges):
        for he2_idx, he2 in enumerate(hedges):
            if he1_idx < he2_idx:
                if not he1.is_overlapped_with(he2):
                    coverage_dist_value = abs(coverages[he1_idx] - coverages[he2_idx])
                    coverage_dist_matrix[he1_idx][he2_idx] = coverage_dist_value
                    coverage_dist_matrix[he2_idx][he1_idx] = coverage_dist_value
    return coverage_dist_matrix


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
            print(hedges_dict, p)
            hedges_dict.pop(p)
        hedges[key] = hedges_dict
    return hedges


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
    print(sum_sample_interval, frequencies, freq1, freq2)
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
    print('----Algo started----')

    remove_leftovers(hedges, error_probability)
    pairs = get_pairs(hedges, ever_created_hedges)
    old_hedges = None
    haplo_hedges = []
    while len(pairs) > 0:
        print('----Iteration started----')
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
        print(f'Merging {pair[0]} and {pair[1]}')
        new_hedge = HEdge.union(he1, he2)
        freq1, freq2, freq_new = check_leftovers_distribution(he1.weight, he1.frequency, he2.weight, he2.frequency)
        new_hedge.frequency = freq_new * 100
        print(freq1, freq2, freq_new)
        new_hedge.weight = (1 - freq1) * he1.weight + (1 - freq2) * he2.weight
        if len(new_hedge.positions) == target_snp_count:
            assert new_hedge.frequency > error_probability * 100
            haplo_hedges.append(new_hedge)
        else:
            new_h_nucls = new_hedge.nucls
            frozen_positions = frozenset(new_hedge.positions)
            print('frozen', frozen_positions)
            print('ever_created', ever_created_hedges)
            print('new_nucls', new_h_nucls)
            if frozen_positions not in ever_created_hedges or new_h_nucls not in ever_created_hedges[frozen_positions]:
                hedges[frozen_positions][new_h_nucls] = new_hedge
                ever_created_hedges[frozen_positions][new_h_nucls] = new_hedge
            else:
                print('-- stopped')
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
