from collections import defaultdict
# from functools import reduce
from typing import List, Mapping, Tuple, Set, Dict
from graphviz import Digraph
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


def get_hedges_merge_order(
        hedges: List[HEdge]
) -> Tuple[List[Tuple[float, int, int]], Mapping]:
    metrics = defaultdict(list)
    order = []
    for i, he1 in enumerate(tqdm(hedges, desc='Ordering hedges')):
        for j, he2 in enumerate(hedges):
            if i < j:
                jaccard = he1.positional_jaccard(he2)
                metrics['jaccard'].append(jaccard)

                accuracy = he1.overlapped_accuracy(he2)
                metrics['accuracy'].append(accuracy)

                consistency = he1.is_consistent_with(he2)
                metrics['consistency'].append(int(consistency))

                if consistency:
                    order.append((-jaccard, i, j))
    return order, metrics


def get_hedges_merge_order_in_window(
        sorted_hedges: List[HEdge],
        verbose: bool,
        save_metrics: bool = False
) -> Tuple[List[Tuple[int, float, int, Tuple[int, int]]], Dict[str, List]]:
    metrics = defaultdict(list)
    order = []

    with tqdm(total=len(sorted_hedges), desc='Linkage metrics evaluation', disable=not verbose) as pbar:
        window_begin = 0
        while window_begin < len(sorted_hedges):
            hedge = sorted_hedges[window_begin]  # current leader
            for i in range(window_begin + 1, len(sorted_hedges)):
                other_hedge = sorted_hedges[i]
                if hedge.end() < other_hedge.begin():
                    if save_metrics:
                        metrics['window_size'].append(i - window_begin)
                    break  # some of previous hedges were large than current leader
                intersection = HEdge.intersect(hedge, other_hedge)
                if HEdge.is_allowed_merge_linkage(hedge, other_hedge, intersection):
                    overlap = HEdge.Overlap.metrics(hedge, other_hedge, intersection)
                    if overlap.is_consistent:
                        # TODO search optimal metrics
                        order.append((
                            overlap.match_size / overlap.overlap_size,
                            overlap.positional_jaccard,
                            (window_begin, i),
                            overlap.match_size
                        ))
                    if save_metrics:
                        metrics['match_size'].append(overlap.match_size)
                        metrics['overlap_size'].append(overlap.overlap_size)
                        metrics['positional_jaccard'].append(overlap.positional_jaccard)
                        metrics['overlap_accuracy'].append(overlap.overlap_accuracy)
                else:
                    if save_metrics:
                        metrics['forbidden_hedge_pair'].append((window_begin, i))
            window_begin += 1
            pbar.update(1)
    return order, metrics


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


def algo_merge_hedge_contigs(
        hedges: List[HEdge],
        hedge_match_size_thresh: int = 5,
        hedge_jaccard_thresh: float = 0.5,
        master_match_size_thresh: int = 10,
        master_jaccard_thresh: float = 0.9,
        verbose: bool = False,
        debug: bool = False
) -> Tuple[List[HEdge], Mapping[str, List]]:
    metrics = defaultdict(list)

    # conflicting_hedge_indexes = find_conflicting_hedges(target_hedges)
    # print('Conflicting hedges count:', len(conflicting_hedge_indexes))

    # Sort hedges to speed up pairwise linkages evaluation
    hedges = sorted(hedges, key=lambda he: (he.begin(), he.end()))

    # Fix errors in reads
    # TODO

    # Evaluate all linkages
    hedges_linkages, linkage_metrics = get_hedges_merge_order_in_window(
        sorted_hedges=hedges,
        verbose=verbose,
        save_metrics=debug
    )
    metrics.update(linkage_metrics)

    # Merge linkages
    hedges_linkages.sort(reverse=True)

    masters = DSU(size=len(hedges))
    master_hedges: List[HEdge] = hedges[:]  # copy

    for overlap_size, positional_jaccard, (he1_idx, he2_idx), match_size \
            in tqdm(hedges_linkages, desc='Merging hedges', disable=not verbose):
        contig_1_idx = masters.find(he1_idx)
        contig_2_idx = masters.find(he2_idx)

        if contig_1_idx == contig_2_idx:
            continue

        master_hedge_1 = master_hedges[contig_1_idx]
        master_hedge_2 = master_hedges[contig_2_idx]

        master_hedges_inter = HEdge.intersect(master_hedge_1, master_hedge_2)
        master_hedges_overlap = HEdge.Overlap.metrics(master_hedge_1, master_hedge_2, master_hedges_inter)

        # Log master metrics
        if debug:
            metrics['master_overlap_size'].append(master_hedges_overlap.overlap_size)
            metrics['master_match_size'].append(master_hedges_overlap.match_size)
            metrics['master_positional_jaccard'].append(master_hedges_overlap.positional_jaccard)
            metrics['masters_are_consistent'].append(int(master_hedges_overlap.is_consistent))
            if master_hedges_overlap.overlap_size > 0:
                metrics['master_overlap_accuracy'].append(master_hedges_overlap.overlap_accuracy)

        # Merging condition
        consistent_merging = master_hedges_overlap.is_consistent
        hedges_are_close = (match_size >= hedge_match_size_thresh or
                            positional_jaccard >= hedge_jaccard_thresh)
        masters_are_close = (master_hedges_overlap.match_size >= master_match_size_thresh or
                             master_hedges_overlap.positional_jaccard >= master_jaccard_thresh)

        if consistent_merging and (hedges_are_close or masters_are_close):
            master_idx = masters.union(he1_idx, he2_idx)
            master_hedge = HEdge.union(master_hedge_1, master_hedge_2)
            master_hedges[master_idx] = master_hedge

    master_hedge_indexes = set(masters.find(i) for i in range(len(hedges)))
    master_hedges = [master_hedges[i] for i in master_hedge_indexes]
    print(f'Master hedges after merging: {len(master_hedges)}')
    haplo_hedges = master_hedges

    # Scaffolding
    # TODO
    # INF = int(1e10)
    # distance_threshold = int(1e9)
    # coverage_dist_matrix = eval_coverage_dist_matrix(master_hedges, INF=INF)
    # clustering = AgglomerativeClustering(
    #     n_clusters=None,
    #     affinity='precomputed',
    #     linkage='complete',
    #     distance_threshold=distance_threshold).fit(coverage_dist_matrix)
    # print(f'Contig clustesrs count: {len(set(clustering.labels_))}')

    # haplo_hedges_dict = defaultdict(list)
    # for label, hedge in zip(clustering.labels_, master_hedges):
    #     haplo_hedges_dict[label].append(hedge)

    # haplo_hedges = [
    #     reduce(HEdge.merge_with, hedges)
    #     for hedges in tqdm(haplo_hedges_dict.values(), desc='Merging clusters to single hedge')
    # ]
    # print(f'Contigs after clustering by freq: {len(haplo_hedges)}')

    return haplo_hedges, metrics
