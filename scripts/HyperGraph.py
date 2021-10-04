from copy import deepcopy
from typing import Dict

from scripts.HyperEdge.HyperEdge import HyperEdge
from scripts.HyperEdge.hyper_utils import get_distance_between_hedges, coverages_union, hyper_edges_union
from scipy.stats import binom

from scripts.algo import get_max_pair, check_leftovers_distribution


class HyperGraph:
    def __init__(self, hedges: Dict[frozenset, Dict[str, HyperEdge]]):
        self.hedges = hedges
        self.ever_created_hedges = deepcopy(hedges)

    def __getitem__(self, key):
        return self.hedges[key]

    def items(self):
        return self.hedges.items()

    def remove_leftovers(self, error_prob: float):
        for key in self.hedges:
            hedges_dict = self.hedges[key]
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
            self.hedges[key] = hedges_dict

    def remove_referencing_hyperedges(self):
        keys_to_delete = set()
        hedges_keys = list(self.hedges.keys())
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
            del self.hedges[key]

    def get_pairs(self):
        int_sets = list(k for k in self.hedges.keys() if len(self.hedges[k]) > 0)
        interesting_snp_sets = set(k for k in self.hedges.keys() if len(self.hedges[k]) > 0)
        found_any_new_conditions = False
        pairs = []
        for set1_i in range(len(int_sets)):
            set1 = int_sets[set1_i]
            for set2_i in range(set1_i + 1, len(interesting_snp_sets)):
                set2 = int_sets[set2_i]
                if min(max(set1), max(set2)) + 1 < max(min(set1), min(set2)):
                    continue
                if set1 == set2:
                    continue
                found_any_new_conditions = not set1.issubset(set2) and not set2.issubset(set1)
                # check_for_snps = not set1.isdisjoint(set2)
                for nucls1, hedge1 in self.hedges[set1].items():
                    for nucls2, hedge2 in self.hedges[set2].items():
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
                            new_hedge = coverages_union(hedge1, hedge2)
                            new_h_nucls = new_hedge.nucls
                            frozen_positions = frozenset(new_hedge.positions)
                            if frozen_positions not in self.ever_created_hedges or new_h_nucls not in \
                                    self.ever_created_hedges[
                                        frozen_positions]:
                                hedge1.used = False
                                hedge2.used = False
                                pairs.append((hedge1, hedge2))
        return pairs if found_any_new_conditions else []

    def merge_all_pairs(self, target_snp_count, error_probability, verbose):
        pairs = self.get_pairs()
        old_hedges = None
        haplo_hedges = []
        while len(pairs) > 0:
            print('Iteration started, pairs count: ', len(pairs))
            if verbose:
                print('Current hedges')
                for s, h in self.hedges.items():
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
                if frozen_positions not in self.ever_created_hedges or new_h_nucls not in self.ever_created_hedges[
                    frozen_positions]:
                    self.hedges[frozen_positions][new_h_nucls] = new_hedge
                    self.ever_created_hedges[frozen_positions][new_h_nucls] = new_hedge
                else:
                    self.remove_leftovers(error_probability)
                    pairs = self.get_pairs(self.ever_created_hedges)
                    continue
            pairs[index][0].weight *= freq1 * 100 / pairs[index][0].frequency
            pairs[index][1].weight *= freq2 * 100 / pairs[index][1].frequency
            pairs[index][0].frequency = freq1 * 100
            pairs[index][1].frequency = freq2 * 100
            # print(pairs[i])
            self[frozenset(pairs[index][0].positions)][he1.nucls] = pairs[index][0]
            self[frozenset(pairs[index][1].positions)][he2.nucls] = pairs[index][1]
            self.remove_leftovers(error_probability)
            pairs = self.get_pairs(self.ever_created_hedges)
        return haplo_hedges
