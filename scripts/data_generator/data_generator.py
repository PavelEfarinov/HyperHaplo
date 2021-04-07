import os
import random
import time
from collections import defaultdict
from copy import copy

import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout
import pandas as pd


class Phylogeny:
    colors = {'insertion': 'r', 'double': 'g', 'deletion': 'b', 'alteration': 'y', 'recombination': 'm'}

    def __init__(self, reference, snp_count):
        self.graph = nx.DiGraph()
        # nx.set_node_attributes(self.graph, "alive", True)
        self.graph.add_node(reference, color='c')
        self.snp_positions = random.sample([i for i in range(len(reference))], snp_count)
        self.snp_positions.sort()

    def get_edges_count(self, node):
        degree = 0
        for edge in self.graph.edges:
            if edge[0] == node:
                degree += 1
        return degree

    def add_child(self, edge_from: list, seq, mutation_type):
        for node in edge_from:
            if self.get_edges_count(node) > 1:
                return
        if seq not in self.graph.nodes:
            self.graph.add_node(seq, alive=True, color='c')
        for edge in edge_from:
            self.graph.add_edge(edge, seq, color=self.colors[mutation_type])

    def die_out(self, seq):
        if seq in self.graph.nodes():
            self.graph[seq]['alive'] = False
            self.graph[seq].color = 'k'

    def choose_n_elements(self, n):
        alive_haplotypes = set()
        # print(self.graph.nodes())
        for haplo in self.graph.nodes():
            # if haplo['alive']:
            #     print(haplo)/\
            alive_haplotypes.add(haplo)
        chosen_haplotypes = set()
        while len(chosen_haplotypes) < n:
            chosen = random.choice(list(alive_haplotypes))
            if chosen not in chosen_haplotypes:
                chosen_haplotypes.add(chosen)
        return list(chosen_haplotypes)

    def plot(self):
        plt.figure(figsize=(20, 20))
        pos = graphviz_layout(self.graph, prog="dot")
        edge_colors = [self.graph[u][v]['color'] for u, v in self.graph.edges()]
        node_colors = [('b' if v not in self.important_nodes else 'r') for v in self.graph.nodes()]
        for v in node_colors:
            print(v)
        nx.draw(self.graph, pos, edge_color=edge_colors, node_color=node_colors)

    def mark_nodes_as_important(self, nodes_list):
        self.important_nodes = nodes_list


def insertion(genome, snps):
    # insertion_size = poisson.rvs(len(genome) // 100, size=1)[0]
    insertion_size = 1
    insertion_index = random.randint(0, len(genome) - insertion_size)
    genome = genome[:insertion_index] + ''.join([random.choice(['A', 'C', 'T', 'G']) for i in range(insertion_size)]) + \
             genome[insertion_index:]
    return genome


def double(genome, snps):
    insertion_index = random.randint(0, len(genome) - 1)
    genome = genome[:insertion_index] + genome[insertion_index] + genome[insertion_index:]
    return genome


def deletion(genome, snps):
    deletion_size = 1
    # deletion_size = poisson.rvs(len(genome) / 100, size=1)[0]
    deletion_index = random.randint(0, len(genome) - deletion_size)
    genome = genome[:deletion_index] + genome[deletion_index + deletion_size:]
    return genome


def alteration(genome, snps):
    complementary_bases = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    alteration_index = random.choice(snps)
    base = genome[alteration_index]
    available = ['A', 'C', 'T', 'G']
    available = [x for x in available if x != base]
    available.append(complementary_bases[base])
    changed = random.choice(available)
    genome = genome[:alteration_index] + changed + genome[alteration_index + 1:]
    return genome


def recombination(genome_1, genome_2):
    genome_1_percent = random.randint(0, 100)
    resulting_genome = genome_1[:len(genome_1) * genome_1_percent // 100] + genome_2[
                                                                            len(genome_2) * genome_1_percent // 100:]
    return resulting_genome


class DataGenerator:
    mutation_types = {'insertion': insertion, 'double': double, 'deletion': deletion, 'alteration': alteration,
                      'recombination': recombination}
    mutation_probs = {'insertion': 0, 'double': 0, 'deletion': 0, 'alteration': 0.6, 'recombination': 0.4}

    def __init__(self, reference, mutation_prob=0.01, snp_count=5):
        self.mutation_prob = mutation_prob
        self.phylogeny_tree = Phylogeny(reference, snp_count)

    def mutate(self, max_elapsed_time=10, max_phylogeny_size=100):
        start_time = time.time()
        epochs_count = 0
        while time.time() - start_time < max_elapsed_time and len(self.phylogeny_tree.graph.nodes) < max_phylogeny_size:
            epochs_count += 1
            if random.random() < self.mutation_prob:
                mutation_type = \
                    np.random.choice(list(self.mutation_probs.keys()), 1, p=list(self.mutation_probs.values()))[0]
                mutation_type = 'alteration'
                if mutation_type != 'recombination':
                    parents_list = self.phylogeny_tree.choose_n_elements(1)
                elif len(self.phylogeny_tree.graph.nodes) > 1 and random.random() < 0.5:
                    parents_list = self.phylogeny_tree.choose_n_elements(2)
                else:
                    continue
                # print(parents_list)
                generated_genome = self.mutation_types[mutation_type](*parents_list, self.phylogeny_tree.snp_positions)
                self.phylogeny_tree.add_child(parents_list, generated_genome, mutation_type)
            # if epochs_count % 100 == 0:
                # print('Processed', epochs_count, 'epochs')

    def get_haplotypes(self, haplotypes_count):
        chosen_haplotypes = self.phylogeny_tree.choose_n_elements(haplotypes_count)
        return chosen_haplotypes, self.phylogeny_tree.snp_positions

    def generate_haplotypes_by_snp_count(self, reference, snp_count, haplo_count):
        snp_positions = random.sample([i for i in range(len(reference))], snp_count)
        snp_positions.sort()
        haplos = []
        for _ in range(haplo_count):
            haplo = copy(reference)
            for i in range(snp_count):
                pos = snp_positions[i]
                haplo = haplo[:pos] + random.choice('ACTG') + haplo[pos + 1:]
            haplos.append(haplo)
        return haplos, snp_positions

def generate_probabilities(n):
    random_numbers = [0] + [random.random() for _ in range(n - 1)] + [1]
    random_numbers.sort()
    probabilities = [random_numbers[i] - random_numbers[i - 1] for i in range(1, n + 1)]
    assert sum(probabilities) == 1
    return probabilities


if __name__ == "__main__":
    reference_name = 'sequence'
    haplo_count = 25
    snp_count = 50
    with open(f'../../data/fasta_sequences/{reference_name}.fasta') as inf:
        lines = inf.readlines()
        reference = ''.join(l.strip() for l in lines[1:])
    print(f'Reference length is {len(reference)}')
    data_generator = DataGenerator(reference, snp_count=snp_count)
    data_generator.mutate(max_elapsed_time=1, max_phylogeny_size=10000)
    print('finished mutating')

    chosen_haplotypes, snps = data_generator.get_haplotypes(haplotypes_count=haplo_count - 1)
    haplo_probabilities = generate_probabilities(haplo_count)
    haplo_probabilities.sort(reverse=True)

    data_generator.phylogeny_tree.mark_nodes_as_important(chosen_haplotypes)

    fasta = ''.join(lines) + '\n'
    dir = f'../../generated_haplos/{reference_name}_haplo_{haplo_count}_snp_{snp_count}/'
    if not os.path.exists(os.path.dirname(dir)):
        try:
            os.makedirs(os.path.dirname(dir))
        except:
            raise IOError('Could not create directory ' + dir)
    for i, haplo in enumerate(chosen_haplotypes):
        fasta += '>' + f'haplo{i}' + '\n'
        fasta += haplo + '\n'
    with open(f'../../generated_haplos/{reference_name}_haplo_{haplo_count}_snp_{snp_count}/sequences.fasta', 'w') as ouf:
        ouf.write(fasta)

    print(snps)
    snps_to_freq = str([reference[i] for i in snps]) + '\t' + str(haplo_probabilities[0]) + '\n'
    for haplo, prob in zip(chosen_haplotypes, haplo_probabilities[1:]):
        snps_to_freq += str([haplo[i] for i in snps]) + '\t' + str(prob) + '\n'
    with open(f'../../generated_haplos/{reference_name}_haplo_{haplo_count}_snp_{snp_count}/snps_to_freq.txt', 'w') as ouf:
        ouf.write(snps_to_freq)

    abundance = lines[0][1:].strip() + '\t' + str(haplo_probabilities[0]) + '\n'
    for haplo_index, prob in zip(range(haplo_count - 1), haplo_probabilities[1:]):
        abundance += f'haplo{haplo_index}' + '\t' + str(prob) + '\n'
    with open(f'../../generated_haplos/{reference_name}_haplo_{haplo_count}_snp_{snp_count}/abundance.txt', 'w') as ouf:
        ouf.write(abundance)
