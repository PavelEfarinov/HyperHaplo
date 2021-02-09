import pysam
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from typing import List, Dict
from pathlib import Path
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from more_itertools import pairwise

from data import SNP, Haplotype, get_snp_indeces
from hedge import HEdge, SingleHEdge, PairedHEdge


def reads_stats(bamfile_path: Path):
    with pysam.AlignmentFile(bamfile_path, 'rb') as bf:
        total_cnt = 0
        unmapped_cnt = 0
        proper_pair_cnt = 0
        paired = 0
        fst = 0
        snd = 0

        names = []
        read: pysam.AlignedSegment
        for read in tqdm(bf.fetch(), desc='Load reads'):
            names.append(read.query_name)
            total_cnt += 1

            if read.is_unmapped:
                unmapped_cnt += 1
                continue

            if read.is_paired:
                paired += 1

            if read.is_proper_pair:
                proper_pair_cnt += 1

            if read.is_read1:
                fst += 1
            elif read.is_read2:
                snd += 1

        print(f'Total reads : {total_cnt}')
        print(f'Unmapped    : {unmapped_cnt}')
        print(f'Paired      : {paired} ({total_cnt - unmapped_cnt})')
        print(f'Proper pair : {proper_pair_cnt}')
        print(f'Fst read    : {fst}')
        print(f'Snd read    : {snd}')
        print(f'Names:', Counter(Counter(names).values()))


def paired_reads_stats(paired_reads: List[List[pysam.AlignedSegment]]):
    isize = []
    overlap_size = []
    fst_in_snd = 0
    snd_in_fst = 0
    fst_then_snd = []
    snd_then_fst = []
    r1_len = []
    r2_len = []

    paired_reads = [parts for parts in paired_reads if len(parts) == 2]

    def is_contains(seg1_pos, seg2_pos):
        return seg1_pos[0] <= seg2_pos[0] and seg1_pos[-1] >= seg2_pos[-1]

    r1: pysam.AlignedSegment
    r2: pysam.AlignedSegment
    for r1, r2 in tqdm(paired_reads, desc='Process reads'):
        if r1.is_read2:
            r1, r2 = r2, r1
        p1 = r1.get_reference_positions()
        p2 = r2.get_reference_positions()

        r1_len.append(len(p1))
        r2_len.append(len(p2))

        isize.append(max(p1[-1], p2[-1]) - min(p1[0], p2[0]))

        if is_contains(p1, p2):
            snd_in_fst += 1
        elif is_contains(p2, p1):
            fst_in_snd += 1

        if p1[-1] < p2[0]:
            fst_then_snd.append(p2[0] - p1[-1])
        elif p2[-1] < p1[0]:
            snd_then_fst.append(p1[0] - p2[-1])
        else:
            overlap_size.append(len(set(p1) & set(p2)))

    print(f'1 in 2      : {fst_in_snd}')
    print(f'2 in 1      : {snd_in_fst}')
    print(f'1 then 2    : {len(fst_then_snd)}')
    print(f'2 then 1    : {len(snd_then_fst)}')
    print(f'1 overlap 2 : {len(overlap_size)}')

    plt.bar(*zip(*Counter(fst_then_snd).items()))
    plt.title('1 then 2')
    plt.xlabel('skip size')
    plt.ylabel('count')
    plt.show()

    plt.bar(*zip(*Counter(snd_then_fst).items()))
    plt.title('2 then 1')
    plt.xlabel('skip size')
    plt.ylabel('count')
    plt.show()

    plt.hist(overlap_size, bins=100)
    plt.title('overlap size')
    plt.show()

    plt.hist(isize, bins=100)
    plt.title('insertion size')
    plt.show()

    plt.hist([r1_len, r2_len], bins=100, color=['red', 'blue'])
    plt.title('read length')
    plt.legend(labels=['read1', 'read2'])
    plt.show()


def _plot_variable_regions(xs, ys, xlim):
    print(f'Всего: {len(xs)}')
    plt.figure(figsize=(16, 2))
    # plt.title('Распределение SNP')
    plt.xlabel('Позиция в геноме')
    plt.xlim([0, xlim])
    plt.yticks([])
    plt.scatter(xs, ys, s=3)
    plt.show()


def plot_snps_distr(snps: List[SNP], target_snps: List[SNP]):
    xs = np.array([snp.position for snp in target_snps])
    ys = np.ones(xs.shape)
    _plot_variable_regions(xs, ys, xlim=len(snps))


def plot_variable_position_in_haplos(haplos: List[Haplotype]) -> List[int]:
    assert len(haplos) > 0
    assert len(set(len(h.nucls) for h in haplos)) == 1

    genome_len = len(haplos[0].nucls)
    xs = []
    for i in range(genome_len):
        nucls = set()
        for haplo in haplos:
            nucls.add(haplo.nucls[i])
        if len(nucls) > 1:
            xs.append(i)

    xs = np.array(xs)
    ys = np.ones(xs.shape)
    _plot_variable_regions(xs, ys, xlim=genome_len)
    return list(xs)


def plot_snp_sensitivity(snps: List[SNP], lower=0, upper=0.01, num=100):
    thresholds = np.linspace(lower, upper, num=num)
    snps_count = [
        sum(1 for snp in snps if snp.minor_coverage >= min_cover)
        for min_cover in thresholds
    ]

    # plt.plot(thresholds, snps_count)
    plt.scatter(thresholds, snps_count, s=10)
    # plt.title('Чувстительность SNP')
    plt.ylabel('Количество SNP')
    plt.xlabel('Минимальное минорное покрытие $MC$')
    plt.show()


def plot_snps_coverage(minors, title: str, head: int = None):
    minors.value_counts().sort_index()[:head].plot.bar()
    plt.title(title.upper())
    plt.xlabel('Величина покрытия конкретной позиции')
    plt.ylabel('Количество позиций')
    plt.show()


def plot_snv_count_by_coverage_threshold(indexed_snp_positions, lower=1, upper=100):
    coverage_threshs = list(range(lower, upper))
    snp_counts = [len(get_snp_indeces(indexed_snp_positions, thresh)) for thresh in coverage_threshs]

    plt.figure(figsize=(16, 4))
    plt.xlabel('Порог покрытия snp')
    plt.ylabel('Количество snp')
    pd.Series(data=snp_counts, index=coverage_threshs).plot.bar()
    plt.show()


def plot_hedges_type_distr(hedges: List[HEdge]):
    edges_counter = Counter([type(he).__name__ for he in hedges])
    names, counts = zip(*edges_counter.items())

    for name, count in zip(names, counts):
        print(f'{name} reads: {count}')

    x = list(range(len(names)))
    plt.bar(x, counts)
    plt.xticks(x, names)
    # plt.yscale('log')
    plt.show()


def plot_hedges_snp_count_distr(hedges: List[HEdge], lower=None, upper=None):
    hedges_sizes = [he.snp_count for he in hedges]

    # ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    # ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    pd.Series(hedges_sizes).value_counts().sort_index()[lower:upper].plot.bar(ax=ax1)
    ax1.set_title('Распределение размеров (количество SNP) hedges')
    ax1.set_ylabel('Количество')
    pd.Series(hedges_sizes).sort_index().plot.hist(bins=100, ax=ax2)
    plt.xlabel('Размер hedge')
    plt.show()


def plot_hedges_weight_distr(hedges: List[HEdge], lower=None, upper=None):
    if lower and lower > len(hedges):
        print(f'No hedges: [{lower}, {upper}]')
        return

    lower = lower - 1 if lower is not None else 0
    hedges_weights = [he.weight for he in hedges]

    plt.figure(figsize=(16, 4))
    pd.Series(hedges_weights).value_counts().sort_index()[lower:upper].plot.bar()
    plt.xlabel('Вес гиперребра')
    plt.ylabel('Количество')
    plt.show()


def plot_hedges_length_distr(
        hedges: List[HEdge],
        split_paired_hedges=False,
        lower=None,
        upper=None
):
    hedges_lengths = []
    for he in hedges:
        if not split_paired_hedges or isinstance(he, SingleHEdge):
            hedges_lengths.append(len(he))
        elif split_paired_hedges and isinstance(he, PairedHEdge):
            for l, r in [he.first_segment_in_genome(), he.second_segment_in_genome()]:
                hedges_lengths.append(r - l + 1)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    pd.Series(hedges_lengths).value_counts().sort_index()[lower:upper].plot.bar(ax=ax1)
    ax1.set_title('Распределение длины гиперребер')
    ax1.set_ylabel('Количество')
    pd.Series(hedges_lengths).sort_index()[lower:upper].plot.hist(bins=100, ax=ax2)
    plt.xlabel('Длина покрытия гиперребра')
    plt.show()


def plot_hedges_snp_count_and_weight_ratio(hedges: List[HEdge], weight_thresh_lower=20, weight_thresh_upper=None):
    if weight_thresh_upper is None:
        weight_thresh_upper = max(he.weight for he in hedges)

    sizes = [he.snp_count for he in hedges if weight_thresh_lower <= he.weight <= weight_thresh_upper]
    weights = [he.weight for he in hedges if weight_thresh_lower <= he.weight <= weight_thresh_upper]

    fig = plt.figure(figsize=(16, 8))
    sns.scatterplot(x=sizes, y=weights)
    plt.title(f'Hedges.weight >= {weight_thresh_lower}')
    plt.xlabel('Size')
    plt.ylabel('Weight')
    plt.show()


def plot_coverage(
        hedges: List[HEdge],
        reference_len: int,
        weight_thresh=1,
        split_paired_hedges=False,
        save=False,
        save_path='output/cover_he_{weight_thresh}.png'
):
    coverage = [0 for _ in range(reference_len)]
    hedges = [he for he in hedges if he.weight >= weight_thresh]

    for he in hedges:
        if not split_paired_hedges or isinstance(he, SingleHEdge):
            coverage[he.begin_in_genome()] += 1
            if he.end_in_genome() + 1 < reference_len:
                # if hedges ends at the end of the reference then there is no `end` dec position.
                # So after coverage evaluation acc may be more than 0.
                coverage[he.end_in_genome() + 1] -= 1
        elif split_paired_hedges and isinstance(he, PairedHEdge):
            for l, r in [he.first_segment_in_genome(), he.second_segment_in_genome()]:
                coverage[l] += 1
                if r + 1 < reference_len:
                    coverage[r + 1] -= 1

    acc = 0
    for i in range(len(coverage)):
        acc += coverage[i]
        coverage[i] = acc

    print(f'Used hedges: {len(hedges)}')

    fig = plt.figure(figsize=(16, 4))
    plt.title(f'Покрытие гиперребрами веса {weight_thresh}')
    plt.plot(list(range(len(coverage))), coverage)
    plt.show()

    if save:
        fig.savefig(save_path.format(weight_thresh=weight_thresh), dpi=fig.dpi)


class HyperGraphViz:
    def __init__(self):
        self.segments = defaultdict(lambda: {
            'w': 0,
            'count': 0,
            'alleles': defaultdict(lambda: {'w': 0, 'count': 0})
        })
        self.nucl_coverage = defaultdict(int)
        self.yticks = ' ATCGN'
        self.linewidth = 15
        self.linewidths = 15

    def _add_nucl_cover(self, nucl_idx, nucl_letter, weight):
        self.nucl_coverage[(nucl_idx, nucl_letter)] += weight
        self.nucl_coverage[(nucl_idx, '*')] += weight

    def _add_segment(self, segment, weight):
        self.segments[segment]['count'] += 1
        self.segments[segment]['w'] += weight

    def _add_segment_alleles(self, segment, segment_alleles, weight):
        self.segments[segment]['alleles'][segment_alleles]['count'] += 1
        self.segments[segment]['alleles'][segment_alleles]['w'] += weight

    def add_hedge(self, hedge: HEdge, w_threshold=1):
        weight = hedge.weight
        # TODO support of SingleHEdge and PairedHEdges
        snv_alleles = list(zip(hedge.positions, hedge.nucls))

        assert len(snv_alleles), 'Empty hedge!!!'
        if weight >= w_threshold:
            snv_alleles = sorted(snv_alleles)

            for idx, nucl in snv_alleles:
                self._add_nucl_cover(idx, nucl, weight)

            for (li, ln), (ri, rn) in pairwise(snv_alleles):
                segment = (li, ri)
                self._add_segment(segment, weight)

                segment_alleles = (ln, rn)
                self._add_segment_alleles(segment, segment_alleles, weight)

    def add_all_hedges(self, hedges, w_threshold=1):
        for hedge in tqdm(hedges):
            self.add_hedge(hedge, w_threshold)

    def _nucl(self, nucl_letter):
        return self.yticks.find(nucl_letter)

    def _get_snv_color(self, rel_cov):
        breakpoints = [0.01, 0.05, 0.1, 0.3]
        colors = 'kbgyr'
        return colors[bisect_right(breakpoints, rel_cov)]

    def _get_hedge_color(self, rel_cov):
        breakpoints = [0.01, 0.05, 0.1, 0.3]
        colors = 'kbgyr'
        return colors[bisect_right(breakpoints, rel_cov)]

    def plot(self, l, r, figsize=(16, 12), verbose=False, save=False):
        # Print canvas
        fig = plt.figure(figsize=figsize)
        plt.xlim([l - 2, r + 2])
        plt.ylim([0, 6])
        plt.yticks(range(6), self.yticks)

        # Print segments on target interval
        segments = sorted(self.segments.items())
        start_index = bisect_left([li for (li, _), _ in segments], l)
        for (li, ri), segment_alleles in tqdm(segments[start_index:], disable=not verbose):
            if ri > r:
                break

            # Print HEdge`s
            for (lN, rN), meta in segment_alleles['alleles'].items():
                rel_cov = meta['w'] / segment_alleles['w']
                if rel_cov >= 0.005:
                    plt.plot(
                        [li, ri],
                        [self._nucl(lN), self._nucl(rN)],
                        c=self._get_hedge_color(rel_cov),
                        ls='-',
                        linewidth=self.linewidth * rel_cov
                    )

            # Print SNV`s
            for nucl_letter in self.yticks:
                l_key = (li, nucl_letter)
                if l_key in self.nucl_coverage:
                    l_rel_cov = self.nucl_coverage[l_key] / self.nucl_coverage[(li, '*')]
                    if l_rel_cov >= 0.01:
                        plt.scatter(
                            [li],
                            [self._nucl(nucl_letter)],
                            c=[self._get_snv_color(l_rel_cov)],
                            linewidths=self.linewidths * l_rel_cov
                        )

                if ri == r:
                    r_key = (ri, nucl_letter)
                    if r_key in self.nucl_coverage:
                        r_rel_cov = self.nucl_coverage[r_key] / self.nucl_coverage[(ri, '*')]
                        if r_rel_cov >= 0.01:
                            plt.scatter(
                                [ri],
                                [self._nucl(nucl_letter)],
                                c=[self._get_snv_color(r_rel_cov)],
                                linewidths=self.linewidths * r_rel_cov
                            )
        plt.show()
        if save:
            fig.savefig(f'output/gen_{l:04}-{r:04}.png', dpi=fig.dpi)

    def print_segments(self):
        for s in self.segments.items():
            print(s)
