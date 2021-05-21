from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import itertools
from itertools import product
from collections import Counter, defaultdict
from tqdm import tqdm
import pysam
from operator import eq
from scripts.data import Haplotype


def get_nucls_counter_from_haplos(haplos: List[Haplotype]):
    return Counter(itertools.chain.from_iterable([h.nucls for h in haplos]))


def save_haplos(haplos: List[Haplotype], fasta_path: Path):
    with open(fasta_path, 'w') as fout:
        for haplo in haplos:
            haplo_record = f'>{haplo.name}_{haplo.freq}\n{haplo.nucls}\n'
            fout.write(haplo_record)


def normalize_freq(values: List[Union[int, float]]) -> List[float]:
    total = sum(values)
    return [value / total for value in values]


def read_simulated_error_free_haplos(path_to_file: Path) -> List[Haplotype]:
    haplo_nucls, counts = [], []
    with open(path_to_file, 'r') as fin:
        for line in fin:
            haplo_nucl_str, count = line.strip().split()
            haplo_nucls.append(haplo_nucl_str)
            counts.append(int(count))

    freqs = normalize_freq(counts)
    haplos = [
        Haplotype(name=f'gt_{i}', nucls=nucls, freq=freq)
        for i, (nucls, freq) in enumerate(zip(haplo_nucls, freqs), start=1)
    ]
    return haplos


def read_haplos_from_fasta(
        fasta_file: Path,
        template_name: str = 'hap',
        normalize: bool = True
) -> List[Haplotype]:
    freqs = []
    haplo_nucls = []
    with pysam.FastaFile(fasta_file) as rf:
        for name in rf.references:
            haplo_nucls.append(rf.fetch(name))
            try:
                freqs.append(float(name.split('_')[-1]))
            except Exception as e:
                freqs.append(0)


    if normalize:
        freqs = normalize_freq(freqs)

    haplos = [
        Haplotype(name=f'{template_name}_{i}', nucls=nucls, freq=freq)
        for i, (nucls, freq) in enumerate(zip(haplo_nucls, freqs), start=1)
    ]
    return haplos


def read_clique(fasta_file: Path, normalize: bool = True) -> List[Haplotype]:
    return read_haplos_from_fasta(fasta_file, template_name='cliq', normalize=normalize)


def read_clique_gag() -> List[Haplotype]:
    fasta_file = Path('runs/clique-snv-1.5.3_gag/mix5_gag.fasta')
    return read_clique(fasta_file)


# ===================== VIRUS MIX 5 =====================
def read_mix5(region_start: int, region_end: int) -> List[Haplotype]:
    haplos = []
    fasta_file = Path('data') / 'mix5_gag' / '5VirusMixReference.fasta'
    with pysam.FastaFile(fasta_file) as rf:
        for name in rf.references:
            nucls = rf.fetch(region=f'{name}:{region_start}-{region_end}')
            haplos.append(Haplotype(name, nucls))
    return haplos


def read_mix5_gag() -> List[Haplotype]:
    return read_mix5(region_start=336, region_end=1838)


def hamming_dist(seq1, seq2) -> int:
    mismathces = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
    return mismathces


def get_edist_matrix(
        gt_haplos: List[Haplotype],
        pred_haplos: List[Haplotype],
        dist_func=hamming_dist,
        verbose=False
) -> Tuple[np.ndarray, Tuple[List[str]], List[str]]:
    with_itself = len(gt_haplos) == len(pred_haplos) and all(map(eq, gt_haplos, pred_haplos))
    if with_itself:
        print(f'Eval edist matrix with itself')

    distance_matrix = np.zeros((len(gt_haplos), len(pred_haplos)), dtype=int)
    for i, gt_haplo in enumerate(tqdm(gt_haplos, desc='GT', disable=not verbose)):
        for j, pred_haplo in enumerate(pred_haplos):
            if not with_itself:
                distance_matrix[i][j] = dist_func(gt_haplo.nucls, pred_haplo.nucls)
            else:
                if i < j:
                    distance_matrix[i][j] = distance_matrix[j][i] = dist_func(gt_haplo.nucls, pred_haplo.nucls)

    gt_names = [haplo.name for haplo in gt_haplos]
    pred_names = [haplo.name for haplo in pred_haplos]
    names = (gt_names, pred_names)
    return distance_matrix, names


def plot_edist_matrix(
        distance_matrix: np.ndarray,
        row_names: List[str],
        col_names: List[str],
        figsize=(18, 8)
):
    plt.figure(figsize=figsize)
    sns.heatmap(distance_matrix,
                annot=True,
                xticklabels=col_names,
                yticklabels=row_names,
                linewidths=.5,
                fmt="d",
                cmap='YlOrRd',
                square=len(row_names) == len(col_names)
                )
    plt.yticks(rotation=0)
    plt.show()


def get_closest_haplo(
        haplo: Haplotype,
        others: List[Haplotype],
        dist_func=hamming_dist
) -> Tuple[Haplotype, int]:
    min_dist = len(haplo.nucls)
    closest_haplo = None
    for other_haplo in others:
        dist = dist_func(haplo.nucls, other_haplo.nucls)
        if dist < min_dist:
            min_dist = dist
            closest_haplo = other_haplo
    return closest_haplo, min_dist


def pairwise_dataset_dist(
        true_name: str,
        true_haplos: List[Haplotype],
        pred_name: str,
        pred_haplos: List[Haplotype]
):
    print(f'{true_name} by {pred_name}')
    left_by_right_set_dist(true_haplos, pred_haplos)
    print()
    print(f'{pred_name} by {true_name}')
    left_by_right_set_dist(pred_haplos, true_haplos)


def left_by_right_set_dist(left: List[Haplotype], right: List[Haplotype]):
    for haplo in left:
        closest_haplo, dist = get_closest_haplo(haplo, right)
        print(f'[{haplo.name:>5}] <- [{closest_haplo.name:>5}]: {dist}')


def set_to_set_distance_dict(
        left: List[Haplotype],
        right: List[Haplotype]
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    left_to_right = []
    for haplo in left:
        closest_haplo, min_dist = get_closest_haplo(haplo, right)
        left_to_right.append((min_dist, haplo.freq))

    right_to_left = []
    for haplo in right:
        closest_haplo, min_dist = get_closest_haplo(haplo, left)
        right_to_left.append((min_dist, haplo.freq))

    return left_to_right, right_to_left


def _precision_and_recall(freq_tp: float, freq_fp: float) -> Tuple[float, float]:
    precision = freq_tp / (freq_tp + freq_fp)
    recall = freq_tp  # == freq_tp / (freq_tp + (1  - freq_tp))
    return precision, recall


def eval_precision_and_recall(
        gt_haplos: List[Haplotype],
        pred_haplos: List[Haplotype],
        dist_threshold: int = 12
) -> Tuple[float, float]:
    ltr, rtl = set_to_set_distance_dict(gt_haplos, pred_haplos)

    freq_tp = sum([freq for dist, freq in ltr if dist <= dist_threshold])
    freq_fp = sum([freq for dist, freq in rtl if dist > dist_threshold])

    precision, recall = _precision_and_recall(freq_tp, freq_fp)
    return precision, recall


def eval_precision_and_recall_from_csv_dist_matrix(
        file_path: Path,
        dist_threshold: int = 12
) -> Tuple[float, float]:
    df = pd.read_csv(file_path, index_col=0)

    ltr = df.groupby('gt_idx').min()
    rtl = df.groupby('pred_idx').min()

    freq_tp = ltr[ltr.edist <= dist_threshold]['gt_freq'].sum()
    freq_fp = rtl[rtl.edist > dist_threshold]['pred_freq'].sum()

    precision, recall = _precision_and_recall(freq_tp, freq_fp)
    return precision, recall


@ticker.FuncFormatter
def major_formatter(x, pos):
    return '1e' + str(int(np.log10(x)))


def get_mutation_rates() -> List[str]:
    return [
        '1e-3', '1e-4', '1e-5',
        '1e-6', '1e-7', '1e-8',
        '3e-4', '3e-5', '3e-8',
        '5e-3', '5e-4', '5e-5',
        '5e-6', '5e-7', '5e-8'
    ]


def get_plot_styles():
    return {
        'CliqueSNV_fast': ('b', 'x'),
        'CliqueSNV': ('m', 'x'),
        'ITMO': ('r', 'D'),
        'PredictHaplo': ('g', '^'),
        'aBayesQR': ('y', 'v'),
    }


def eval_and_save_precision_and_recall(template_name: str, output_path: Path):
    data = []

    mutation_rates = get_mutation_rates()
    Ne = [500, 1000, 2500]
    replicas = [1, 2, 3, 4, 5]

    params = list(product(mutation_rates, Ne, replicas))
    for mu, ne, rep in tqdm(params):
        file_path = Path(template_name.format(mu=mu, ne=ne, rep=rep))
        if not file_path.exists():
            print(f'Skip: {file_path}')
            data.append((float(mu), ne, rep, '', ''))
        else:
            precision, recall = eval_precision_and_recall_from_csv_dist_matrix(file_path)
            data.append((float(mu), ne, rep, precision, recall))

    pd.DataFrame(
        data,
        columns=['mu', 'Ne', 'rep', 'precision', 'recall']
    ).to_csv(output_path)


def plot_precision_and_recall_ax(named_dfs, Ne=500, mean=False):
    styles = get_plot_styles()

    def plot_metric(ax, metric_name):
        names = []
        for name, df in named_dfs:
            names.append(name)
            df = df[df['Ne'] == Ne]
            if mean:
                res = df.groupby('mu')[metric_name].mean()
                mu_rates = res.index.values
                metric = res.values
            else:
                mu_rates = df['mu'].values
                metric = df[metric_name].values
            ax.plot(mu_rates, metric,
                    c=styles[name][0],
                    marker=styles[name][1],
                    ms=6,
                    mew=3,
                    linestyle='-',
                    linewidth=2)

        # ax.set_title(f'Ne = {Ne}')
        ax.set_ylabel(metric_name)
        ax.set_xlabel('Скорость мутации')
        ax.set_xscale('log')
        ax.legend(names)
        ax.axvspan(3e-5, 1e-4, color='blue', alpha=0.1)
        ax.axvspan(2.5e-5, 1.2e-4, color='red', alpha=0.1)
        ax.xaxis.set_major_formatter(major_formatter)

    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(211)
    ax1.set_title(f'Ne = {Ne}')
    plot_metric(ax1, 'precision')

    ax2 = fig.add_subplot(212)
    # ax2.set_xlabel('Скорость мутации')
    plot_metric(ax2, 'recall')
    plt.show()


def plot_emd_ax(named_dfs, Ne=500, mean=False):
    styles = get_plot_styles()

    names = []
    fig = plt.figure(figsize=(10, 4))
    # fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    for name, df in named_dfs:
        names.append(name)
        df = df[df['Ne'] == Ne]
        if mean:
            res = df.groupby('mu')['emd'].mean()
            mu_rates = res.index.values
            emds = res.values
        else:
            mu_rates = df['mu'].values
            emds = df['emd'].values

        ax.loglog(mu_rates, emds,
                  c=styles[name][0],
                  marker=styles[name][1],
                  ms=6,
                  mew=3,
                  linestyle='-',
                  linewidth=2)

    ax.set_title(f'Ne = {Ne}')
    ax.set_ylabel('EMD')
    ax.set_xlabel('Скорость мутации')
    ax.legend(names)
    ax.axvspan(3e-5, 1e-4, color='blue', alpha=0.1)
    ax.axvspan(2.5e-5, 1.2e-4, color='red', alpha=0.1)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.yaxis.set_major_formatter(major_formatter)
    plt.show()
