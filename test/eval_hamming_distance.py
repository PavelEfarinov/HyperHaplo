import numpy as np
from numpy import argmin

from scipy.stats import wasserstein_distance

from scripts.metrics import read_haplos_from_fasta, get_edist_matrix

if __name__ == "__main__":
    pred_file = '../data/hiv-lab-mix/itmo_final_0.95/itmo_final.fasta'
    ground_truth = '../data/hiv-lab-mix/ground_truth.fasta'
    region_start = 2253
    region_end = 3390

    predicted_haplos = read_haplos_from_fasta(pred_file)
    ground_truth_haplos = read_haplos_from_fasta(ground_truth)

    gt_setoff = [-2, 0, 7, 0, -2]
    for haplo, setoff in zip(ground_truth_haplos, gt_setoff):
        haplo.freq = 0.2
        haplo.nucls = haplo.nucls[region_start + setoff: region_end + setoff]
    edist_matrix, (gt_names, pred_names) = get_edist_matrix(ground_truth_haplos, predicted_haplos)
    weights_matrix = [[0 for _ in range(len(predicted_haplos))] for _ in range(len(ground_truth_haplos))]
    for i, gt_haplo in enumerate(ground_truth_haplos):
        for j, pr_haplo in enumerate(predicted_haplos):
            weights_matrix[i][j] = gt_haplo.freq * pr_haplo.freq
    weights_matrix = np.array(weights_matrix)
    print(edist_matrix)
    print('mean min edist', np.mean(edist_matrix.min(axis=0)))
    min_freq_pred = edist_matrix.min(axis=0)
    min_freq_gt = edist_matrix.min(axis=1)
    wasserstein = 0
    for i in range(len(ground_truth_haplos)):
        for j in range(len(predicted_haplos)):
            wasserstein += weights_matrix[i][j] * edist_matrix[i][j]
    print('wasserstein', wasserstein)

    delta = 12  # (region_end - region_end) / 100
    tp_rate = sum(0.2 for i in min_freq_gt if i < delta)
    fn_rate = 1 - tp_rate
    fp_rate = 0
    for j in range(len(weights_matrix[0])):
        min_dist = min(weights_matrix[:, j])
        if min_dist > delta:
            fp_rate += predicted_haplos[j].freq

    # print('precision', tp_rate/(tp_rate + fp_rate))
    # print('recall', tp_rate/(tp_rate + fn_rate))
