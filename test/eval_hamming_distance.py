from numpy import argmin

from scripts.metrics import read_haplos_from_fasta, get_edist_matrix

if __name__ == "__main__":
    pred_file = '../data/hiv-lab-mix/itmo_final/itmo_final.fasta'
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
    print(edist_matrix, gt_names, pred_names)
    for line in edist_matrix:
        print(min(line), predicted_haplos[argmin(line)].freq)

