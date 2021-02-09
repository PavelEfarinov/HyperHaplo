from pathlib import Path
import pandas as pd
import numpy as np
import click

from metrics import (
    get_edist_matrix,
    read_simulated_error_free_haplos,
    read_haplos_from_fasta
)


@click.command()
@click.option("--ne", type=int, help="Ne. Effective population size.")
@click.option("--replica", nargs=2, type=int, help="Range of target replicas. Both inclusive.")
@click.option("--run_name", type=str, help="Meaningful name of the run.")
@click.option("--tool", type=str, default='itmo', help="Tool name.")
@click.option("--predicted_fasta", type=str, default='itmo_final.fasta', help="Result file name predicted by tool.")
def main(ne, replica, run_name, tool, predicted_fasta):
    ROOT = Path('/scratch/rlebedev/shared')
    METRICS = ROOT / 'hamming_edists'
    LOG = ROOT / 'log'

    GT = ROOT / 'error_free_silulations_ground_truth'

    RESULT = ROOT / 'output' / tool / run_name / f'error_free_silulations_bam_Ne_{ne}'

    for file in RESULT.glob(f'u*_s2000_Ne{ne}'):
        params = file.name
        for rep in range(replica[0], replica[1] + 1):
            try:
                gt_file = GT / params / f'rep{rep}_count_haps.txt'
                pred_file = RESULT / params / f'sequences00{rep}' / predicted_fasta
                if not pred_file.exists():
                    continue

                gt_haplos = read_simulated_error_free_haplos(gt_file)
                pred_haplos = read_haplos_from_fasta(pred_file)

                edist_matrix, _ = get_edist_matrix(gt_haplos, pred_haplos)

                edist_list = []
                for i, gt_haplo in enumerate(gt_haplos):
                    for j, pred_haplo in enumerate(pred_haplos):
                        edist_list.append((
                            i + 1, j + 1,
                            gt_haplo.name, pred_haplo.name,
                            gt_haplo.freq, pred_haplo.freq,
                            edist_matrix[i][j]
                        ))

                pd.DataFrame(
                    edist_list,
                    columns=['gt_idx', 'pred_idx', 'gt_name', 'pred_name', 'gt_freq', 'pred_freq', 'edist']
                ).to_csv(METRICS / f'{run_name}_{params}_rep_{rep}.csv')
            except Exception as err:
                with open(LOG / f'{run_name}_{params}_rep_{rep}.log', 'a') as fout:
                    fout.write(str(err))


if __name__ == '__main__':
    main()
