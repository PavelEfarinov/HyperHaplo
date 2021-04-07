import time
import click

from scripts.data import *
from scripts.core import *
from scripts.algo import *
from scripts.metrics import *


@click.command()
@click.option('-bam', '--bamfile_path', type=str, help="Input bam file.")
@click.option('-ref', '--reference_length', type=str, help="Reference length.")
@click.option('-err', '--error_probability', type=str, default=0, help="Error probability.")
@click.option('-bq', '--min_base_quality', type=int, default=0, help="Minimal base quality for position in SNP.")
@click.option('-mq', '--min_mapping_quality', type=int, default=0, help="Minimal mapping quality for a read.")
@click.option('-mc', '--min_minor_coverage', type=float, default=0.05, help="Minimal minor coverage to call SNP.")
@click.option('-w', '--target_hedge_weight', type=int, default=3, help="Minimal weight for hedges.")
@click.option('-cl', '--min_contig_length', type=int, default=0, help="Minimal length for contig to infer haplotype.")
@click.option('-rl', '--min_relative_contig_length', type=float, default=0.0,
              help="Minimal relative length of a contig to infer haplotype.")
@click.option('-rs', '--region_start', type=int, default=None, help="Start of target region in the reference.")
@click.option('-re', '--region_end', type=int, default=None, help="End of target region in the reference.")
@click.option('-out', '--output_folder', type=str, default='.', help="Output folder to store results.")
@click.option('-log', '--verbose', is_flag=True, help="Print tqdm status bars in console.")
@click.option('-hos', '--hedge_match_size', type=int, default=5, help="Metrics.")
@click.option('-mos', '--master_match_size', type=int, default=10, help="Metrics.")
@click.option('-hj', '--hedge_jaccard', type=float, default=0.5, help="Metrics.")
@click.option('-mj', '--master_jaccard', type=float, default=0.9, help="Metrics.")
def main(
        bamfile_path: str,
        reference_length: int,
        error_probability: float,
        min_base_quality: int,
        min_mapping_quality: int,
        min_minor_coverage: float,
        target_hedge_weight: int,
        min_contig_length: int,
        min_relative_contig_length: float,
        region_start: int,
        region_end: int,
        output_folder: str,
        verbose: bool,
        hedge_match_size: int,
        master_match_size: int,
        hedge_jaccard: float,
        master_jaccard: float
):
    tin = time.time()
    bamfile_path = Path(bamfile_path)
    reference_length = int(reference_length)

    paired_reads, reference_length = load_paired_reads(bamfile_path, reference_length, min_mapping_quality)

    # print(paired_reads[0][0].pos, paired_reads[0][1].pos, reference_length)

    snps = create_snps_from_aligned_segments(
        paired_reads,
        reference_length,
        region_start=region_start,
        region_end=region_end,
        min_base_quality=min_base_quality,
        verbose=verbose
    )
    snp_nucls = get_unique_nucls(snps)
    print(f'Nucleotides from reads: {snp_nucls}')
    target_snps = SNP.filter(snps, min_minor_coverage=min_minor_coverage)
    print(f'Target SNP count: {len(target_snps)}, {target_snps}')

    major_reference = GenomeReference(snps)
    print(f'Read data time: {round(time.time() - tin)} sec')

    tin = time.time()
    hedges = create_hedges(paired_reads, target_snps, region_start=region_start or 0, verbose=verbose)
    print(f'|HyperEdges| = {len(hedges)}')

    target_hedges = [he for he in hedges if he.frequency >= error_probability * 100]
    print(f'|Target HyperEdges| = {len(target_hedges)}')

    new_target_hedges = init_frequencies(target_hedges)

    dict_dict_hedges = defaultdict(dict)
    for pos_set, h_list in new_target_hedges.items():
        for h in h_list:
            dict_dict_hedges[pos_set][''.join(h.nucls)] = h

    print(dict_dict_hedges)
    # Run algo
    haplo_hedges, metrics_log = algo_merge_hedge_contigs(
        hedges=dict_dict_hedges,
        target_snp_count=len(target_snps),
        error_probability=error_probability,
        verbose=verbose
    )
    print(f'|Master HyperEdges| = {len(haplo_hedges)}')

    long_haplo_hedges = select_hedges_by_length(haplo_hedges, length_thresh=min_contig_length)
    print(f'After absolute length filter: |Master HyperEdges| = {len(long_haplo_hedges)}')

    if len(target_snps) > 0:
        snps_len = (target_snps[-1].position - target_snps[0].position + 1)
        long_haplo_hedges = select_hedges_by_relative_length(
            long_haplo_hedges,
            normalization_length=snps_len,
            relative_length_thresh=min_relative_contig_length
        )
        print(f'After relative length filter: |Master HyperEdges| = {len(long_haplo_hedges)}')

    haplos = padding_contigs_with_major(long_haplo_hedges, major_reference)
    set_frequencies_by_weight_normalization(haplos, long_haplo_hedges)

    if len(haplos) == 0:
        print('Only major haplo')
        major_haplo = Haplotype('major', ''.join(major_reference.nucls), freq=1)
        haplos = [major_haplo]

    haplos = rename_in_frequency_order(haplos)
    print(f'|Haplos| = {len(haplos)}')

    print(f'Algorithm time: {round(time.time() - tin)} sec')

    OUTPUT = Path(output_folder)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_haplos(haplos, fasta_path=OUTPUT / 'itmo_final.fasta')
    # TODO Add SNP output for haplos hedges

    # TODO print metrics (Recall and Precision) with GT


if __name__ == '__main__':
    main()
