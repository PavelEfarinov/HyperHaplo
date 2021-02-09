from collections import defaultdict
from typing import Tuple, List, Set, Dict

import pysam
from tqdm import tqdm

from data import SNP, GenomeReference, Haplotype
from hedge import HEdge
from metrics import normalize_freq


def create_hedges(
        reads: List[List[pysam.AlignedSegment]],
        target_snps: List[SNP],
        region_start: int,
        verbose=True
) -> List[HEdge]:
    all_snp_positions = set(snp.position for snp in target_snps)  # ref positions
    snp2genome, genome2snp = SNP.reindex_snp_and_genome_mapping(target_snps)

    non_snp_reads_count = 0
    holed_reads_count = 0
    chimera_paired_read_count = 0

    hedges: List[HEdge] = []
    hedges_weight = defaultdict(int)
    for read in tqdm(reads, desc='Create hedges from reads', disable=not verbose):
        # here still ref positions
        if len(read) == 1:
            positions, nucls = SNP.select_snps_from_single_read(read[0], all_snp_positions, region_start)
            positions, nucls = [positions], [nucls]
        elif len(read) == 2:
            selected_snps = SNP.select_snps_from_paired_read(read, all_snp_positions, region_start)
            if selected_snps is not None:
                positions, nucls = selected_snps
            else:
                chimera_paired_read_count += 1
                continue
        else:
            raise ValueError(f'Read must be single or paired, but given read with {len(read)} parts')

        if any(map(len, positions)):  # any part has SNP
            # here still ref positions
            try:
                hedge = HEdge.build(positions, nucls, snp2genome, genome2snp)
                if hedge is not None:
                    if hash(hedge) not in hedges_weight:
                        hedges.append(hedge)
                    hedges_weight[hash(hedge)] += 1
                else:
                    # TODO handle holes in reads
                    holed_reads_count += 1
            except ValueError as err:
                # TODO fix indels in read
                raise ValueError('Fix indels in read')

        else:
            non_snp_reads_count += 1

    if verbose:
        print(f'Skipped reads without SNP   : {non_snp_reads_count}')
        print(f'Skipped reads with holes    : {holed_reads_count}')
        print(f'Skipped chimera paired reads: {chimera_paired_read_count}')

    for he in hedges:
        he.init_weight(hedges_weight[hash(he)])

    return hedges


def select_hedges_by_length(hedges: List[HEdge], length_thresh: int) -> List[HEdge]:
    return [he for he in hedges if len(he) >= length_thresh]


def select_hedges_by_relative_length(
        hedges: List[HEdge],
        normalization_length: float,
        relative_length_thresh: float) -> List[HEdge]:
    return [hedge
            for hedge in hedges
            if len(hedge) / normalization_length >= relative_length_thresh]


def padding_contigs_with_major(
        contigs: List[HEdge],
        major_ref: GenomeReference
) -> List[Haplotype]:
    haplos = []
    for i, contig in enumerate(contigs, start=1):
        haplo_nucls = list(major_ref.nucls)
        for positions_list, nucls_list in zip(
                contig.get_position_segments(),
                contig.get_nucl_segments(),
        ):
            for pos, nucl in zip(positions_list, nucls_list):
                haplo_nucls[contig.snp2genome[pos]] = nucl

        haplos.append(Haplotype(
            name=f'{i}_hap',
            nucls=''.join(haplo_nucls)
        ))
    return haplos


def set_frequencies_by_weight_normalization(haplos: List[Haplotype], hedges: List[HEdge]):
    assert len(haplos) == len(hedges), 'Must 1-to-1 relationship. '

    # Normalize by hedge length.
    freqs = [
        hedge.weight / (hedge.end_in_genome() - hedge.begin_in_genome() + 1)
        for hedge in hedges
    ]

    # Normalize by total count.
    freqs = normalize_freq(freqs)

    for haplo, freq in zip(haplos, freqs):
        haplo.freq = freq


def rename_in_frequency_order(haplos: List[Haplotype]) -> List[Haplotype]:
    haplos = sorted(haplos, key=lambda h: h.freq, reverse=True)
    for i, haplo in enumerate(haplos, start=1):
        haplo.name = f'hap_{i}'
    return haplos
