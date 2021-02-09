from __future__ import annotations
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set

import pandas as pd
import pysam
from tqdm import tqdm

from segment import Segment


class SNP:
    def __init__(self, position: int, nucls: List[str], coverage: List[str]):
        self.position = position
        self.coverage, self.nucls, = zip(*sorted(zip(coverage, nucls), reverse=True))
        self.total_coverage = sum(coverage)

    @property
    def major_coverage(self) -> float:
        return self.coverage[0] / self.total_coverage

    @property
    def minor_coverage(self) -> float:
        return 1 - self.major_coverage

    def __len__(self) -> int:
        return len(self.nucls)

    def __repr__(self) -> str:
        return f'SNP(pos={self.position}, cov={self.total_coverage},' \
               f' major_cov={self.major_coverage:.4f}, nucls={self.nucls})'

    @staticmethod
    def filter(
            snps: List[SNP],
            min_minor_coverage: float = 0.004
    ) -> List[SNP]:
        return [snp for snp in snps if snp.minor_coverage >= min_minor_coverage]

    @staticmethod
    def reindex_snp_and_genome_mapping(
            snps: List[SNP]
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        snp2genome = dict()
        genome2snp = dict()
        for i, snp in enumerate(snps):
            genome_pos = snp.position
            genome2snp[genome_pos] = i
            snp2genome[i] = genome_pos
        return snp2genome, genome2snp

    @staticmethod
    def del_nucl() -> str:
        return '-'

    @staticmethod
    def process_nucl(nucl: str) -> str:
        # rename deletion nucl from '' to '-'
        if nucl == '':
            nucl = SNP.del_nucl()

        nucl = nucl.upper()
        assert nucl in 'AGCT-N', nucl
        return nucl

    @staticmethod
    def select_snps_from_single_read(
            read: pysam.AlignedSegment,
            snp_positions: Set[int],
            region_start: int
    ) -> Tuple[List[int], List[str]]:
        positions, nucls = [], []
        # TODO use indels
        seq = read.query_sequence
        for read_pos, ref_pos in read.get_aligned_pairs(matches_only=True):
            ref_pos = ref_pos - region_start
            if ref_pos in snp_positions:
                positions.append(ref_pos)
                nucls.append(SNP.process_nucl(seq[read_pos]))
        return positions, nucls

    @staticmethod
    def select_snps_from_paired_read(
            paired_read: List[pysam.AlignedSegment],
            snp_positions: Set[int],
            region_start: int = 0
    ) -> Optional[Tuple[List[List[int]], List[List[str]]]]:
        assert len(paired_read) == 2, 'Paired read has 2 parts'

        positions1, nucls1 = SNP.select_snps_from_single_read(paired_read[0], snp_positions, region_start)
        positions2, nucls2 = SNP.select_snps_from_single_read(paired_read[1], snp_positions, region_start)

        positions = []
        nucls = []
        if len(positions1):
            positions.append(positions1)
            nucls.append(nucls1)
        if len(positions2):
            positions.append(positions2)
            nucls.append(nucls2)

        if len(positions) == 2:
            p1, p2 = positions
            s1 = Segment(p1[0], p1[-1])
            s2 = Segment(p2[0], p2[-1])
            if s1 in s2 or s2 in s1 or s1.has_intersection(s2):
                merge_result = SNP._merge_paired_read_parts(
                    (positions1, nucls1), (positions2, nucls2))

                if merge_result is not None:
                    merged_positions, merged_nucls = merge_result
                    positions = [merged_positions]
                    nucls = [merged_nucls]
                else:
                    # Chimera paired read
                    return None

            elif s1 > s2:
                positions = [p2, p1]
                nucls = [nucls[-1], nucls[0]]

        def _check_order(positions):
            # TODO remove debug _check_order
            from more_itertools import pairwise
            for li, list in enumerate(positions):
                for pp, p in pairwise(list):
                    assert pp < p
            if len(positions) == 2:
                assert positions[0][-1] < positions[1][0], f'{positions[0][-1]} < {positions[1][0]}, s1{s1}, s2{s2}'

        _check_order(positions)

        return positions, nucls

    @staticmethod
    def _merge_paired_read_parts(
            r1: Tuple[List[int], List[str]],
            r2: Tuple[List[int], List[str]]
    ) -> Optional[Tuple[List[int], List[str]]]:
        positions1, nucls1 = r1
        positions2, nucls2 = r2

        positions = []
        nucls = []

        i, j = 0, 0
        while i < len(positions1) and j < len(positions2):
            if positions1[i] < positions2[j]:
                positions.append(positions1[i])
                nucls.append(nucls1[i])
                i += 1
            elif positions1[i] > positions2[j]:
                positions.append(positions2[j])
                nucls.append(nucls2[j])
                j += 1
            else:
                if nucls1[i] != nucls2[j]:
                    # Chimera paired read
                    # TODO split conflict part to different reads
                    # print(f'Warning: conflict overlap of the read parts '
                    #       f'({positions1[i]}, {nucls1[i]}) != ({positions2[j]}, {nucls2[j]})')
                    # print(list(zip(positions1, nucls1)))
                    # print(list(zip(positions2, nucls2)))
                    return None

                # assert nucls1[i] == nucls2[j], \
                #     f'Nucls at the same positions on the different strings' \
                #     f' are supposed to be the same but' \
                #     f'({positions1[i]}, {nucls1[i]}) != ({positions2[j]}, {nucls2[j]})'
                positions.append(positions1[i])
                nucls.append(nucls1[i])
                i += 1
                j += 1
        return positions, nucls


class GenomeReference:
    def __init__(
            self,
            snps: List[SNP] = None,
            reference_nucls: str = None,
            coverage: List[int] = None
    ):
        if snps is not None:
            assert all(map(len, snps)), 'Empty reference'

            self.nucls = [snp.nucls[0] for snp in snps]
            self.coverage = [snp.coverage for snp in snps]
        elif reference_nucls is not None and coverage is not None:
            assert len(reference_nucls), 'Empty reference'
            assert len(reference_nucls) == len(coverage)

            self.nucls = reference_nucls.upper()
            self.coverage = coverage
        else:
            raise ValueError('Not enough params')

    def __repr__(self) -> str:
        nucl_show_size = 10
        nucls_prefix = ''.join(self.nucls[:nucl_show_size])
        nucls_suffix = ''.join(self.nucls[-nucl_show_size:])
        return f'GenomeReference(' \
               f'len={len(self.nucls)}, ' \
               f'nucls={nucls_prefix}...{nucls_suffix})'


class Haplotype:
    def __init__(self, name: str, nucls: str, freq: Optional[float] = None):
        self.name = name
        self.freq = freq
        self.nucls = nucls

    def __eq__(self, other: Haplotype) -> bool:
        freq_eq = (type(self.freq) == type(other.freq) and
                   (self.freq is None or abs(self.freq - other.freq) < 1e-5))
        return freq_eq and self.name == other.name and self.nucls == other.nucls

    def __repr__(self) -> str:
        nucl_show_size = 10
        nucls_prefix = ''.join(self.nucls[:nucl_show_size])
        nucls_suffix = ''.join(self.nucls[-nucl_show_size:])
        return f'Haplotype(' \
               f'name={self.name:<5}, ' \
               f'len={len(self.nucls)}, ' \
               f'freq={round(self.freq if self.freq else 0, 3):.3f}, ' \
               f'nucls={nucls_prefix}...{nucls_suffix})'


def load_single_reads(bamfile_path: Path) -> List[pysam.AlignedSegment]:
    with pysam.AlignmentFile(bamfile_path, 'rb') as bf:
        reads = [read for read in tqdm(bf.fetch(), desc='Load reads')]
    return reads


def load_paired_reads(
        bamfile_path: Path,
        reference_name: str,
        min_mapping_quality: int = 20,
        verbose: bool = True
) -> Tuple[List[List[pysam.AlignedSegment]], int]:
    not_proper_pair_reads = 0
    supplementary_read_count = 0
    low_mapping_quality_reads = 0
    unmapped_read_count = 0
    qcfail_count = 0

    reads_dict = defaultdict(list)
    with pysam.AlignmentFile(bamfile_path, 'rb') as bf:
        reference_length = bf.get_reference_length(reference_name)

        read: pysam.AlignedSegment
        for read in tqdm(bf.fetch(), desc='Load reads', disable=not verbose):
            if read.is_unmapped:
                unmapped_read_count += 1
                continue

            if not read.is_proper_pair:
                not_proper_pair_reads += 1
                continue

            if read.is_supplementary:
                supplementary_read_count += 1
                continue

            if read.is_qcfail:
                qcfail_count += 1
                continue

            if read.mapping_quality < min_mapping_quality:
                low_mapping_quality_reads += 1
                continue

            reads_dict[read.query_name].append(read)

    read_pairs = list(reads_dict.values())

    reads_counter = Counter(map(len, read_pairs))
    print(f'Dropped [not proper pair]: {not_proper_pair_reads}')
    print(f'Dropped [supplementary  ]: {supplementary_read_count}')
    print(f'Dropped [unmapped       ]: {unmapped_read_count}')
    print(f'Dropped [qcfail         ]: {qcfail_count}')
    print(f'Dropped [mapping quality]: {low_mapping_quality_reads}')
    print(f'Valid reads: {len(read_pairs)} (single={reads_counter[1]}, paired={reads_counter[2]})')

    assert set(reads_counter.keys()).issubset(
        {1, 2}), f'Allowed read has no more than 2 parts, but found {reads_counter.keys()}'
    return read_pairs, reference_length


def create_snps_from_pileup_columns(
        bamfile_path: Path,
        reference_length: int,
        region_start: int = None,
        region_end: int = None,
        min_base_quality: int = 0,
        min_mapping_quality: int = 0,
        verbose: bool = True
) -> List[SNP]:
    """ Create SNP class for each position in the reference.
    Args:
        bamfile_path: Path to bam file with single and/or paired reads.
        reference_length: Length of the reference genome.
        region_start: 0-based (bam) inclusive index of genome target region start.
        region_end: 0-based (bam) exclusive index of genome target region end.
        min_base_quality:
        min_mapping_quality:

    Returns:
        List[SNP]: List of all alleles within genome target region end.
    """
    if region_start is None and region_end is None:
        region_start = 0
        region_end = reference_length
    print(f'Using reference region: [{region_start}, {region_end})', flush=True)

    out_of_region = 0

    snps = []
    with pysam.AlignmentFile(bamfile_path, 'rb') as bf:
        col: pysam.PileupColumn
        for col in tqdm(bf.pileup(), desc='SNPs from [PileupColumn]', disable=not verbose):
            if not (region_start <= col.reference_pos < region_end):
                out_of_region += 1
                continue

            # TODO use PileupColumn.get_query_sequences(add_indels=True)
            col.set_min_base_quality(min_base_quality)

            nucl_counter = Counter([
                nucl.upper() for nucl, mapping_qual
                in zip(col.get_query_sequences(), col.get_mapping_qualities())
                if mapping_qual >= min_mapping_quality
            ])
            # TODO remove debug wrapper
            try:
                nucl_counter_keys_list = [SNP.process_nucl(nucl) for nucl in nucl_counter.keys()]
                snps.append(SNP(
                    position=col.reference_pos - region_start,
                    nucls=nucl_counter_keys_list,
                    coverage=list(nucl_counter.values())
                ))
            except Exception as err:
                col.set_min_base_quality(0)
                print(f'qual={col.get_mapping_qualities()}')
                print(f'seq={col.get_query_sequences()}')
                print(f'position= {col.reference_pos}')
                print(f'nucls= {list(nucl_counter.keys())}')
                print(f'coverage= {list(nucl_counter.values())}')
                print(str(err))
                raise err
    return snps


def create_snps_from_aligned_segments(
        reads: List[List[pysam.AlignedSegment]],
        reference_length: int,
        region_start: int = None,
        region_end: int = None,
        min_base_quality: int = 0,
        verbose: bool = True
) -> List[SNP]:
    """ Create SNP class for each position in the reference based on read alignments.
    Args:
        reads: List with single and/or paired reads.
        reference_length: Length of the reference genome.
        region_start: 0-based (bam) inclusive index of genome target region start.
        region_end: 0-based (bam) exclusive index of genome target region end.
        min_base_quality:
        verbose: Mainly to show tqdm status bar.

    Returns:
        List[SNP]: List of all alleles within genome target region end.
    """
    if region_start is None and region_end is None:
        region_start = 0
        region_end = reference_length
    print(f'Using reference region: [{region_start}, {region_end})', flush=True)

    out_of_region = 0
    fail_quality = 0
    snp_counter = defaultdict(lambda: defaultdict(int))  # snp_counter[pos][nucl]

    out_of_ref_positions = set()
    ref_positions = set()
    for read in tqdm(reads, desc='SNPs from [AlignedSegment]', disable=not verbose):
        for read_part in read:
            seq = read_part.query_sequence
            quals = read_part.query_alignment_qualities
            for read_pos, ref_pos in read_part.get_aligned_pairs(matches_only=True):
                if not (region_start <= ref_pos < region_end):
                    out_of_region += 1
                    out_of_ref_positions.add(ref_pos)
                    continue

                refined_read_pos = read_pos - read_part.query_alignment_start
                qual = quals[refined_read_pos]
                nucl = seq[read_pos]

                if qual < min_base_quality:
                    fail_quality += 1
                    continue

                ref_positions.add(ref_pos)
                snp_counter[ref_pos - region_start][SNP.process_nucl(nucl)] += 1

    for i in range(region_start, region_end):
        pos = i - region_start
        if pos not in snp_counter:
            snp_counter[pos][SNP.del_nucl()] += 1

    snps = []
    for ref_pos, nucl_counter in sorted(snp_counter.items(), key=lambda x: x[0]):
        snps.append(SNP(
            position=ref_pos,
            nucls=list(nucl_counter.keys()),
            coverage=list(nucl_counter.values())
        ))

    ref_positions = sorted(list(ref_positions))
    out_of_ref_positions = sorted(list(out_of_ref_positions))
    print(f'Reference position covered [{len(ref_positions)}]:'
          f'{ref_positions[:5]}..{ref_positions[-5:]} ')
    print(f'Reference position out of target region covered [{len(out_of_ref_positions)}]:'
          f'{out_of_ref_positions[:5]}..{out_of_ref_positions[-5:]} ')
    print(f'Positions are out of region: {out_of_region}')
    print(f'Fail  quality: {fail_quality}')
    return snps


def get_unique_nucls(snps: List[SNP]) -> Counter:
    return Counter(chain.from_iterable(snp.nucls for snp in snps))


def get_minor_snps_coverage(snps, minor_order=2) -> pd.Series:
    return pd.Series([
        snp.coverage[minor_order - 1]
        for snp in snps if len(snp.nucls) >= minor_order
    ])


def is_snp(snp, coverage_thresh=15):
    minor2 = snp.most_common()[1]
    minor2_coverage = minor2[1]
    return minor2_coverage >= coverage_thresh


def get_snp_indeces(indexed_snp_positions, snp_coverage_thresh):
    snp_indeces = [i for i, snp in indexed_snp_positions if is_snp(snp, snp_coverage_thresh)]
    return snp_indeces
