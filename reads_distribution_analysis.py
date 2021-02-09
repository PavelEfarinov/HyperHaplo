import time
import click

from data import *
from core import *
from algo import *
from metrics import *


def main():
    tin = time.time()
    bamfile_path = Path('test_data/u5e-8_s2000_Ne500/sequences001/final.bam')

    paired_reads, reference_length = load_paired_reads(bamfile_path, 'pol_HXB2', 0)
    snps = create_snps_from_aligned_segments(paired_reads, reference_length)

    snp_nucls = get_unique_nucls(snps)
    print(f'Nucleotides from reads: {snp_nucls}')

    target_snps = SNP.filter(snps, min_minor_coverage=0.006)
    print(target_snps)
    print(f'Target SNP count: {len(target_snps)}')
    snp_positions = set(snp.position for snp in target_snps)
    print(f'Target SNP positions{snp_positions}')

    # cur_read = paired_reads[0][0]
    # print(cur_read)
    # print(cur_read.pos)
    # print(cur_read.seq)
    # print(cur_read.aend)
    # print(cur_read.isize)
    # print(cur_read.positions)
    # cur_read = paired_reads[100][1]
    # print(cur_read)
    # print(cur_read.pos)
    # print(cur_read.seq)
    # print(cur_read.aend)
    # print(cur_read.isize)
    # print(cur_read.positions)

    read_to_snp_count = defaultdict(list)
    for read in paired_reads:
        # print(read)
        result = SNP.select_snps_from_paired_read(read, snp_positions, read[0].pos)
        if result is not None:
            positions, nucls = result
            if len(positions) > 0:
                positions_count = len(set(x[0] for x in positions))
                read_to_snp_count[read[0].pos].append(positions_count)
            else:
                read_to_snp_count[read[0].pos].append(0)
    data = pd.DataFrame(
        {'average': [sum(x) / len(x) for x in read_to_snp_count.values()]})
    print(data)
    print('generated data')
    data.plot()
    # print(positions)

    major_reference = GenomeReference(snps)
    print(f'Read data time: {round(time.time() - tin)} sec')


if __name__ == '__main__':
    main()
