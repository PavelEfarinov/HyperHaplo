cd ../../data/hiv-lab-mix
bwa index ref_HXB2.fasta
bwa mem ref_HXB2.fasta sra_data.fasta > alignment.sam
samtools view -S -b alignment.sam > alignment.bam
samtools flagstat alignment.bam
samtools sort alignment.bam > alignment_sorted.bam
samtools index alignment_sorted.bam