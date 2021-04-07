cd ../../generated_haplos/sequence_haplo_25_snp_50
iss generate --n_reads 400000 --mode perfect --genomes sequences.fasta --abundance_file abundance.txt --output reads
bwa index ../../data/fasta_sequences/sequence.fasta
bwa mem ../../data/fasta_sequences/sequence.fasta reads_R1.fastq reads_R2.fastq > alignment.sam
samtools view -S -b alignment.sam > alignment.bam
samtools flagstat alignment.bam
samtools sort alignment.bam > alignment_sorted.bam
samtools index alignment_sorted.bam