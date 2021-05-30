# HyperHaplo. Variant caller.
Many viruses such as human immunodeficiency virus and hepatitis C virus have a very high mutation rate, which results in several haplotypes, closely related genomic variants, coexisting in the carrier's organism. Thus, identifying the haplotype  sequences and their percentage in the mix is a very important task for biologists and doctors in order to create a more efficient treatment.  

We present HyperHaplo, a novel method for haplotype variant calling from the NGS data which is based on the hypergraph approach. We build an allele graph, where a vertex is a single nucleotide polymorphism (SNP) and a hyperedge is a set of vertices. A hyperedge connects vertices if they appear in one read. After creating starting hyperedges, we run the merging algorithm. According to the algorithm, we merge two hyperedges if they intersect and do not conflict. The algorithm always tends to assemble the most frequent haplotype in a mix so the most frequent haplotypes are assembled mostly correct. We consider that a haplotype is assembled as soon as a hyperedge contains all the SNPs. 

We used the HIV lab mix dataset to test the algorithm’s behavior via the UniFrac and mean minimum distance metrics. The dataset is an Illumina sequencing of a mix of five HIV strains. The UniFrac metric is the Wasserstain distance and the mean minimum distance is a metric which looks for each predicted haplotype the closest real strain and calculates the average value over all the predicted haplotypes. Our method outperforms the recent ITMO hypergraph approach, but it is unfortunately worse than the CliqueSNV2 approach. 

# Setup
You need the Unix based system with some packages preinstalled.
The best way is to make a new conda environment via the following commands.
```
conda env create -n ngs
conda activate ngs
conda install -c bioconda pysam
conda install -c condaforge networkx, graphviz, scipy
```

The other way is to install the packages using the pip environment file:
```
pip install -f environment.yml
```
# Parameters 
To run the tool use the following command:
```cmd
python run_vc.py --bamfile_path reads.bam --reference_length XXXX
```

#### Essential parameters
```-bam, --bamfile_path```  - a path to a file with reads.

```--reference_length``` - reference genome length.

#### Extra parameters
```-out, --output_folder``` - a path to the output folder. By default: current folder.

``` -bq, --min_base_quality ``` - minimal quality of used reads.

``` -mq, --min_mapping_quality ``` - minimal quality of mapping used.

``` -mc, --min_minor_coverage ``` - minimal minor haplotype coverage to differentiate it from the sequencing error.

``` -w, --target_hedge_weight ``` - minimal weight of hyperedges used.

``` -rs, --region_start ``` - the start of interesting region in reference.

``` -re, --region_end ``` - the end of interesting region in reference.

``` -err, --error_probability ``` - error probability in reads.

``` -out, --output_folder ``` - output folder to store results.

``` -log, --verbose ``` - print tqdm status bars in console.

# Example
Firstly, download the reads for lab mix via [the github link](https://github.com/cbg-ethz/5-virus-mix).
  
Afterwards you need to align the reads to the reference genome (XHB2 strain was chosen as a reference). 
You can do it by running the ```scripts/data_generator/hiv_alignment.sh``` script.

To test the tool using the HIV lab mix dataset use the command: 
```
python run_vc.py --bamfile_path data/hiv-lab-mix/alignment_sorted.bam
-out data/hiv-lab-mix/itmo_final_0.95
--reference_length 9720
-rs 2253
-re 3390
-err 0.01
```
The results are stored in the specified folder in fasta format (itmo_final.fasta). 
#Generated data testing
You can also test the tool using the generated data. For data generation you should use the```scripts/data_generator/data_generator.py``` script.
You can change the haplotype count and SNP positions count in the main function. 
After running the script the generated haplotypes are stored in ```generated_haplos``` folder.
You should run the ```scripts/data_generator/read_generator.sh``` script to run the iss reads generation tool and align the generated reads to the reference.
The following pipeline if the same as for the HIV data.

#Analysis
There is a script to measure main metrics for the generated haplotypes. 
It's located in the test folder: ```test/eval_hamming_distance.py```
There are two strings in the beginning of the script which specify the ground truth and predicted haplotypes paths.
You need to change them according to your requirements.

# Helpful links
* [HIV lab mix data github](https://github.com/cbg-ethz/5-virus-mix)
* [InSilicoSeq (iss tool) documentation](https://insilicoseq.readthedocs.io/en/latest/iss/model.html)
