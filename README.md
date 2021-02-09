# HyperHaplo. Variant caller.
The tool is programmed to find the haplotypes and percentage of each haplotype.

The main approach is to use all the SNPs from each read, making them the vertices in a *_hyper_* graph. The hyperedges are the connections between different vertices. Each hyperedge should finally be a single haplotype.

# Parameters 
To run the tool use the following command:
```cmd
python run_vc.py --bamfile_path reads.bam
```

#### Essential parameters
```-bam, --bamfile_path```  - a path to a file with reads.

#### Extra parameters
```-out, --output_folder``` - a path to the output folder. By default: current folder.

``` -bq, --min_base_quality ``` - minimal quality of used reads.

``` -mq, --min_mapping_quality ``` - minimal quality of mapping used.

``` -mc, --min_minor_coverage ``` - minimal minor haplotype coverage to differentiate it from the sequencing error.

``` -w, --target_hedge_weight ``` - minimal weight of hyperedges used.

``` -rl, --min_relative_contig_length``` - 

